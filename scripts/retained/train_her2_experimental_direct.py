#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import traceback
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset


class NpyCoordDataset(Dataset):
    def __init__(
        self,
        image_path: Path,
        coord_path: Path,
        max_atoms: int = 1489,
        sample_weights: np.ndarray | None = None,
        indices: np.ndarray | None = None,
    ):
        self.images = np.load(image_path, mmap_mode="r")
        self.coords = np.load(coord_path, mmap_mode="r")
        self.max_atoms = max_atoms
        if self.images.shape[0] != self.coords.shape[0]:
            raise ValueError(f"Count mismatch: {image_path} has {self.images.shape[0]} rows, {coord_path} has {self.coords.shape[0]}")
        if indices is None:
            self.indices = None
        else:
            idx = np.asarray(indices, dtype=np.int64)
            if idx.ndim != 1:
                raise ValueError("indices must be a 1D array")
            if idx.size and (idx.min() < 0 or idx.max() >= self.images.shape[0]):
                raise ValueError(f"indices out of range for {image_path}: n={self.images.shape[0]}")
            self.indices = idx
        if sample_weights is None:
            self.sample_weights = None
        else:
            weights = np.asarray(sample_weights, dtype=np.float32)
            expected = len(self) if self.indices is not None else self.images.shape[0]
            if weights.shape[0] != expected:
                raise ValueError(f"Weight count mismatch: weights={weights.shape[0]} expected={expected}")
            self.sample_weights = weights

    def __len__(self) -> int:
        return int(self.indices.shape[0]) if self.indices is not None else int(self.images.shape[0])

    def __getitem__(self, idx: int):
        source_idx = int(self.indices[idx]) if self.indices is not None else int(idx)
        image = np.array(self.images[source_idx], dtype=np.float32, copy=True)
        coords = np.array(self.coords[source_idx], dtype=np.float32, copy=True).reshape(self.max_atoms, 3)
        mask = np.ones(self.max_atoms, dtype=np.float32)
        item = {
            "image": torch.from_numpy(image).unsqueeze(0).float(),
            "coords": torch.from_numpy(coords).permute(1, 0).float(),
            "mask": torch.from_numpy(mask).float(),
            "id": source_idx,
        }
        if self.sample_weights is not None:
            item["sample_weight"] = torch.tensor(float(self.sample_weights[idx]), dtype=torch.float32)
        return item


def collate_fn(batch):
    out = {
        "image": torch.stack([b["image"] for b in batch]),
        "coords": torch.stack([b["coords"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "id": [b["id"] for b in batch],
    }
    if "sample_weight" in batch[0]:
        out["sample_weight"] = torch.stack([b["sample_weight"] for b in batch])
    return out


class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.dropout(h)
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.skip(x), inplace=True)


class StrongCNNToCoords(nn.Module):
    def __init__(
        self,
        max_atoms: int,
        hidden_dim: int = 2048,
        mlp_layers: int = 4,
        dropout: float = 0.15,
        bottleneck_dim: int = 512,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layers = nn.Sequential(
            ResidualConvBlock(32, 64, stride=2, dropout=dropout * 0.5),
            ResidualConvBlock(64, 128, stride=2, dropout=dropout * 0.5),
            ResidualConvBlock(128, 256, stride=2, dropout=dropout),
            ResidualConvBlock(256, 256, stride=1, dropout=dropout),
            ResidualConvBlock(256, 512, stride=2, dropout=dropout),
            ResidualConvBlock(512, 512, stride=1, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck = nn.Identity() if bottleneck_dim == 512 else nn.Sequential(
            nn.Linear(512, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        mlp: list[nn.Module] = []
        in_dim = bottleneck_dim
        for _ in range(max(mlp_layers - 1, 1)):
            mlp.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)])
            in_dim = hidden_dim
        mlp.append(nn.Linear(in_dim, max_atoms * 3))
        self.head = nn.Sequential(*mlp)
        self.max_atoms = max_atoms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.layers(h)
        h = self.pool(h).flatten(1)
        h = self.bottleneck(h)
        out = self.head(h)
        return out.view(x.size(0), self.max_atoms, 3).transpose(1, 2)


def huber_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, beta: float) -> torch.Tensor:
    per = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    per = per * mask.unsqueeze(1)
    denom = mask.sum(dim=1).clamp(min=1) * pred.size(1)
    return (per.sum(dim=[1, 2]) / denom).mean()


def weighted_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    per = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    per = per * mask.unsqueeze(1)
    denom = mask.sum(dim=1).clamp(min=1) * pred.size(1)
    per_sample = per.sum(dim=[1, 2]) / denom
    if sample_weight is None:
        return per_sample.mean()
    weights = sample_weight.to(device=per_sample.device, dtype=per_sample.dtype).clamp(min=0)
    return (per_sample * weights).sum() / weights.sum().clamp(min=1.0e-8)


def load_train_weights(args, expected_len: int) -> np.ndarray | None:
    if not args.train_confidence_csv:
        return None
    rows = []
    with Path(args.train_confidence_csv).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if args.train_confidence_column not in (reader.fieldnames or []):
            raise ValueError(
                f"Column {args.train_confidence_column!r} not found in {args.train_confidence_csv}; "
                f"available={reader.fieldnames}"
            )
        for row in reader:
            rows.append(row)
    if len(rows) != expected_len:
        raise ValueError(f"Confidence row count mismatch: csv={len(rows)} expected={expected_len}")
    weights = np.asarray([float(row[args.train_confidence_column]) for row in rows], dtype=np.float32)
    if args.train_confidence_floor is not None:
        weights = np.maximum(weights, float(args.train_confidence_floor))
    if args.train_confidence_power != 1.0:
        weights = np.power(weights, float(args.train_confidence_power)).astype(np.float32)
    if args.train_weight_normalize_mean:
        mean = float(weights.mean())
        if mean <= 0:
            raise ValueError("Cannot mean-normalize non-positive confidence weights")
        weights = weights / mean
    return weights.astype(np.float32)


def load_indices(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p).astype(np.int64)
    return np.loadtxt(p, dtype=np.int64, ndmin=1)


def fit_pca_projector(coord_path: Path, max_atoms: int, n_components: int, max_samples: int, seed: int):
    coords = np.load(coord_path, mmap_mode="r")
    n = int(coords.shape[0])
    rng = np.random.default_rng(seed)
    if max_samples > 0 and n > max_samples:
        idx = np.sort(rng.choice(n, size=max_samples, replace=False))
        fit_coords = np.asarray(coords[idx], dtype=np.float32)
    else:
        fit_coords = np.asarray(coords, dtype=np.float32)
    fit_coords = fit_coords.reshape(fit_coords.shape[0], max_atoms * 3)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    pca.fit(fit_coords)
    scale = np.sqrt(np.maximum(pca.explained_variance_, 1e-8)).astype(np.float32)
    return {
        "mean": pca.mean_.astype(np.float32),
        "components": pca.components_.astype(np.float32),
        "scale": scale,
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
        "fit_count": int(fit_coords.shape[0]),
    }


def pca_scores(pred: torch.Tensor, target: torch.Tensor, mean: torch.Tensor, components: torch.Tensor, scale: torch.Tensor):
    pred_flat = pred.transpose(1, 2).reshape(pred.size(0), -1)
    target_flat = target.transpose(1, 2).reshape(target.size(0), -1)
    pred_score = ((pred_flat - mean) @ components.t()) / scale
    target_score = ((target_flat - mean) @ components.t()) / scale
    return pred_score, target_score


def pca_score_loss(pred: torch.Tensor, target: torch.Tensor, mean: torch.Tensor, components: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    pred_score, target_score = pca_scores(pred, target, mean, components, scale)
    return F.mse_loss(pred_score, target_score)


def pca_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mean: torch.Tensor,
    components: torch.Tensor,
    scale: torch.Tensor,
    margin: float,
    hard_k: int,
    duplicate_eps: float,
    local_radius: float,
) -> torch.Tensor:
    pred_score, target_score = pca_scores(pred, target, mean, components, scale)
    batch_size = pred_score.size(0)
    if batch_size < 2 or hard_k <= 0:
        return pred_score.new_zeros(())

    pred_to_target = torch.cdist(pred_score, target_score, p=2)
    target_to_target = torch.cdist(target_score, target_score, p=2)
    pos = pred_to_target.diagonal().unsqueeze(1)

    valid = ~torch.eye(batch_size, dtype=torch.bool, device=pred_score.device)
    valid = valid & (target_to_target > duplicate_eps)
    if local_radius > 0:
        local_valid = valid & (target_to_target <= local_radius)
        has_local = local_valid.any(dim=1, keepdim=True)
        valid = torch.where(has_local, local_valid, valid)
    if not valid.any():
        return pred_score.new_zeros(())

    masked = pred_to_target.masked_fill(~valid, float("inf"))
    k = min(int(hard_k), max(1, batch_size - 1))
    hard_neg, _ = torch.topk(masked, k=k, largest=False, dim=1)
    finite = torch.isfinite(hard_neg)
    if not finite.any():
        return pred_score.new_zeros(())
    losses = F.relu(float(margin) + pos.expand_as(hard_neg) - hard_neg)
    return losses[finite].mean()


def compute_raw_rmsd(pred_sample: torch.Tensor, target_sample: torch.Tensor, mask_sample: torch.Tensor, eps: float = 1e-8):
    valid = mask_sample > 0.5
    diff = pred_sample[:, valid] - target_sample[:, valid]
    return torch.sqrt(diff.pow(2).sum(dim=0).mean() + eps)


def kabsch_align(pred_sample: torch.Tensor, target_sample: torch.Tensor, mask_sample: torch.Tensor, eps: float = 1e-8):
    valid = mask_sample > 0.5
    P = pred_sample[:, valid].transpose(0, 1)
    T = target_sample[:, valid].transpose(0, 1)
    n_valid = P.shape[0]
    if n_valid < 3:
        return compute_raw_rmsd(pred_sample, target_sample, mask_sample, eps=eps)
    mu_P = P.mean(dim=0, keepdim=True)
    mu_T = T.mean(dim=0, keepdim=True)
    P_center = P - mu_P
    T_center = T - mu_T
    cov = P_center.transpose(0, 1) @ T_center / float(n_valid)
    U, _, Vh = torch.linalg.svd(cov)
    sign = -1.0 if torch.det(U @ Vh) < 0 else 1.0
    diag = torch.ones(3, device=pred_sample.device, dtype=pred_sample.dtype)
    diag[-1] = sign
    rot = U @ torch.diag(diag) @ Vh
    P_aligned = P_center @ rot + mu_T
    diff = P_aligned - T
    return torch.sqrt(diff.pow(2).sum(dim=1).mean() + eps)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_train(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_indices = load_indices(args.train_indices)
    val_indices = load_indices(args.val_indices)
    test_indices = load_indices(args.test_indices)
    expected_train_len = int(train_indices.shape[0]) if train_indices is not None else np.load(Path(args.x_train), mmap_mode="r").shape[0]
    train_weights = load_train_weights(args, expected_train_len)
    ds_train = NpyCoordDataset(Path(args.x_train), Path(args.coords_train), args.max_atoms, sample_weights=train_weights, indices=train_indices)
    ds_val = NpyCoordDataset(Path(args.x_val), Path(args.coords_val), args.max_atoms, indices=val_indices)
    ds_test = NpyCoordDataset(Path(args.x_test), Path(args.coords_test), args.max_atoms, indices=test_indices)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StrongCNNToCoords(
        args.max_atoms,
        hidden_dim=args.hidden_dim,
        mlp_layers=args.mlp_layers,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
    ).to(device)
    init_epoch = 0
    init_val = None
    partial_loaded = None
    if args.init_checkpoint:
        state = torch.load(args.init_checkpoint, map_location=device)
        loaded_state = state["model_state"] if "model_state" in state else state
        if args.init_partial_matching:
            model_state = model.state_dict()
            compatible = {
                key: value
                for key, value in loaded_state.items()
                if key in model_state and model_state[key].shape == value.shape
            }
            skipped = sorted(set(loaded_state.keys()) - set(compatible.keys()))
            model_state.update(compatible)
            model.load_state_dict(model_state)
            partial_loaded = {"loaded": len(compatible), "skipped": len(skipped), "skipped_keys": skipped[:20]}
        else:
            model.load_state_dict(loaded_state)
        init_epoch = int(state.get("epoch", 0)) if isinstance(state, dict) else 0
        init_val = float(state.get("val_huber")) if isinstance(state, dict) and state.get("val_huber") is not None else None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs), eta_min=args.lr_min)

    pca_tensors = None
    if args.pca_score_weight > 0 or args.rank_weight > 0:
        pca_state = fit_pca_projector(
            Path(args.coords_train),
            args.max_atoms,
            args.pca_components,
            args.pca_fit_samples,
            args.seed,
        )
        np.savez(
            out_dir / "pca_score_projector.npz",
            mean=pca_state["mean"],
            components=pca_state["components"],
            scale=pca_state["scale"],
            explained_variance_ratio=pca_state["explained_variance_ratio"],
            fit_count=np.array([pca_state["fit_count"]], dtype=np.int32),
        )
        pca_tensors = {
            "mean": torch.from_numpy(pca_state["mean"]).to(device),
            "components": torch.from_numpy(pca_state["components"]).to(device),
            "scale": torch.from_numpy(pca_state["scale"]).to(device),
        }

    train_hist, val_hist = [], []
    best_val = float("inf")
    best_epoch = -1
    stale_epochs = 0
    log_path = out_dir / "training.log"
    with log_path.open("w") as log:
        log.write(f"[{datetime.now().isoformat()}] Training start\n")
        log.write(f"PID: {os.getpid()}\n")
        log.write(f"Dataset sizes -> train: {len(ds_train)}  val: {len(ds_val)}  test: {len(ds_test)}\n")
        log.write(json.dumps(vars(args), indent=2, default=str) + "\n")
        if train_weights is not None:
            log.write(
                "Confidence weighting enabled: "
                f"csv={args.train_confidence_csv} column={args.train_confidence_column} "
                f"floor={args.train_confidence_floor} power={args.train_confidence_power} "
                f"normalize_mean={args.train_weight_normalize_mean} "
                f"mean={float(train_weights.mean()):.6f} min={float(train_weights.min()):.6f} "
                f"p10={float(np.quantile(train_weights, 0.10)):.6f} "
                f"p50={float(np.quantile(train_weights, 0.50)):.6f} "
                f"p90={float(np.quantile(train_weights, 0.90)):.6f} "
                f"max={float(train_weights.max()):.6f}\n"
            )
        if pca_tensors is not None:
            log.write(
                f"PCA score loss enabled: components={args.pca_components}, "
                f"weight={args.pca_score_weight}, fit_samples={pca_state['fit_count']}\n"
            )
            if args.rank_weight > 0:
                log.write(
                    f"PCA rank loss enabled: weight={args.rank_weight}, "
                    f"margin={args.rank_margin}, hard_k={args.rank_hard_k}, "
                    f"duplicate_eps={args.rank_duplicate_eps}, local_radius={args.rank_local_radius}\n"
                )
        if args.init_checkpoint:
            log.write(
                f"Initialized model weights from {args.init_checkpoint}"
                f" (checkpoint_epoch={init_epoch}"
                + (f", checkpoint_val_huber={init_val:.4f}" if init_val is not None else "")
                + ")\n"
            )
            if partial_loaded is not None:
                log.write(
                    "Partial checkpoint loading enabled: "
                    f"loaded_keys={partial_loaded['loaded']} skipped_keys={partial_loaded['skipped']} "
                    f"first_skipped={partial_loaded['skipped_keys']}\n"
                )
        log.flush()
        try:
            for epoch in range(1, args.epochs + 1):
                start = time.time()
                model.train()
                train_losses = []
                log.write(f"Epoch {epoch}: train loop start\n")
                log.flush()
                for batch_idx, batch in enumerate(dl_train, start=1):
                    image = batch["image"].to(device)
                    target = batch["coords"].to(device)
                    mask = batch["mask"].to(device)
                    sample_weight = batch.get("sample_weight")
                    if sample_weight is not None:
                        sample_weight = sample_weight.to(device)
                    pred = model(image)
                    coord_loss = weighted_huber_loss(pred, target, mask, beta=args.huber_beta, sample_weight=sample_weight)
                    loss = coord_loss
                    if pca_tensors is not None:
                        if args.pca_score_weight > 0:
                            score_loss = pca_score_loss(pred, target, pca_tensors["mean"], pca_tensors["components"], pca_tensors["scale"])
                            loss = loss + args.pca_score_weight * score_loss
                        if args.rank_weight > 0:
                            rank_loss = pca_rank_loss(
                                pred,
                                target,
                                pca_tensors["mean"],
                                pca_tensors["components"],
                                pca_tensors["scale"],
                                args.rank_margin,
                                args.rank_hard_k,
                                args.rank_duplicate_eps,
                                args.rank_local_radius,
                            )
                            loss = loss + args.rank_weight * rank_loss
                    opt.zero_grad()
                    loss.backward()
                    if args.grad_clip is not None:
                        clip_grad_norm_(model.parameters(), args.grad_clip)
                    opt.step()
                    train_losses.append(float(loss.item()))
                    if args.log_every_batches > 0 and (
                        batch_idx == 1 or batch_idx % args.log_every_batches == 0
                    ):
                        log.write(
                            f"Epoch {epoch} batch {batch_idx}/{len(dl_train)} "
                            f"loss={train_losses[-1]:.4f} elapsed={time.time()-start:.1f}s\n"
                        )
                        log.flush()
                    if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                        break

                model.eval()
                val_losses = []
                log.write(f"Epoch {epoch}: val loop start\n")
                log.flush()
                with torch.no_grad():
                    for val_batch_idx, batch in enumerate(dl_val, start=1):
                        image = batch["image"].to(device)
                        target = batch["coords"].to(device)
                        mask = batch["mask"].to(device)
                        pred = model(image)
                        coord_loss = huber_loss(pred, target, mask, beta=args.huber_beta)
                        loss = coord_loss
                        if pca_tensors is not None:
                            if args.pca_score_weight > 0:
                                score_loss = pca_score_loss(pred, target, pca_tensors["mean"], pca_tensors["components"], pca_tensors["scale"])
                                loss = loss + args.pca_score_weight * score_loss
                            if args.rank_weight > 0:
                                rank_loss = pca_rank_loss(
                                    pred,
                                    target,
                                    pca_tensors["mean"],
                                    pca_tensors["components"],
                                    pca_tensors["scale"],
                                    args.rank_margin,
                                    args.rank_hard_k,
                                    args.rank_duplicate_eps,
                                    args.rank_local_radius,
                                )
                                loss = loss + args.rank_weight * rank_loss
                        val_losses.append(float(loss.item()))
                        if args.max_val_batches > 0 and val_batch_idx >= args.max_val_batches:
                            break

                train_loss = float(np.mean(train_losses))
                val_loss = float(np.mean(val_losses))
                train_hist.append(train_loss)
                val_hist.append(val_loss)
                msg = f"Epoch {epoch}/{args.epochs}  train_huber={train_loss:.4f}  val_huber={val_loss:.4f}  lr={opt.param_groups[0]['lr']:.2e}  epoch_time={time.time()-start:.2f}s"
                print(msg, flush=True)
                log.write(msg + "\n")
                improved = val_loss < (best_val - args.early_stop_min_delta)
                if improved:
                    best_val = val_loss
                    best_epoch = epoch
                    stale_epochs = 0
                    torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_huber": best_val}, out_dir / "cnn_best.pt")
                    log.write(f"New best checkpoint at epoch {epoch} with val_huber={best_val:.4f}\n")
                else:
                    stale_epochs += 1
                if epoch % args.save_every == 0 or epoch == args.epochs:
                    torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_huber": val_loss}, out_dir / f"cnn_epoch{epoch}.pt")
                sched.step()
                log.flush()
                if (
                    args.early_stop_patience is not None
                    and epoch >= args.min_epochs
                    and stale_epochs >= args.early_stop_patience
                ):
                    msg = (
                        f"Early stopping at epoch {epoch}; best_epoch={best_epoch}, "
                        f"best_val_huber={best_val:.4f}, stale_epochs={stale_epochs}"
                    )
                    print(msg, flush=True)
                    log.write(msg + "\n")
                    break
            log.write(f"Training complete. Best epoch: {best_epoch}  best val_huber={best_val:.4f}\n")
        except Exception:
            tb = traceback.format_exc()
            log.write("Training failed with exception:\n")
            log.write(tb)
            log.flush()
            print(tb, file=sys.stderr, flush=True)
            raise

    np.savetxt(out_dir / "train_huber.txt", np.asarray(train_hist))
    np.savetxt(out_dir / "val_huber.txt", np.asarray(val_hist))
    if train_weights is not None:
        np.savetxt(out_dir / "train_confidence_applied.txt", train_weights)
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(train_hist) + 1), train_hist, label="train")
    plt.plot(range(1, len(val_hist) + 1), val_hist, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Huber Loss")
    plt.title("Train vs Val HUBER")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_train_val.png", dpi=200)
    plt.savefig(out_dir / "loss_train_val.svg")


def run_infer(args):
    out_dir = Path(args.out_dir)
    ds = NpyCoordDataset(Path(args.x_path), Path(args.coords_path), args.max_atoms, indices=load_indices(args.indices))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StrongCNNToCoords(
        args.max_atoms,
        hidden_dim=args.hidden_dim,
        mlp_layers=args.mlp_layers,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.eval()
    rows = []
    saved_preds = [] if args.save_pred_coords else None
    saved_ids = [] if args.save_pred_coords else None
    with torch.no_grad():
        for batch in dl:
            image = batch["image"].to(device)
            target = batch["coords"].to(device)
            mask = batch["mask"].to(device)
            pred = model(image)
            if saved_preds is not None and saved_ids is not None:
                saved_preds.append(pred.transpose(1, 2).detach().cpu().numpy().astype(np.float32))
                saved_ids.extend(int(x) for x in batch["id"])
            for b in range(pred.size(0)):
                rmsd = kabsch_align(pred[b], target[b], mask[b]) if args.align_infer else compute_raw_rmsd(pred[b], target[b], mask[b])
                rows.append((batch["id"][b], float(rmsd.item()), args.max_atoms))
    rows.sort(key=lambda x: x[0])
    rmsd_vals = np.array([r[1] for r in rows], dtype=np.float32)
    with (out_dir / f"inference_{args.split_name}_rmsd.csv").open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["id", "rmsd", "n_atoms"])
        w.writerows(rows)
    summary = {
        "split": args.split_name,
        "count": int(len(rows)),
        "mean_rmsd": float(rmsd_vals.mean()),
        "median_rmsd": float(np.median(rmsd_vals)),
        "min_rmsd": float(rmsd_vals.min()),
        "max_rmsd": float(rmsd_vals.max()),
    }
    (out_dir / f"inference_{args.split_name}_summary.json").write_text(json.dumps(summary, indent=2))
    if saved_preds is not None and saved_ids is not None:
        np.save(out_dir / f"{args.split_name}_pred_coords.npy", np.concatenate(saved_preds, axis=0))
        np.save(out_dir / f"{args.split_name}_sample_ids.npy", np.asarray(saved_ids, dtype=np.int64))
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--x_train", required=True)
    tr.add_argument("--x_val", required=True)
    tr.add_argument("--x_test", required=True)
    tr.add_argument("--coords_train", required=True)
    tr.add_argument("--coords_val", required=True)
    tr.add_argument("--coords_test", required=True)
    tr.add_argument("--out_dir", required=True)
    tr.add_argument("--max_atoms", type=int, default=1489)
    tr.add_argument("--batch_size", type=int, default=8)
    tr.add_argument("--epochs", type=int, default=80)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--lr_min", type=float, default=1e-5)
    tr.add_argument("--hidden_dim", type=int, default=2048)
    tr.add_argument("--mlp_layers", type=int, default=4)
    tr.add_argument("--bottleneck_dim", type=int, default=512)
    tr.add_argument("--dropout", type=float, default=0.15)
    tr.add_argument("--weight_decay", type=float, default=1e-4)
    tr.add_argument("--save_every", type=int, default=10)
    tr.add_argument("--grad_clip", type=float, default=1.0)
    tr.add_argument("--huber_beta", type=float, default=5.0)
    tr.add_argument("--init_checkpoint")
    tr.add_argument("--init_partial_matching", action="store_true")
    tr.add_argument("--pca_score_weight", type=float, default=0.0)
    tr.add_argument("--pca_components", type=int, default=10)
    tr.add_argument("--pca_fit_samples", type=int, default=20000)
    tr.add_argument("--rank_weight", type=float, default=0.0)
    tr.add_argument("--rank_margin", type=float, default=0.5)
    tr.add_argument("--rank_hard_k", type=int, default=4)
    tr.add_argument("--rank_duplicate_eps", type=float, default=1e-6)
    tr.add_argument("--rank_local_radius", type=float, default=0.0)
    tr.add_argument("--train_confidence_csv")
    tr.add_argument("--train_confidence_column", default="confidence")
    tr.add_argument("--train_confidence_floor", type=float, default=None)
    tr.add_argument("--train_confidence_power", type=float, default=1.0)
    tr.add_argument("--train_weight_normalize_mean", action="store_true")
    tr.add_argument("--early_stop_patience", type=int, default=None)
    tr.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    tr.add_argument("--min_epochs", type=int, default=1)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--log_every_batches", type=int, default=200)
    tr.add_argument("--max_train_batches", type=int, default=0)
    tr.add_argument("--max_val_batches", type=int, default=0)
    tr.add_argument("--train_indices")
    tr.add_argument("--val_indices")
    tr.add_argument("--test_indices")

    inf = sub.add_parser("infer")
    inf.add_argument("--x_path", required=True)
    inf.add_argument("--coords_path", required=True)
    inf.add_argument("--out_dir", required=True)
    inf.add_argument("--checkpoint", required=True)
    inf.add_argument("--split_name", choices=["train", "val", "test"], required=True)
    inf.add_argument("--max_atoms", type=int, default=1489)
    inf.add_argument("--batch_size", type=int, default=16)
    inf.add_argument("--hidden_dim", type=int, default=2048)
    inf.add_argument("--mlp_layers", type=int, default=4)
    inf.add_argument("--bottleneck_dim", type=int, default=512)
    inf.add_argument("--dropout", type=float, default=0.15)
    inf.add_argument("--align_infer", action="store_true")
    inf.add_argument("--save_pred_coords", action="store_true")
    inf.add_argument("--indices")
    inf.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    seed_everything(args.seed)
    if args.mode == "train":
        run_train(args)
    else:
        run_infer(args)


if __name__ == "__main__":
    main()
