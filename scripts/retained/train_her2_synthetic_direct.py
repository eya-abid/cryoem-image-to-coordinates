#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset


def extract_id(path: Path) -> str | None:
    m = re.search(r"(\d+)", path.name)
    return m.group(1) if m else None


def find_pairs(spi_dir: str, pdb_dir: str, pdb_glob: str = "*.pdb", spi_glob: str = "*.spi") -> List[Tuple[str, str, str]]:
    spi_paths = sorted(Path(spi_dir).glob(f"**/{spi_glob}"))
    pdb_paths = sorted(Path(pdb_dir).glob(f"**/{pdb_glob}"))
    spi_map = {extract_id(p): p for p in spi_paths if extract_id(p)}
    pdb_map = {extract_id(p): p for p in pdb_paths if extract_id(p)}
    ids = sorted(set(spi_map) & set(pdb_map))
    return [(str(spi_map[i]), str(pdb_map[i]), i) for i in ids]


def load_image(spi_path: str, image_hw: Tuple[int, int]) -> np.ndarray:
    arr = np.array(Image.open(spi_path).convert("F"), dtype=np.float32)
    minv = float(arr.min())
    maxv = float(arr.max())
    if maxv > minv:
        arr = (arr - minv) / (maxv - minv)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    if arr.shape != image_hw:
        im = Image.fromarray(arr)
        im = im.resize((image_hw[1], image_hw[0]), resample=Image.BILINEAR)
        arr = np.array(im, dtype=np.float32)
    return arr


def load_ca_coords_from_pdb(pdb_path: str, max_atoms: int | None = None) -> np.ndarray:
    coords = []
    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            try:
                coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            except ValueError:
                continue
    arr = np.asarray(coords, dtype=np.float32)
    if max_atoms is not None and arr.shape[0] > max_atoms:
        arr = arr[:max_atoms]
    return arr


class SpiImageDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str, str]], max_atoms: int, image_hw: Tuple[int, int]):
        self.pairs = pairs
        self.max_atoms = max_atoms
        self.image_hw = image_hw

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        spi_path, pdb_path, sample_id = self.pairs[idx]
        image = load_image(spi_path, self.image_hw)
        coords = load_ca_coords_from_pdb(pdb_path, self.max_atoms)
        n_atoms = coords.shape[0]
        padded = np.zeros((self.max_atoms, 3), dtype=np.float32)
        mask = np.zeros(self.max_atoms, dtype=np.float32)
        if n_atoms > 0:
            padded[:n_atoms] = coords
            mask[:n_atoms] = 1.0
        return {
            "image": torch.from_numpy(image).unsqueeze(0).float(),
            "coords": torch.from_numpy(padded).permute(1, 0).float(),
            "mask": torch.from_numpy(mask).float(),
            "id": sample_id,
        }


def collate_fn(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "coords": torch.stack([b["coords"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "id": [b["id"] for b in batch],
    }


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
        mlp: List[nn.Module] = []
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


def compute_raw_rmsd(pred_sample: torch.Tensor, target_sample: torch.Tensor, mask_sample: torch.Tensor, eps: float = 1e-8):
    valid = mask_sample > 0.5
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return pred_sample.clone(), pred_sample.new_tensor(0.0), n_valid
    diff = pred_sample[:, valid] - target_sample[:, valid]
    rmsd = torch.sqrt(diff.pow(2).sum(dim=0).mean() + eps)
    return pred_sample.clone(), rmsd, n_valid


def kabsch_align(pred_sample: torch.Tensor, target_sample: torch.Tensor, mask_sample: torch.Tensor, eps: float = 1e-8):
    valid = mask_sample > 0.5
    n_valid = int(valid.sum().item())
    pred_clone = pred_sample.clone()
    if n_valid < 3:
        return compute_raw_rmsd(pred_sample, target_sample, mask_sample, eps=eps)
    P = pred_sample[:, valid].transpose(0, 1)
    T = target_sample[:, valid].transpose(0, 1)
    mu_P = P.mean(dim=0, keepdim=True)
    mu_T = T.mean(dim=0, keepdim=True)
    P_center = P - mu_P
    T_center = T - mu_T
    cov = P_center.transpose(0, 1) @ T_center / float(max(n_valid, 1))
    U, _, Vh = torch.linalg.svd(cov)
    sign = -1.0 if torch.det(U @ Vh) < 0 else 1.0
    diag = torch.ones(3, device=pred_sample.device, dtype=pred_sample.dtype)
    diag[-1] = sign
    rot = U @ torch.diag(diag) @ Vh
    P_aligned = P_center @ rot + mu_T
    diff = P_aligned - T
    rmsd = torch.sqrt(diff.pow(2).sum(dim=1).mean() + eps)
    pred_clone[:, valid] = P_aligned.transpose(0, 1)
    return pred_clone, rmsd, n_valid


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_splits(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def subset_pairs(pairs, ids):
    keep = set(ids)
    return [p for p in pairs if p[2] in keep]


def write_ca_pdb(coords: np.ndarray, path: Path):
    with path.open("w") as h:
        for i, (x, y, z) in enumerate(coords, start=1):
            h.write(f"ATOM  {i:5d}  CA  ALA A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
        h.write("END\n")


def train(args):
    pairs = find_pairs(args.spi_dir, args.pdb_dir, args.pdb_glob, args.spi_glob)
    splits = load_splits(Path(args.splits_path))
    train_pairs = subset_pairs(pairs, splits["train"])
    val_pairs = subset_pairs(pairs, splits["val"])
    test_pairs = subset_pairs(pairs, splits["test"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_train = SpiImageDataset(train_pairs, args.max_atoms, (args.image_h, args.image_w))
    ds_val = SpiImageDataset(val_pairs, args.max_atoms, (args.image_h, args.image_w))
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
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs), eta_min=args.lr_min)

    train_hist, val_hist = [], []
    best_val = float("inf")
    best_epoch = -1
    log_path = out_dir / "training.log"
    with log_path.open("w") as log:
        log.write(f"[{datetime.now().isoformat()}] Training start\n")
        log.write(f"Dataset sizes -> train: {len(train_pairs)}  val: {len(val_pairs)}  test: {len(test_pairs)}\n")
        log.write(json.dumps(vars(args), indent=2, default=str) + "\n")
        log.flush()
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            model.train()
            train_losses = []
            for batch in dl_train:
                image = batch["image"].to(device)
                target = batch["coords"].to(device)
                mask = batch["mask"].to(device)
                pred = model(image)
                loss = huber_loss(pred, target, mask, beta=args.huber_beta)
                opt.zero_grad()
                loss.backward()
                if args.grad_clip is not None:
                    clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                train_losses.append(float(loss.item()))

            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in dl_val:
                    image = batch["image"].to(device)
                    target = batch["coords"].to(device)
                    mask = batch["mask"].to(device)
                    pred = model(image)
                    val_losses.append(float(huber_loss(pred, target, mask, beta=args.huber_beta).item()))

            train_loss = float(np.mean(train_losses))
            val_loss = float(np.mean(val_losses))
            train_hist.append(train_loss)
            val_hist.append(val_loss)
            msg = f"Epoch {epoch}/{args.epochs}  train_huber={train_loss:.4f}  val_huber={val_loss:.4f}  lr={opt.param_groups[0]['lr']:.2e}  epoch_time={time.time()-start:.2f}s"
            print(msg)
            log.write(msg + "\n")
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_rmsd": best_val}, out_dir / "cnn_best.pt")
                log.write(f"New best checkpoint at epoch {epoch} with val_huber={best_val:.4f}\n")
            if epoch % args.save_every == 0 or epoch == args.epochs:
                torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_rmsd": val_loss}, out_dir / f"cnn_epoch{epoch}.pt")
            sched.step()
            log.flush()
        log.write(f"Training complete. Best epoch: {best_epoch}  best val_huber={best_val:.4f}\n")
    np.savetxt(out_dir / "train_huber.txt", np.asarray(train_hist))
    np.savetxt(out_dir / "val_huber.txt", np.asarray(val_hist))
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


def infer(args):
    pairs = find_pairs(args.spi_dir, args.pdb_dir, args.pdb_glob, args.spi_glob)
    infer_pairs = subset_pairs(pairs, load_splits(Path(args.splits_path))[args.infer_split])
    out_dir = Path(args.out_dir)
    pred_dir = Path(args.pred_dir) if args.pred_dir else out_dir / f"predictions_{args.infer_split}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    ds = SpiImageDataset(infer_pairs, args.max_atoms, (args.image_h, args.image_w))
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
    with torch.no_grad():
        for batch in dl:
            image = batch["image"].to(device)
            target = batch["coords"].to(device)
            mask = batch["mask"].to(device)
            pred = model(image)
            for b in range(pred.size(0)):
                aligned, rmsd, n_valid = (kabsch_align if args.align_infer else compute_raw_rmsd)(pred[b], target[b], mask[b])
                sid = batch["id"][b]
                valid = mask[b] > 0.5
                coords_np = aligned[:, valid].transpose(0, 1).cpu().numpy() if n_valid > 0 else np.zeros((0, 3), dtype=np.float32)
                write_ca_pdb(coords_np, pred_dir / f"{sid}_pred.pdb")
                rows.append((sid, float(rmsd.item()), int(n_valid)))
    rows.sort(key=lambda x: x[0])
    rmsd_vals = np.array([r[1] for r in rows], dtype=np.float32)
    with (out_dir / f"inference_{args.infer_split}_rmsd.csv").open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["id", "rmsd", "n_atoms"])
        w.writerows(rows)
    summary = {
        "split": args.infer_split,
        "count": int(len(rows)),
        "mean_rmsd": float(rmsd_vals.mean()),
        "median_rmsd": float(np.median(rmsd_vals)),
        "min_rmsd": float(rmsd_vals.min()),
        "max_rmsd": float(rmsd_vals.max()),
    }
    (out_dir / f"inference_{args.infer_split}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spi_dir", required=True)
    ap.add_argument("--spi_glob", default="*_projected.spi")
    ap.add_argument("--pdb_dir", required=True)
    ap.add_argument("--pdb_glob", default="*.pdb")
    ap.add_argument("--splits_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", choices=["train", "infer"], default="train")
    ap.add_argument("--checkpoint")
    ap.add_argument("--infer_split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--pred_dir")
    ap.add_argument("--max_atoms", type=int, default=1489)
    ap.add_argument("--image_h", type=int, default=128)
    ap.add_argument("--image_w", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_min", type=float, default=1e-5)
    ap.add_argument("--hidden_dim", type=int, default=2048)
    ap.add_argument("--mlp_layers", type=int, default=4)
    ap.add_argument("--bottleneck_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--huber_beta", type=float, default=5.0)
    ap.add_argument("--align_infer", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    seed_everything(args.seed)
    if args.mode == "train":
        train(args)
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for infer mode")
        infer(args)


if __name__ == "__main__":
    main()
