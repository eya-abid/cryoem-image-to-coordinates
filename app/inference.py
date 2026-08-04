from __future__ import annotations

import base64
import io
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from model import StrongCNNToCoordinates
from staged_model import ConvAutoencoder, StagedCoordinateDecoder, StagedImageToCoordinates


APP_DIR = Path(__file__).resolve().parent
ASSET_DIR = APP_DIR / "assets"
CHECKPOINT_DIR = Path(os.environ.get("CRYOEM_APP_CHECKPOINT_DIR", APP_DIR / "checkpoints"))
CHAIN_COLORS = {
    "A": "#1769aa",
    "B": "#20a39e",
    "C": "#087f8c",
    "D": "#ef8354",
    "E": "#c83e2b",
}


def checkpoint_path(system: str, component: str) -> Path:
    environment_name = f"CRYOEM_{system.upper()}_{component.upper()}_CHECKPOINT"
    default_name = {
        "direct": "direct.pt",
        "staged_encoder": "staged_encoder.pt",
        "staged_decoder": "staged_decoder.pt",
    }[component]
    return Path(os.environ.get(environment_name, CHECKPOINT_DIR / system / default_name)).expanduser()


@dataclass(frozen=True)
class SystemConfig:
    key: str
    short_name: str
    full_name: str
    checkpoint: Path
    staged_encoder_checkpoint: Path
    staged_decoder_checkpoint: Path
    staged_latent_dim: int
    staged_encoder_channels: tuple[int, ...]
    staged_output_activation: str
    staged_decoder_dropout: float
    staged_decoder_progressive: bool
    atom_count: int
    representation: str
    normalization: str
    target_status: str
    evidence: tuple[tuple[str, str], ...]
    conclusion: str
    caveat: str


SYSTEMS = {
    "ak": SystemConfig(
        key="ak",
        short_name="AK",
        full_name="AK controlled synthetic benchmark",
        checkpoint=checkpoint_path("ak", "direct"),
        staged_encoder_checkpoint=checkpoint_path("ak", "staged_encoder"),
        staged_decoder_checkpoint=checkpoint_path("ak", "staged_decoder"),
        staged_latent_dim=32,
        staged_encoder_channels=(16, 32, 64),
        staged_output_activation="sigmoid",
        staged_decoder_dropout=0.1,
        staged_decoder_progressive=False,
        atom_count=1656,
        representation="1,656-position known full-atom rotshift coordinate array",
        normalization="per-image min-max to [0, 1]",
        target_status="paired known synthetic full-atom target",
        evidence=(("2.356 Å", "mean posed-frame test RMSD"), ("1.922 Å", "median posed-frame test RMSD"), ("3,000", "held-out particles")),
        conclusion="The controlled AK benchmark establishes direct image-to-coordinate feasibility in a known-target setting.",
        caveat="AK is a within-generator synthetic benchmark; it does not establish recovery from experimental cryo-EM particles.",
    ),
    "her2_synthetic": SystemConfig(
        key="her2_synthetic",
        short_name="Synthetic HER2",
        full_name="Synthetic HER2 known-target benchmark",
        checkpoint=checkpoint_path("her2_synthetic", "direct"),
        staged_encoder_checkpoint=checkpoint_path("her2_synthetic", "staged_encoder"),
        staged_decoder_checkpoint=checkpoint_path("her2_synthetic", "staged_decoder"),
        staged_latent_dim=64,
        staged_encoder_channels=(16, 32, 64),
        staged_output_activation="sigmoid",
        staged_decoder_dropout=0.1,
        staged_decoder_progressive=False,
        atom_count=1489,
        representation="1,489-position known synthetic C-alpha rotshift coordinate array",
        normalization="per-image min-max to [0, 1]",
        target_status="paired known synthetic C-alpha target",
        evidence=(("2.975 Å", "mean posed-frame test RMSD"), ("2.739 Å", "median posed-frame test RMSD"), ("1.518 Å", "mean rigidly aligned test RMSD")),
        conclusion="Synthetic HER2 tests whether the direct model scales to a larger multi-chain C-alpha representation.",
        caveat="These targets are known synthetic structures from the retained generator, not experimental per-particle structures.",
    ),
    "her2_experimental": SystemConfig(
        key="her2_experimental",
        short_name="Experimental HER2",
        full_name="Experimental HER2 surrogate-target regime",
        checkpoint=checkpoint_path("her2_experimental", "direct"),
        staged_encoder_checkpoint=checkpoint_path("her2_experimental", "staged_encoder"),
        staged_decoder_checkpoint=checkpoint_path("her2_experimental", "staged_decoder"),
        staged_latent_dim=64,
        staged_encoder_channels=(16, 32, 64, 128),
        staged_output_activation="identity",
        staged_decoder_dropout=0.3,
        staged_decoder_progressive=True,
        atom_count=1489,
        representation="1,489-position MDSPACE-derived C-alpha surrogate coordinate array",
        normalization="p1-p99 clipping followed by per-image z-score",
        target_status="paired MDSPACE-derived C-alpha surrogate target",
        evidence=(("4.309 Å", "mean surrogate-target test RMSD"), ("66.94%", "improved over the training-target mean"), ("2.33%", "top-100 paired-target recovery")),
        conclusion="Image-dependent coordinate signal is present, but particle-specific conformational assignment remains unresolved.",
        caveat="Outputs are predictions within a globally inferred MDSPACE surrogate system, not directly observed per-particle atomic structures.",
    ),
}


def center_crop_or_pad(image: np.ndarray, size: int = 128) -> tuple[np.ndarray, str | None]:
    height, width = image.shape
    warning = None
    if height > size:
        top = (height - size) // 2
        image = image[top : top + size, :]
        height = size
        warning = "The uploaded image was center-cropped to 128 x 128 pixels."
    if width > size:
        left = (width - size) // 2
        image = image[:, left : left + size]
        width = size
        warning = "The uploaded image was center-cropped to 128 x 128 pixels."
    if height < size or width < size:
        output = np.zeros((size, size), dtype=np.float32)
        top = (size - height) // 2
        left = (size - width) // 2
        output[top : top + height, left : left + width] = image
        image = output
        warning = "The uploaded image was zero-padded to 128 x 128 pixels."
    return image, warning


def resize_image(image: np.ndarray, size: int = 128) -> tuple[np.ndarray, str | None]:
    if image.shape == (size, size):
        return image, None
    resized = Image.fromarray(np.asarray(image, dtype=np.float32), mode="F").resize(
        (size, size), resample=Image.Resampling.BILINEAR
    )
    return np.asarray(resized, dtype=np.float32), "The uploaded image was bilinearly resized to 128 x 128 pixels."


def normalize_minmax(image: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    image = np.asarray(image, dtype=np.float32)
    minimum = float(image.min())
    maximum = float(image.max())
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        raise ValueError("The image has no usable intensity variation.")
    normalized = (image - minimum) / (maximum - minimum)
    return normalized.astype(np.float32), {
        "input_min": minimum,
        "input_max": maximum,
        "normalized_min": float(normalized.min()),
        "normalized_max": float(normalized.max()),
    }


def normalize_zscore(image: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    image = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(image, [1.0, 99.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("The image has no usable intensity variation.")
    clipped = np.clip(image, low, high)
    mean = float(clipped.mean())
    standard_deviation = float(clipped.std())
    if standard_deviation < 1e-6:
        raise ValueError("The clipped image is effectively constant.")
    normalized = (clipped - mean) / standard_deviation
    return normalized.astype(np.float32), {
        "clip_low": float(low),
        "clip_high": float(high),
        "clipped_mean": mean,
        "clipped_std": standard_deviation,
        "normalized_min": float(normalized.min()),
        "normalized_max": float(normalized.max()),
    }


def decode_image(filename: str, encoded_content: str, config: SystemConfig) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        content = base64.b64decode(encoded_content, validate=True)
    except ValueError as error:
        raise ValueError("The uploaded image payload is not valid base64.") from error
    if len(content) > 25 * 1024 * 1024:
        raise ValueError("The uploaded file exceeds the 25 MB limit.")
    suffix = Path(filename).suffix.lower()
    if suffix == ".npy":
        image = np.load(io.BytesIO(content), allow_pickle=False)
    else:
        try:
            with Image.open(io.BytesIO(content)) as opened:
                image = np.asarray(opened.convert("F"), dtype=np.float32)
        except Exception as error:
            raise ValueError("Use a readable SPI, PNG, TIFF, JPEG, or two-dimensional NPY particle image.") from error
    image = np.squeeze(np.asarray(image))
    if image.ndim != 2:
        raise ValueError(f"Expected one 2D particle image, received shape {tuple(image.shape)}.")
    original_shape = [int(value) for value in image.shape]
    if config.normalization.startswith("per-image min-max"):
        image, normalization = normalize_minmax(image)
        image, shape_warning = resize_image(image)
    else:
        image, shape_warning = center_crop_or_pad(image)
        image, normalization = normalize_zscore(image)
    return image, {
        "original_shape": original_shape,
        "model_shape": [128, 128],
        "shape_warning": shape_warning,
        "normalization": config.normalization,
        **normalization,
    }


def rmsd(first: np.ndarray, second: np.ndarray) -> float:
    difference = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(difference * difference, axis=1))))


def radius_of_gyration(coordinates: np.ndarray) -> float:
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


def image_to_png_data_url(image: np.ndarray) -> str:
    low, high = np.percentile(image, [1.0, 99.0])
    scaled = np.clip((image - low) / max(high - low, 1e-6), 0.0, 1.0)
    output = Image.fromarray(np.uint8(scaled * 255), mode="L")
    buffer = io.BytesIO()
    output.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class SystemService:
    def __init__(self, config: SystemConfig, device: torch.device):
        self.config = config
        self.device = device
        asset_directory = ASSET_DIR / config.key
        self.training_mean = np.load(asset_directory / "training_target_mean.npy").astype(np.float32)
        staged_mean_path = asset_directory / "staged_training_target_mean.npy"
        self.staged_training_mean = (
            np.load(staged_mean_path).astype(np.float32)
            if staged_mean_path.exists()
            else self.training_mean
        )
        examples_path = asset_directory / "held_out_examples.npz"
        self.examples = np.load(examples_path) if examples_path.exists() else None
        fresh_path = asset_directory / "fresh_scipion_examples.npz"
        self.fresh_examples = np.load(fresh_path) if fresh_path.exists() else None
        self.topology = json.loads((asset_directory / "topology.json").read_text())
        self.segments = self.topology["segments"]
        self.model: StrongCNNToCoordinates | None = None
        self.staged_model: StagedImageToCoordinates | None = None
        self.checkpoint_epoch: int | None = None
        self.staged_checkpoint_epochs: tuple[int, int] | None = None
        self._lock = threading.Lock()

    def load_model(self) -> StrongCNNToCoordinates:
        with self._lock:
            if self.model is None:
                if not self.config.checkpoint.exists():
                    raise FileNotFoundError(
                        f"Direct checkpoint is unavailable for {self.config.short_name}. "
                        "See app/CHECKPOINTS.md for installation instructions."
                    )
                model = StrongCNNToCoordinates(atom_count=self.config.atom_count).to(self.device)
                checkpoint = torch.load(self.config.checkpoint, map_location=self.device, weights_only=False)
                model.load_state_dict(checkpoint["model_state"])
                model.eval()
                self.model = model
                self.checkpoint_epoch = int(checkpoint.get("epoch", -1))
            return self.model

    def load_staged_model(self) -> StagedImageToCoordinates:
        with self._lock:
            if self.staged_model is None:
                missing = [
                    path
                    for path in (
                        self.config.staged_encoder_checkpoint,
                        self.config.staged_decoder_checkpoint,
                    )
                    if not path.exists()
                ]
                if missing:
                    raise FileNotFoundError(
                        f"Staged checkpoints are unavailable for {self.config.short_name}. "
                        "See app/CHECKPOINTS.md for installation instructions."
                    )
                autoencoder = ConvAutoencoder(
                    latent_dim=self.config.staged_latent_dim,
                    channels=self.config.staged_encoder_channels,
                    output_activation=self.config.staged_output_activation,
                )
                coordinate_decoder = StagedCoordinateDecoder(
                    latent_dim=self.config.staged_latent_dim,
                    atom_count=self.config.atom_count,
                    dropout=self.config.staged_decoder_dropout,
                    progressive=self.config.staged_decoder_progressive,
                )
                encoder_checkpoint = torch.load(
                    self.config.staged_encoder_checkpoint,
                    map_location=self.device,
                    weights_only=False,
                )
                decoder_checkpoint = torch.load(
                    self.config.staged_decoder_checkpoint,
                    map_location=self.device,
                    weights_only=False,
                )
                autoencoder.load_state_dict(encoder_checkpoint["model_state"])
                coordinate_decoder.load_state_dict(decoder_checkpoint["model_state"])
                model = StagedImageToCoordinates(autoencoder, coordinate_decoder).to(self.device)
                model.eval()
                self.staged_model = model
                self.staged_checkpoint_epochs = (
                    int(encoder_checkpoint.get("epoch", -1)),
                    int(decoder_checkpoint.get("epoch", -1)),
                )
            return self.staged_model

    def metadata(self) -> dict[str, Any]:
        return {
            "key": self.config.key,
            "short_name": self.config.short_name,
            "full_name": self.config.full_name,
            "atom_count": self.config.atom_count,
            "representation": self.config.representation,
            "normalization": self.config.normalization,
            "target_status": self.config.target_status,
            "evidence": [{"value": value, "label": label} for value, label in self.config.evidence],
            "conclusion": self.config.conclusion,
            "caveat": self.config.caveat,
            "availability": {
                "direct": self.config.checkpoint.exists(),
                "staged": self.config.staged_encoder_checkpoint.exists()
                and self.config.staged_decoder_checkpoint.exists(),
                "examples": self.examples is not None or self.fresh_examples is not None,
            },
            "methods": [
                {"key": "direct", "label": "Direct", "description": "residual CNN · 512D coordinate head"},
                {
                    "key": "staged",
                    "label": "Staged",
                    "description": f"CAE {self.config.staged_latent_dim}D latent · 1D U-Net decoder",
                },
            ],
        }

    def example_catalog(self) -> list[dict[str, Any]]:
        entries = []
        if self.examples is not None:
            identifiers = self.examples["identifiers"]
            for slot, identifier in enumerate(identifiers):
                label = str(identifier)
                prefix = "Unseen test" if self.config.key == "her2_synthetic" else "Held-out"
                entries.append(
                    {
                        "slot": slot,
                        "kind": "held_out",
                        "identifier": label,
                        "label": f"{prefix} {label.zfill(5)}",
                        "thumbnail": image_to_png_data_url(self.examples["images"][slot]),
                    }
                )
        if self.fresh_examples is not None:
            for slot, identifier in enumerate(self.fresh_examples["identifiers"]):
                entries.append(
                    {
                        "slot": slot,
                        "kind": "fresh_scipion",
                        "identifier": str(identifier),
                        "label": f"Post-training Scipion {int(identifier):02d}",
                        "thumbnail": image_to_png_data_url(self.fresh_examples["images"][slot]),
                    }
                )
        return entries

    def _predict_array(self, image: np.ndarray, method: str = "direct") -> np.ndarray:
        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32))[None, None].to(self.device)
        if method == "direct":
            model = self.load_model()
        elif method == "staged":
            model = self.load_staged_model()
        else:
            raise ValueError(f"Unknown prediction method '{method}'.")
        with self._lock, torch.inference_mode():
            prediction = model(tensor)[0].detach().cpu().numpy()
        return prediction.astype(np.float32)

    def predict_upload(self, filename: str, encoded_content: str, method: str = "direct") -> dict[str, Any]:
        image, preprocessing = decode_image(filename, encoded_content, self.config)
        return self._result(self._predict_array(image, method), preprocessing, filename, None, None, method)

    def predict_example(
        self, slot: int, method: str = "direct", kind: str = "held_out"
    ) -> dict[str, Any]:
        examples = self.examples if kind == "held_out" else self.fresh_examples
        if examples is None or slot < 0 or slot >= len(examples["identifiers"]):
            raise ValueError("Unknown held-out example.")
        image = np.asarray(examples["images"][slot], dtype=np.float32)
        target_key = "staged_targets" if method == "staged" and "staged_targets" in examples else "targets"
        target = np.asarray(examples[target_key][slot], dtype=np.float32)
        identifier = str(examples["identifiers"][slot])
        is_fresh = kind == "fresh_scipion"
        preprocessing = {
            "original_shape": [128, 128],
            "model_shape": [128, 128],
            "shape_warning": None,
            "source": (
                "fresh deterministic Scipion3 control; matched pose; not part of the retained split"
                if is_fresh
                else "stored normalized held-out test image"
            ),
            "normalization": self.config.normalization,
        }
        return self._result(
            self._predict_array(image, method),
            preprocessing,
            f"fresh_scipion_{identifier}" if is_fresh else f"held_out_{identifier}",
            target,
            f"fresh:{identifier}" if is_fresh else identifier,
            method,
        )

    def _result(
        self,
        prediction: np.ndarray,
        preprocessing: dict[str, Any],
        source_name: str,
        target: np.ndarray | None,
        test_identifier: str | None,
        method: str,
    ) -> dict[str, Any]:
        training_mean = self.staged_training_mean if method == "staged" else self.training_mean
        displacement = np.linalg.norm(prediction - training_mean, axis=1)
        metrics: dict[str, Any] = {
            "displacement_from_training_mean_rmsd": rmsd(prediction, training_mean),
            "prediction_radius_of_gyration": radius_of_gyration(prediction),
            "training_mean_radius_of_gyration": radius_of_gyration(training_mean),
            "median_position_displacement": float(np.median(displacement)),
            "p95_position_displacement": float(np.quantile(displacement, 0.95)),
        }
        if target is not None:
            metrics["paired_target_raw_rmsd"] = rmsd(prediction, target)
            metrics["training_mean_raw_rmsd"] = rmsd(training_mean, target)
            metrics["improves_on_training_mean"] = metrics["paired_target_raw_rmsd"] < metrics["training_mean_raw_rmsd"]
        result = {
            "system_key": self.config.key,
            "source_name": source_name,
            "test_identifier": test_identifier,
            "prediction": np.round(prediction, 5).tolist(),
            "training_mean": np.round(training_mean, 5).tolist(),
            "paired_target": None if target is None else np.round(target, 5).tolist(),
            "displacement": np.round(displacement, 5).tolist(),
            "segments": self.segments,
            "topology": self.topology,
            "chain_colors": CHAIN_COLORS,
            "preprocessing": preprocessing,
            "metrics": metrics,
            "model": {
                **self.metadata(),
                "method_key": method,
                "branch": (
                    "direct residual CNN, 512D"
                    if method == "direct"
                    else f"staged CAE {self.config.staged_latent_dim}D latent to 1D U-Net"
                ),
                "checkpoint_epoch": (
                    self.checkpoint_epoch if method == "direct" else self.staged_checkpoint_epochs
                ),
                "device": str(self.device),
            },
        }
        result["pdb"] = self.pdb_text(prediction, source_name, method)
        return result

    def pdb_text(self, coordinates: np.ndarray, source_name: str, method: str) -> str:
        lines = [
            f"REMARK 900 IMAGE-CONDITIONED {self.config.short_name.upper()} PREDICTION",
            f"REMARK 900 REPRESENTATION {self.config.representation.upper()}",
            f"REMARK 900 TARGET STATUS {self.config.target_status.upper()}",
            f"REMARK 900 METHOD {method.upper()}",
            "REMARK 900 DISPLAYED MOTION IS LINEAR INTERPOLATION, NOT MOLECULAR DYNAMICS",
            f"REMARK 900 INPUT {source_name[:60]}",
        ]
        for atom, coordinate in zip(self.topology["atoms"], coordinates):
            atom_name = str(atom.get("atom_name", "CA"))[:4]
            element = str(atom.get("element", atom_name[0] if atom_name else "C"))[:2]
            lines.append(
                f"ATOM  {int(atom['serial']):5d} {atom_name:>4s} {str(atom['residue_name']):>3s} "
                f"{str(atom['chain']):1s}{int(atom['residue_number']):4d}    "
                f"{float(coordinate[0]):8.3f}{float(coordinate[1]):8.3f}{float(coordinate[2]):8.3f}"
                f"  1.00  0.00          {element:>2s}"
            )
        lines.extend(["TER", "END"])
        return "\n".join(lines) + "\n"


class PredictionService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.systems = {key: SystemService(config, self.device) for key, config in SYSTEMS.items()}

    def system(self, key: str) -> SystemService:
        try:
            return self.systems[key]
        except KeyError as error:
            raise ValueError(f"Unknown system '{key}'.") from error

    def catalog(self) -> list[dict[str, Any]]:
        return [service.metadata() for service in self.systems.values()]
