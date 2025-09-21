"""Fracture detection pipeline for borehole televiewer imagery."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from skimage import color, exposure, filters, io, morphology, restoration, util
from skimage.filters import rank
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@dataclass
class PipelineConfig:
    """Configuration options controlling the pipeline."""

    gaussian_sigma: float = 1.2
    bilateral_sigma_color: float = 0.1
    bilateral_sigma_spatial: float = 3.0
    bilateral_bins: int = 200
    clahe_clip_limit: float = 0.03
    clahe_kernel_size: Optional[int] = None
    entropy_disk_radius: int = 5
    remove_small_objects: int = 200
    closing_disk_radius: int = 2
    opening_disk_radius: int = 1
    kmeans_n_init: int = 10
    random_state: int = 13
    gabor_frequencies: Sequence[float] = field(
        default_factory=lambda: (0.1, 0.2, 0.3)
    )
    gabor_thetas: Sequence[float] = field(
        default_factory=lambda: tuple(np.deg2rad(v) for v in (0, 30, 60, 90, 120, 150))
    )


def preprocess_image(image: np.ndarray, config: Optional[PipelineConfig] = None) -> np.ndarray:
    """Denoise and enhance contrast of the input image."""

    cfg = config or PipelineConfig()

    if image.ndim == 3:
        image = color.rgb2gray(image)
    image = util.img_as_float(image)

    blurred = filters.gaussian(image, sigma=cfg.gaussian_sigma, preserve_range=True)

    denoised = restoration.denoise_bilateral(
        blurred,
        sigma_color=cfg.bilateral_sigma_color,
        sigma_spatial=cfg.bilateral_sigma_spatial,
        bins=cfg.bilateral_bins,
        channel_axis=None,
    )

    equalized = exposure.equalize_adapthist(
        denoised, clip_limit=cfg.clahe_clip_limit, kernel_size=cfg.clahe_kernel_size
    )

    return equalized


def _compute_entropy(image: np.ndarray, radius: int) -> np.ndarray:
    scaled = util.img_as_ubyte(image)
    entropy = rank.entropy(scaled, morphology.disk(radius))
    entropy = entropy.astype(np.float32)
    entropy /= entropy.max() if entropy.max() > 0 else 1.0
    return entropy


def extract_features(
    image: np.ndarray, config: Optional[PipelineConfig] = None
) -> Tuple[np.ndarray, List[str]]:
    """Extract pixel-wise features suitable for fracture detection."""

    cfg = config or PipelineConfig()

    if image.ndim == 3:
        image = color.rgb2gray(image)
    image = util.img_as_float(image)

    gradients = np.stack(
        [filters.sobel_h(image), filters.sobel_v(image)], axis=-1
    )
    gradient_magnitude = np.hypot(gradients[..., 0], gradients[..., 1])

    laplacian = filters.laplace(image, ksize=3)

    entropy = _compute_entropy(image, cfg.entropy_disk_radius)

    local_std = filters.gaussian((image - filters.gaussian(image, sigma=3)) ** 2, sigma=1)
    local_std = np.sqrt(np.clip(local_std, 0, None))

    gabor_features: List[np.ndarray] = []
    for frequency in cfg.gabor_frequencies:
        for theta in cfg.gabor_thetas:
            real, imag = filters.gabor(image, frequency=frequency, theta=theta)
            magnitude = np.hypot(real, imag)
            gabor_features.append(magnitude)

    feature_stack = [
        image,
        gradients[..., 0],
        gradients[..., 1],
        gradient_magnitude,
        laplacian,
        entropy,
        local_std,
    ] + gabor_features

    features = np.stack(feature_stack, axis=-1)
    feature_names = [
        "intensity",
        "sobel_h",
        "sobel_v",
        "gradient_mag",
        "laplacian",
        "entropy",
        "local_std",
    ]
    feature_names += [
        f"gabor_f{frequency:.2f}_t{int(math.degrees(theta))}"
        for frequency in cfg.gabor_frequencies
        for theta in cfg.gabor_thetas
    ]

    flat_features = features.reshape(-1, features.shape[-1])
    return flat_features, feature_names


def classify_pixels(
    features: np.ndarray,
    feature_names: Sequence[str],
    config: Optional[PipelineConfig] = None,
) -> np.ndarray:
    """Cluster pixels and identify the fracture class."""

    cfg = config or PipelineConfig()

    kmeans = KMeans(
        n_clusters=2,
        n_init=cfg.kmeans_n_init,
        random_state=cfg.random_state,
    )
    labels = kmeans.fit_predict(features)

    gradient_idx = feature_names.index("gradient_mag")
    intensity_idx = feature_names.index("intensity")

    cluster_scores = []
    for cluster_label in range(2):
        cluster_mask = labels == cluster_label
        if not np.any(cluster_mask):
            cluster_scores.append((-np.inf, np.inf))
            continue
        mean_gradient = features[cluster_mask, gradient_idx].mean()
        mean_intensity = features[cluster_mask, intensity_idx].mean()
        cluster_scores.append((mean_gradient, mean_intensity))

    fracture_label = max(range(2), key=lambda idx: (cluster_scores[idx][0], -cluster_scores[idx][1]))
    fracture_mask = labels == fracture_label
    return fracture_mask


def postprocess_mask(
    mask: np.ndarray, image_shape: Tuple[int, int], config: Optional[PipelineConfig] = None
) -> np.ndarray:
    """Apply morphological cleanup steps to the predicted mask."""

    cfg = config or PipelineConfig()
    reshaped = mask.reshape(image_shape)
    cleaned = morphology.remove_small_objects(reshaped, min_size=cfg.remove_small_objects)
    if cfg.closing_disk_radius > 0:
        cleaned = morphology.closing(cleaned, morphology.disk(cfg.closing_disk_radius))
    if cfg.opening_disk_radius > 0:
        cleaned = morphology.opening(cleaned, morphology.disk(cfg.opening_disk_radius))
    return cleaned.astype(bool)


def evaluate_predictions(
    prediction: np.ndarray, ground_truth: np.ndarray
) -> Dict[str, float]:
    """Compute common segmentation metrics given a ground-truth mask."""

    prediction = prediction.astype(bool).ravel()
    ground_truth = ground_truth.astype(bool).ravel()

    if prediction.shape != ground_truth.shape:
        raise ValueError("Prediction and ground truth must have the same number of pixels")

    metrics = {
        "accuracy": accuracy_score(ground_truth, prediction),
        "precision": precision_score(ground_truth, prediction, zero_division=0),
        "recall": recall_score(ground_truth, prediction, zero_division=0),
        "f1": f1_score(ground_truth, prediction, zero_division=0),
    }

    intersection = np.logical_and(prediction, ground_truth).sum()
    union = np.logical_or(prediction, ground_truth).sum()
    metrics["iou"] = float(intersection / union) if union else 0.0
    return metrics


def _load_mask(mask_path: Path) -> np.ndarray:
    mask = util.img_as_bool(io.imread(mask_path))
    return mask


def process_image(
    image_path: Path,
    mask_path: Optional[Path] = None,
    config: Optional[PipelineConfig] = None,
    save_intermediate: bool = False,
    intermediate_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Run the full pipeline on a single image."""

    cfg = config or PipelineConfig()
    image = io.imread(image_path)
    preprocessed = preprocess_image(image, cfg)
    features, feature_names = extract_features(preprocessed, cfg)
    raw_mask = classify_pixels(features, feature_names, cfg)
    cleaned_mask = postprocess_mask(raw_mask, preprocessed.shape, cfg)

    result: Dict[str, object] = {
        "image_path": str(image_path),
        "mask": cleaned_mask,
    }

    if mask_path is not None and mask_path.exists():
        ground_truth = _load_mask(mask_path)
        if ground_truth.shape != cleaned_mask.shape:
            raise ValueError(
                f"Shape mismatch between prediction {cleaned_mask.shape} and ground truth {ground_truth.shape}"
            )
        metrics = evaluate_predictions(cleaned_mask, ground_truth)
        result["metrics"] = metrics

    if save_intermediate and intermediate_dir is not None:
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        np.save(intermediate_dir / f"{image_path.stem}_preprocessed.npy", preprocessed)
        np.save(intermediate_dir / f"{image_path.stem}_raw_mask.npy", raw_mask.reshape(preprocessed.shape))
        np.save(intermediate_dir / f"{image_path.stem}_clean_mask.npy", cleaned_mask)

    return result


def process_images(
    image_paths: Iterable[Path],
    mask_paths: Optional[Dict[str, Path]] = None,
    config: Optional[PipelineConfig] = None,
    output_dir: Optional[Path] = None,
    save_intermediate: bool = False,
) -> pd.DataFrame:
    """Process a collection of images and optionally persist results."""

    cfg = config or PipelineConfig()
    output_dir = output_dir or Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []

    for image_path in image_paths:
        mask_path = None
        if mask_paths is not None:
            mask_path = mask_paths.get(image_path.stem)
        result = process_image(
            image_path,
            mask_path=mask_path,
            config=cfg,
            save_intermediate=save_intermediate,
            intermediate_dir=output_dir / "intermediate",
        )
        cleaned_mask = result.pop("mask")
        mask_output_path = output_dir / f"{image_path.stem}_mask.png"
        io.imsave(mask_output_path, cleaned_mask.astype(np.uint8) * 255, check_contrast=False)

        record: Dict[str, object] = {
            "image": str(image_path),
            "predicted_mask": str(mask_output_path),
        }
        if "metrics" in result:
            record.update(result["metrics"])  # type: ignore[arg-type]
        records.append(record)

    dataframe = pd.DataFrame.from_records(records)
    if not dataframe.empty:
        dataframe.to_csv(output_dir / "metrics.csv", index=False)
    return dataframe


def _resolve_paths(path: Path) -> List[Path]:
    if path.is_dir():
        return sorted(
            [
                p
                for p in path.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
            ]
        )
    return [path]


def _load_mask_mapping(mask_argument: Optional[str]) -> Optional[Dict[str, Path]]:
    if mask_argument is None:
        return None
    mask_path = Path(mask_argument)
    mask_paths = _resolve_paths(mask_path)
    return {p.stem: p for p in mask_paths}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        required=True,
        help="Path to an image file or a directory containing borehole images.",
    )
    parser.add_argument(
        "--masks",
        help="Optional path to ground-truth binary masks for evaluation.",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Directory where predicted masks and metrics will be stored.",
    )
    parser.add_argument(
        "--config",
        help="Optional JSON file describing pipeline configuration overrides.",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Persist intermediate arrays (preprocessed image, raw mask) for debugging.",
    )
    return parser


def _load_config(path: Optional[str]) -> PipelineConfig:
    if not path:
        return PipelineConfig()
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return PipelineConfig(**data)


def cli(argv: Optional[Sequence[str]] = None) -> pd.DataFrame:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    images = [Path(p) for p in _resolve_paths(Path(args.images))]
    mask_mapping = _load_mask_mapping(args.masks)
    config = _load_config(args.config)

    return process_images(
        images,
        mask_paths=mask_mapping,
        config=config,
        output_dir=Path(args.output),
        save_intermediate=args.save_intermediate,
    )


if __name__ == "__main__":  # pragma: no cover
    cli()
