import numpy as np
from skimage import draw, morphology

from fracture_detection import (
    PipelineConfig,
    classify_pixels,
    evaluate_predictions,
    extract_features,
    postprocess_mask,
    preprocess_image,
)


def _synthetic_fracture_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full((180, 120), 0.7, dtype=float)

    rr, cc = draw.line(20, 5, 160, 110)
    image[rr, cc] = 0.15
    rr2, cc2 = draw.line(40, 0, 60, 119)
    image[rr2, cc2] = 0.2

    image = morphology.dilation(image, morphology.disk(1))
    noise = rng.normal(scale=0.03, size=image.shape)
    image = np.clip(image + noise, 0, 1)
    return image


def test_pipeline_recovers_synthetic_fracture():
    image = _synthetic_fracture_image()
    config = PipelineConfig(remove_small_objects=40, closing_disk_radius=1, opening_disk_radius=0)

    preprocessed = preprocess_image(image, config)
    features, names = extract_features(preprocessed, config)
    labels = classify_pixels(features, names, config)
    mask = postprocess_mask(labels, preprocessed.shape, config)

    ground_truth = np.zeros_like(mask, dtype=bool)
    rr, cc = draw.line(20, 5, 160, 110)
    ground_truth[rr, cc] = True
    rr2, cc2 = draw.line(40, 0, 60, 119)
    ground_truth[rr2, cc2] = True
    ground_truth = morphology.dilation(ground_truth, morphology.disk(2))

    metrics = evaluate_predictions(mask, ground_truth)

    assert metrics["recall"] > 0.6
    assert mask.sum() > 200
