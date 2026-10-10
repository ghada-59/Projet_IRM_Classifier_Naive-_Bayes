import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from morphometric_pipeline import FEATURE_NAMES, extract_features, discover_images


def test_feature_vector_has_37_finite_values():
    image = np.zeros((100, 120), dtype=np.uint8)
    image[20:80, 25:95] = 180
    values = extract_features(image)
    assert len(FEATURE_NAMES) == 37
    assert values.shape == (37,)
    assert np.isfinite(values).all()


def test_constant_image_is_supported():
    values = extract_features(np.full((64, 64), 128, dtype=np.uint8))
    assert values.shape == (37,)
    assert np.isfinite(values).all()


def test_empty_image_is_rejected():
    with pytest.raises(ValueError):
        extract_features(np.array([], dtype=np.uint8))


def test_dataset_loader_discovers_class_folders(tmp_path):
    import cv2
    for label in ("glioma", "no_tumor"):
        (tmp_path / label).mkdir()
        for idx in range(2):
            image = np.full((32, 32), idx * 60, dtype=np.uint8)
            assert cv2.imwrite(str(tmp_path / label / f"{idx}.png"), image)
    manifest = discover_images(tmp_path)
    assert len(manifest) == 4
    assert set(manifest["label"]) == {"glioma", "no_tumor"}
