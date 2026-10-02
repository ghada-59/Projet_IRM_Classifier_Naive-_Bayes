import numpy as np
from skimage import feature

IMG_SIZE = (64, 64)
HOG_FEATURES = 1764


def extract_features(images):
    """Extract flattened grayscale pixels and HOG descriptors."""
    images = np.asarray(images)
    if images.ndim != 3 or tuple(images.shape[1:]) != IMG_SIZE:
        raise ValueError(
            f"Expected images with shape (n, {IMG_SIZE[0]}, {IMG_SIZE[1]}), got {images.shape}."
        )

    features_list = []
    for img in images:
        hog_feat = feature.hog(
            img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False
        )
        if hog_feat.shape[0] != HOG_FEATURES:
            raise ValueError(
                f"Unexpected HOG feature length: {hog_feat.shape[0]}; expected {HOG_FEATURES}."
            )
        features_list.append(np.concatenate([img.ravel(), hog_feat]))

    return np.asarray(features_list)
