import numpy as np
from skimage import feature

HOG_FEATURES = 1764

def extract_features(images):
    """
    Extract raw pixel values and HOG descriptors from MRI images.
    """
    features_list = []
    print("\nExtracting features (pixels + HOG descriptor)...")

    for img in images:
        flattened = img.flatten()
        hog_feat = feature.hog(
            img,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False
        )

        if hog_feat.shape[0] != HOG_FEATURES:
            raise ValueError(
                f"Unexpected HOG feature length: {hog_feat.shape[0]}; "
                f"expected {HOG_FEATURES}. Check the input image size."
            )

        combined_features = np.concatenate([flattened, hog_feat])
        features_list.append(combined_features)

    return np.asarray(features_list)
