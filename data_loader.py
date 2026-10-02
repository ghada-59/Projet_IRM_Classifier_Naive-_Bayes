from pathlib import Path

import numpy as np
from skimage import color, io, transform

IMG_SIZE = (64, 64)
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_mri_split(split_path, class_names=None, max_images_per_class=None):
    """Load MRI images, resize them, and convert them to grayscale."""
    split_path = Path(split_path)
    if not split_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {split_path}")
    if max_images_per_class is not None and max_images_per_class <= 0:
        raise ValueError("max_images_per_class must be positive or None.")

    if class_names is None:
        class_names = sorted(p.name for p in split_path.iterdir() if p.is_dir())
    else:
        class_names = list(class_names)

    if not class_names:
        raise ValueError(f"No class directories found in {split_path}")

    images, labels = [], []
    for label_idx, class_name in enumerate(class_names):
        folder_path = split_path / class_name
        if not folder_path.is_dir():
            raise FileNotFoundError(f"Expected class directory not found: {folder_path}")

        image_files = sorted(
            p for p in folder_path.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        )
        selected_files = (
            image_files[:max_images_per_class]
            if max_images_per_class is not None else image_files
        )

        for img_path in selected_files:
            try:
                img = io.imread(img_path)
                if img.ndim == 3:
                    if img.shape[2] == 4:
                        img = img[..., :3]
                    if img.shape[2] != 3:
                        raise ValueError(f"Unsupported channel count: {img.shape[2]}")
                    img = color.rgb2gray(img)
                elif img.ndim != 2:
                    raise ValueError(f"Unsupported image shape: {img.shape}")

                images.append(transform.resize(img, IMG_SIZE, anti_aliasing=True))
                labels.append(label_idx)
            except (OSError, ValueError) as exc:
                raise ValueError(f"Could not load image {img_path}: {exc}") from exc

    return np.asarray(images), np.asarray(labels), class_names
