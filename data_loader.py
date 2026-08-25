import os
import numpy as np
from skimage import io, color, transform

IMG_SIZE = (64, 64)

def load_mri_split(split_path, class_names=None, max_images_per_class=None):
    """
    Loads MRI images from a specified directory (Training or Testing).
    Resizes the images to (64, 64) and converts them to grayscale.
    """
    images = []
    labels = []
    
    # 1. Check if the root directory exists
    if not os.path.exists(split_path):
        print(f"[!] Error: The directory '{split_path}' does not exist.")
        return np.array([]), np.array([]), []

    # 2. Retrieve subdirectories (tumor classes)
    if class_names is None:
        class_names = sorted([
            d for d in os.listdir(split_path) 
            if os.path.isdir(os.path.join(split_path, d))
        ])
    
    print(f"\nLoading from: {split_path}")
    
    # 3. Iterate through each class
    for label_idx, class_name in enumerate(class_names):
        folder_path = os.path.join(split_path, class_name)
        if not os.path.exists(folder_path):
            continue
            
        # Filter valid image formats
        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ]
        
        selected_files = image_files[:max_images_per_class] if max_images_per_class else image_files
        print(f"  Class [{class_name}] : {len(selected_files)} images loaded")
        
        # 4. Read and preprocess each MRI image
        for img_file in selected_files:
            img_path = os.path.join(folder_path, img_file)
            try:
                img = io.imread(img_path)
                
                # Handle RGBA (4 channels) and RGB (3 channels) -> Grayscale
                if len(img.shape) == 3:
                    if img.shape[2] == 4:  # If RGBA, drop the alpha channel
                        img = img[..., :3]
                    img = color.rgb2gray(img)
                    
                img = transform.resize(img, IMG_SIZE, anti_aliasing=True)
                images.append(img)
                labels.append(label_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                
    return np.array(images), np.array(labels), class_names