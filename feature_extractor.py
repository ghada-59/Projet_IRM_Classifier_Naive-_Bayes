import numpy as np
from skimage import feature

def extract_features(images):
    """
    Extracts combined visual biomarkers (raw flattened pixels + HOG descriptor) from a list of images.
    """
    features_list = []
    print("\nExtracting biomarkers (Pixels + HOG Descriptor)...")
    
    for i, img in enumerate(images):
        flattened = img.flatten()
        try:
            hog_feat = feature.hog(
                img, 
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                visualize=False
            )
        except Exception:
            hog_feat = np.zeros(1764)
            
        combined_features = np.concatenate([flattened, hog_feat])
        features_list.append(combined_features)
        
    return np.array(features_list)