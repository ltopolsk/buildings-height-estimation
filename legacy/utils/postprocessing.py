# utils/postprocessing.py
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage

def split_touching_instances(binary_mask):
    """
    Splits touching buildings in a binary mask using Watershed.
    Input: binary_mask (H, W) - numpy array (0 or 1)
    Output: labeled_mask (H, W) - numpy array where each instance has a unique ID (1, 2, 3...)
    """
    # 1. Distance Transform: Calculate distance from every pixel to the nearest background pixel
    distance = ndimage.distance_transform_edt(binary_mask)
    
    # 2. Find Peaks: These are the "centers" of the buildings
    # min_distance=10 prevents finding multiple peaks for a single small roof
    coords = peak_local_max(distance, min_distance=10, labels=binary_mask)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)
    
    # 3. Watershed: "Flood" the image from the markers to define boundaries
    labels = watershed(-distance, markers, mask=binary_mask)
    
    return labels.astype(np.int32)