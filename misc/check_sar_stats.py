import os
import numpy as np
import cv2
from tqdm import tqdm

sar_dir = 'dataset/train/sar'

def calculate_sar_stats(directory):
    sar_files = [f for f in os.listdir(directory) if f.endswith(('.tif', '.png', '.jpg'))]
    
    if not sar_files:
        print(f"[!] Files not found in: {directory}")
        return

    global_min = float('inf')
    global_max = float('-inf')
    
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    total_pixels = 0

    print(f"[*] Analysis of {len(sar_files)} SAR images from training set...")
    
    for filename in tqdm(sar_files, desc="Skanowanie pikseli"):
        filepath = os.path.join(directory, filename)
        
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            continue
            
        img = np.squeeze(img)
        valid_pixels = img[np.isfinite(img)]
        if valid_pixels.size == 0:
            continue
            
        global_min = min(global_min, np.min(valid_pixels))
        global_max = max(global_max, np.max(valid_pixels))
        
        valid_pixels_f64 = valid_pixels.astype(np.float64)
        
        pixel_sum += np.sum(valid_pixels_f64)
        pixel_sq_sum += np.sum(valid_pixels_f64 ** 2)
        total_pixels += valid_pixels_f64.size

    if total_pixels == 0:
        print("[!] No valid pixels found.")
        return

    global_mean = pixel_sum / total_pixels
    global_variance = (pixel_sq_sum / total_pixels) - (global_mean ** 2)
    global_std = np.sqrt(max(global_variance, 0.0))

    print("\n" + "="*50)
    print("SAR CHANNEL STATS")
    print("="*50)
    print(f"Total pixels analyzed  : {total_pixels:,}")
    print(f"Min                    : {global_min:.4f}")
    print(f"Max                    : {global_max:.4f}")
    print(f"Mean                   : {global_mean:.4f}")
    print(f"Std                    : {global_std:.4f}")
    print("="*50)

if __name__ == '__main__':
    calculate_sar_stats(sar_dir)