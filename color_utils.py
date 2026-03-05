import cv2
import numpy as np
from sklearn.cluster import KMeans

def get_dominant_color(image_crop):
    """
    Extracts the dominant color from a clothing bounding box crop.
    Uses HSV color space to pre-filter noise (shadows, highlights, background walls),
    then runs KMeans clustering on the vibrant pixels.
    """
    if image_crop is None or image_crop.size == 0:
        return None, None
        
    # 1. Convert BGR to HSV
    hsv_image = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
    
    # Reshape for filtering
    pixels_bgr = image_crop.reshape(-1, 3)
    pixels_hsv = hsv_image.reshape(-1, 3)
    
    # 2. Pre-filter noisy pixels mathematically
    # We want to remove:
    # - Saturation < 15 (Grays, whites, washed out background walls)
    # - Value < 20 (Pure black, heavy shadows, dark unlit areas)
    # - Value > 95 (for 0-100 scale, but OpenCV V is 0-255. So V > 240)
    
    # OpenCV HSV scales: H: 0-179, S: 0-255, V: 0-255
    min_sat = int(255 * (15 / 100)) # ~38
    min_val = int(255 * (20 / 100)) # ~51
    max_val = int(255 * (95 / 100)) # ~242
    
    # Create a boolean mask of "good" pixels
    mask = (pixels_hsv[:, 1] > min_sat) & (pixels_hsv[:, 2] > min_val) & (pixels_hsv[:, 2] < max_val)
    
    filtered_pixels = pixels_bgr[mask]
    
    # 3. Safety Fallback
    # If the garment itself was black, white, or gray, the filter might remove everything.
    if len(filtered_pixels) < 500:
        # Use a vastly simplified filter (just remove extreme dark/light) or fallback to all pixels
        basic_mask = (pixels_hsv[:, 2] > 10) & (pixels_hsv[:, 2] < 245)
        filtered_pixels = pixels_bgr[basic_mask]
        
        # If STILL too few, just use the raw crop
        if len(filtered_pixels) < 100:
            filtered_pixels = pixels_bgr
            
    # 4. K-Means Clustering on the robust pixel set
    # Using 3 clusters to find the most dominant block of color
    kmeans = KMeans(n_clusters=3, n_init='auto', random_state=42)
    kmeans.fit(filtered_pixels)
    
    # 5. Select the cluster with the highest pixel count
    labels = kmeans.labels_
    counts = np.bincount(labels)
    dominant_cluster_idx = np.argmax(counts)
    
    dominant_bgr = kmeans.cluster_centers_[dominant_cluster_idx]
    
    # Ensure it's 8-bit integer
    dominant_bgr = np.uint8(dominant_bgr)
    
    # Convert BGR to Hex
    hex_color = "#{:02x}{:02x}{:02x}".format(dominant_bgr[2], dominant_bgr[1], dominant_bgr[0]) # RGB hex
    
    # Convert BGR to HSV to get the Hue for our band filtering
    dominant_bgr_array = np.uint8([[dominant_bgr]])
    dominant_hsv = cv2.cvtColor(dominant_bgr_array, cv2.COLOR_BGR2HSV)[0][0]
    
    # OpenCV Hue is 0-179. Standard Hue is 0-360.
    true_hue = int(dominant_hsv[0] * 2)
    
    return hex_color, true_hue
