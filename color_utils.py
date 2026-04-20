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
    min_sat = int(255 * (5 / 100)) # Lowered to 5% to keep gray/white clothing
    min_val = int(255 * (10 / 100)) # Lowered to 10% to keep dark gray/black clothing
    max_val = int(255 * (98 / 100)) # Raised to 98% to keep bright white clothing
    
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
    
    color_name = get_color_name(dominant_hsv)
    
    return hex_color, true_hue, color_name

def get_color_name(hsv_array):
    """
    Given an OpenCV HSV array ([H, S, V] where H: 0-179, S: 0-255, V: 0-255),
    returns a simple, human-readable color name.
    """
    h, s, v = hsv_array[0], hsv_array[1], hsv_array[2]
    
    hue_deg = h * 2  # 0 to 360
    sat_pct = (s / 255.0) * 100
    val_pct = (v / 255.0) * 100
    
    if val_pct < 15:
        return "Black"
    if val_pct > 85 and sat_pct < 15:
        return "White"
    if sat_pct < 35:
        if val_pct < 45: return "Dark Gray"
        if val_pct < 75: return "Gray"
        return "Light Gray"
        
    # Brown/Beige check (Oranges with low sat/val)
    if 10 <= hue_deg <= 45:
        if sat_pct < 40 and val_pct > 60:
            return "Beige"
        if val_pct < 60:
            return "Brown"
            
    # Standard Colors 
    if hue_deg < 10 or hue_deg >= 345:
        if val_pct < 50: return "Burgundy"
        if sat_pct < 50 and val_pct > 70: return "Pink"
        return "Red"
    elif hue_deg < 45:
        return "Orange"
    elif hue_deg < 65:
        if val_pct < 60: return "Olive Green"
        return "Yellow"
    elif hue_deg < 150:
        if val_pct < 50: return "Dark Green"
        if sat_pct < 40 and val_pct > 70: return "Mint Green"
        return "Green"
    elif hue_deg < 200:
        return "Light Blue"
    elif hue_deg < 255:
        if val_pct < 50: return "Navy Blue"
        return "Blue"
    elif hue_deg < 290:
        return "Purple"
    elif hue_deg < 345:
        if val_pct < 50: return "Dark Pink"
        if sat_pct < 40 and val_pct > 70: return "Light Pink"
        return "Pink"

    return "Unknown"
