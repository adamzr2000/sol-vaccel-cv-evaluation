import numpy as np

# ==========================================
# 1. COLOR MAP (Standard VOC Colormap)
# ==========================================
COLORS = np.array([
    (0, 0, 0),       # Background
    (128, 0, 0),     # Aeroplane
    (0, 128, 0),     # Bicycle
    (128, 128, 0),   # Bird
    (0, 0, 128),     # Boat
    (128, 0, 128),   # Bottle
    (0, 128, 128),   # Bus
    (128, 128, 128), # Car
    (64, 0, 0),      # Cat
    (192, 0, 0),     # Chair
    (64, 128, 0),    # Cow
    (192, 128, 0),   # Dining Table
    (64, 0, 128),    # Dog
    (192, 0, 128),   # Horse
    (64, 128, 128),  # Motorbike
    (192, 128, 128), # Person
    (0, 64, 0),      # Potted Plant
    (128, 64, 0),    # Sheep
    (0, 192, 0),     # Sofa
    (128, 192, 0),   # Train
    (0, 64, 128)     # TV Monitor
], dtype=np.uint8)

# ==========================================
# 2. CLASS NAMES (Pascal VOC Index 0-20)
# ==========================================
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def get_position_label(cy, cx, h, w):
    """
    Returns a string like 'Top-Left', 'Center', 'Bottom-Right' based on centroids.
    """
    vert = "Center"
    if cy < h / 3: vert = "Top"
    elif cy > 2 * h / 3: vert = "Bottom"
    
    horz = "" 
    if cx < w / 3: horz = "-Left"
    elif cx > 2 * w / 3: horz = "-Right"
    
    if vert == "Center" and horz == "": return "Center"
    if vert == "Center": return horz.replace("-", "") 
    return f"{vert}{horz}"

def analyze_segmentation_mask(mask_idx):
    """
    Analyzes a segmentation mask (numpy array) and returns a formatted string 
    describing detected objects, their positions, and screen coverage %.
    """
    unique, counts = np.unique(mask_idx, return_counts=True)
    total_pixels = mask_idx.size
    H, W = mask_idx.shape
    
    detected_list = []
    
    for cls_id, count in zip(unique, counts):
        if cls_id == 0: continue # Skip background
        
        # Calculate Percentage
        percentage = (count / total_pixels) * 100
        
        # Heuristic: Ignore objects smaller than 1% of the screen (reduces noise)
        if percentage > 1.0: 
            # Get Name
            class_name = VOC_CLASSES[cls_id] if cls_id < len(VOC_CLASSES) else str(cls_id)
            
            # Get Position (Centroid)
            ys, xs = np.where(mask_idx == cls_id)
            cy, cx = np.mean(ys), np.mean(xs)
            pos_str = get_position_label(cy, cx, H, W)
            
            detected_list.append(f"{class_name} ({pos_str}): {int(percentage)}%")
            
    if detected_list:
        return " -> [" + ", ".join(detected_list) + "]"
    else:
        return " -> [background only]"