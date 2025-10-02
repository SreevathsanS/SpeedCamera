import os
import numpy as np

# ====================================================================
# PROJECT DIRECTORIES
# ====================================================================

# This line is crucial: it goes UP one level from this script's location 
# (from /scripts) to establish the /Backend folder as the base directory.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define all output paths relative to the new BASE_DIR
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
VEHICLE_DIR = os.path.join(OUTPUTS_DIR, "vehicles")
ANPR_DIR = os.path.join(OUTPUTS_DIR, "number_plates")
CHALLAN_DIR = os.path.join(OUTPUTS_DIR, "challans")

# Ensure all output directories exist
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(VEHICLE_DIR, exist_ok=True)
os.makedirs(ANPR_DIR, exist_ok=True)
os.makedirs(CHALLAN_DIR, exist_ok=True)

# ====================================================================
# VIDEO AND CAMERA SETTINGS
# ====================================================================

# Paths to video files should be relative to the BASE_DIR
CAMERA_FEEDS = {
    "street_1": os.path.join(BASE_DIR, "data/videos/examples/eg7.mp4"),
    "street_2": os.path.join(BASE_DIR, "data/videos/test9.mp4"),
}



# ====================================================================
# MODEL PATHS
# ====================================================================

# Paths to models should also be relative to the BASE_DIR
VEHICLE_MODEL_PATH = os.path.join(BASE_DIR, "models/VehicleDetector/yolov8m.onnx")
ANPR_MODEL_PATH = os.path.join(BASE_DIR, "models/NumberPlateDetector/best.onnx")
# config1.py

# Show live preview windows (True/False)
SHOW_PREVIEW = True   # set True if you want cv2.imshow()

# Fallback FPS if the video metadata is missing
DEFAULT_FPS = 25.0

# Process every Nth frame (1 = process all frames, 2 = skip half, etc.)
FRAME_SKIP = 2


# ====================================================================
# SPEED CALCULATION AND THRESHOLDS
# ====================================================================

SOURCE_MATRICES = {
    "street_1": np.array([
        [1, 713],
        [1273, 719],
        [994, 276],
        [400, 266]
    ]),
    "street_2": np.array([
        [192, 439],
        [1110, 433],
        [766, 268],
        [420, 262]
    ])
}

SPEED_LIMITS = {
    "street_1": 20,
    "street_2": 50,
}

DISTANCE_PER_PIXEL_METERS = {
    "street_1": 0.2,
    "street_2": 0.25,
}

# ====================================================================
# DATABASE CONFIGURATION
# ====================================================================

DATABASE_CONFIG = {
    "db_type": "sqlite",
    "db_name": os.path.join(OUTPUTS_DIR, "overspeeding_log.db"),
}