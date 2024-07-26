import os

PLATFORM = "WIN" if os.name == "nt" else "UNIX"

MOTOR_CLASSES = ["Xe So", "Xe Ga", "N/A"]
MOTOR_COLORS = ["Den", "Do", "Xanh", "Trang"]

# Color for drawing bounding boxes of detections
COLOR = {
    0: [172, 47, 117],  # Xe số
    1: [192, 67, 251],  # Xe ga
    2: [195, 103, 9],  # N/A
}

COLOR2 = {
    0: [172, 47, 117],  # Đen
    1: [192, 67, 251],  # Đỏ
    2: [195, 103, 9],  # Xanh
    3: [255, 255, 153],  # Trắng
}

# Blending the mask image with the original image
MASK_ALPHA = 0.3
