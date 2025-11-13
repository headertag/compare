import cv2
from config import CAM_INDEX, CAM_WIDTH, CAM_HEIGHT

def initialize_camera():
    """Initializes and returns the camera object."""
    cam = cv2.VideoCapture(CAM_INDEX)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    return cam
