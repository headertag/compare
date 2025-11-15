import cv2
import threading
import queue
from config import CAM_INDEX, CAM_WIDTH, CAM_HEIGHT

def initialize_camera():
    """Initializes and returns the camera object."""
    cam = cv2.VideoCapture(CAM_INDEX)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    return cam

def camera_reader_thread(camera_object, frame_queue, running_flag):
    """
    Continuously reads frames from the camera and puts the latest into a queue.
    This decouples camera I/O from model inference to prevent blocking.
    """
    while running_flag():
        ret, frame = camera_object.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        # Always keep only the latest frame in the queue
        if not frame_queue.empty():
            try:
                frame_queue.get_nowait()  # Discard old frame
            except queue.Empty:
                pass  # Should not happen with check, but good practice
        frame_queue.put(frame)

    print("Camera reader thread stopped.")
    camera_object.release()  # Release camera when thread stops
