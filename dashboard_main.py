import cv2
import time
import threading
import random
import torch

from config import (
    DEVICE,
    ALERT_SENSITIVITY_THRESHOLD,
    CAM_WIDTH,
    CAM_HEIGHT,
    EXECUTION_MODE,
    INTER_FRAME_DELAY,
)
from camera import get_camera_manager
from model_loader import get_model_pipeline

def main(frame_callback=None):
    """
    Main function to run the object detection and streaming system.
    """
    # Get singleton camera manager
    camera = get_camera_manager()
    camera.start()

    # Get dynamic model pipeline
    pipeline = get_model_pipeline()

    # Give camera time to warm up
    time.sleep(2)

    try:
        while True:
            # Get latest frame from camera manager
            img = camera.get_frame()

            if img is None:
                # No new frame available, wait briefly
                time.sleep(0.01)
                continue

            # Run inference dynamically across all enabled models in pipeline
            results, multi_box = pipeline.run_inference(img, execution_mode=EXECUTION_MODE)

            alert_condition = sum(results) >= ALERT_SENSITIVITY_THRESHOLD
            model_colors = pipeline.get_model_colors()

            if multi_box:
                for box, model_name in multi_box:
                    startX, startY, endX, endY = [int(p) for p in box]
                    color = (0, 255, 0) if alert_condition else model_colors.get(model_name, (255, 255, 255))
                    cv2.rectangle(
                        img, (startX, startY), (endX, endY), color, 2
                    )
                    cv2.putText(img, model_name, (endX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if frame_callback:
                frame_callback(img)

            # Thermal yield delay between frames to prevent continuous 100% duty cycle heat saturation
            if INTER_FRAME_DELAY > 0:
                time.sleep(INTER_FRAME_DELAY)

    except KeyboardInterrupt:
        print("Program interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Stop camera (only releases if no other consumers)
        camera.stop()
        print("Dashboard main exiting.")

if __name__ == "__main__":
    main()
