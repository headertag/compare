import cv2
import time
import threading
import random
import torch

from config import (
    DEVICE,
    ALERT_SENSITIVITY_THRESHOLD,
)
from camera import get_camera_manager
from model_loader import (
    run_detr,
    run_yolos,
    run_torchvision_model,
    run_yolov5,
    frcnn_model,
    retinanet_model,
    MODELS_CONFIG,
)

def main(frame_callback=None):
    """
    Main function to run the object detection and streaming system.
    """
    # Get singleton camera manager
    camera = get_camera_manager()
    camera.start()

    # Give camera time to warm up
    time.sleep(2)

    model_colors = {
        "detr": (255, 0, 0),  # Red
        "yolos": (0, 0, 255),  # Blue
        "frcnn": (204, 204, 0),  # Yellow
        "retinanet": (0, 204, 204),  # Cyan
        "yolov5": (255, 102, 0),  # Orange
    }

    try:
        while True:
            # Get latest frame from camera manager
            img = camera.get_frame()

            if img is None:
                # No new frame available, wait briefly
                time.sleep(0.01)
                continue

            results = []
            multi_box = []

            # The ensemble approach: running multiple models in parallel
            threads = [
                threading.Thread(target=run_detr, args=(img, results, multi_box)),
                threading.Thread(target=run_yolos, args=(img, results, multi_box)),
                threading.Thread(
                    target=run_torchvision_model,
                    args=(
                        frcnn_model,
                        img,
                        results,
                        multi_box,
                        MODELS_CONFIG["frcnn_resnet"]["confidence_threshold"],
                        "frcnn",
                    ),
                ),
                threading.Thread(
                    target=run_torchvision_model,
                    args=(
                        retinanet_model,
                        img,
                        results,
                        multi_box,
                        MODELS_CONFIG["retinanet"]["confidence_threshold"],
                        "retinanet",
                    ),
                ),
                threading.Thread(target=run_yolov5, args=(img, results, multi_box)),
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            alert_condition = sum(results) >= ALERT_SENSITIVITY_THRESHOLD

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
