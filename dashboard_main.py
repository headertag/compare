import cv2
import time
import threading
import random
import queue
import torch

from config import (
    DEVICE,
)
from camera import initialize_camera, camera_reader_thread
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
    # Initialize camera
    cam = initialize_camera()

    # Initialize frame queue and camera thread
    latest_frame_queue = queue.Queue(maxsize=1)
    camera_thread_running = True

    # Start camera reader thread
    capture_thread = threading.Thread(
        target=camera_reader_thread,
        args=(cam, latest_frame_queue, lambda: camera_thread_running)
    )
    capture_thread.daemon = True
    capture_thread.start()
    print("Camera reader thread started.")

    # Give camera time to warm up and fill queue
    time.sleep(2)

    try:
        while True:
            img = None
            try:
                # Get latest frame from queue without blocking
                img = latest_frame_queue.get_nowait()
            except queue.Empty:
                # No new frame available, wait briefly
                time.sleep(0.01)
                continue

            if img is None:
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
                    ),
                ),
                threading.Thread(target=run_yolov5, args=(img, results, multi_box)),
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            if multi_box:
                for i, box in enumerate(multi_box[:3]):
                    startX, startY, endX, endY = [int(p) for p in box]
                    cv2.rectangle(
                        img, (startX, startY), (endX, endY), (25 * i, 255, 25 * i), 2
                    )

            if frame_callback:
                frame_callback(img)

    except KeyboardInterrupt:
        print("Program interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Stop camera thread and cleanup
        camera_thread_running = False
        if capture_thread and capture_thread.is_alive():
            capture_thread.join(timeout=2)
        print("Dashboard main exiting.")

if __name__ == "__main__":
    main()
