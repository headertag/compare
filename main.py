import cv2
import time
import threading
import random
from datetime import datetime
import torch

from config import (
    DEVICE,
    ALERT_SENSITIVITY_THRESHOLD,
    MIN_ALERT_INTERVAL,
    ALERT_COOLDOWN_THRESHOLD,
    ALERT_COOLDOWN,
)
from camera import get_camera_manager
from alerts import initialize_bot, send_alert
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
    Main function to run the object detection and alerting system.
    """
    # Get singleton camera manager
    camera = get_camera_manager()
    camera.start()

    bot = initialize_bot()
    last_alert = 0

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

            if sum(results) >= ALERT_SENSITIVITY_THRESHOLD:
                current_epoch = datetime.now().timestamp()
                time_delta = current_epoch - last_alert
                if time_delta > MIN_ALERT_INTERVAL:
                    if time_delta / MIN_ALERT_INTERVAL < ALERT_COOLDOWN_THRESHOLD:
                        time.sleep(ALERT_COOLDOWN)
                    else:
                        print(f"Alert triggered. Score: {sum(results)}")
                        last_alert = current_epoch
                        if multi_box:
                            for i, box in enumerate(multi_box[:3]):
                                startX, startY, endX, endY = [int(p) for p in box]
                                cv2.rectangle(
                                    img, (startX, startY), (endX, endY), (25 * i, 255, 25 * i), 2
                                )
                        cv2.imwrite("ALERT.jpg", img)
                        alert_thread = threading.Thread(target=send_alert, args=(bot,))
                        alert_thread.daemon = True  # Allow exit without waiting
                        alert_thread.start()

                        # Scramble the seed to prevent sequential bad predictions
                        torch.manual_seed(random.randint(1, 3000000))

            if frame_callback:
                frame_callback(img)

    except KeyboardInterrupt:
        print("Program interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Stop camera (only releases if no other consumers)
        camera.stop()
        print("Main program exiting.")

if __name__ == "__main__":
    main()
