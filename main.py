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
    CAM_WIDTH,
    CAM_HEIGHT,
    EXECUTION_MODE,
    INTER_FRAME_DELAY,
)
from camera import get_camera_manager
from alerts import initialize_bot, send_alert
from streamer import start_preview_server, get_broadcaster
from model_loader import get_model_pipeline

def main(frame_callback=None):
    """
    Main function to run the object detection and alerting system.
    """
    # Start live preview HTTP server (port 8080)
    start_preview_server(host="0.0.0.0", port=8080)
    broadcaster = get_broadcaster()

    # Get singleton camera manager
    camera = get_camera_manager()
    camera.start()

    # Get dynamic model pipeline
    pipeline = get_model_pipeline()

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

            # Run inference dynamically across all enabled models in pipeline
            results, multi_box = pipeline.run_inference(img, execution_mode=EXECUTION_MODE)

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
                            for i, (box_coords, model_name) in enumerate(multi_box[:3]):
                                startX, startY, endX, endY = [int(p) for p in box_coords]
                                cv2.rectangle(
                                    img, (startX, startY), (endX, endY), (25 * i, 255, 25 * i), 2
                                )
                        cv2.imwrite("ALERT.jpg", img)
                        alert_thread = threading.Thread(target=send_alert, args=(bot,))
                        alert_thread.daemon = True  # Allow exit without waiting
                        alert_thread.start()

                        # Scramble the seed to prevent sequential bad predictions
                        torch.manual_seed(random.randint(1, 3000000))

            # Broadcast latest frame with bounding boxes and detection metrics to HTTP preview
            broadcaster.update_frame(
                img,
                results=results,
                threshold=ALERT_SENSITIVITY_THRESHOLD,
                multi_box=multi_box,
                model_colors=pipeline.get_model_colors(),
            )

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
        print("Main program exiting.")

if __name__ == "__main__":
    main()
