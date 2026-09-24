import time
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
    ALERT_MEDIA_CONFIG,
)
from camera import get_camera_manager
from alerts import initialize_bot, AlertMediaSender
from alert_media import MediaConfig, AlertHistory, draw_person_zoom
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

    media = MediaConfig(**{"history_frames": pipeline.tracking_config.history_frames, **ALERT_MEDIA_CONFIG})
    history = AlertHistory(media)
    bot = initialize_bot()
    sender = AlertMediaSender(bot, media)
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

            # Overlay copies only: raw pixels remain untouched for the next inference.
            display = img.copy()
            pipeline.draw_trajectories(display)
            grid = pipeline.tracking_config if pipeline.tracking_config.enabled else None
            display = draw_person_zoom(img, pipeline.preview_boxes, media, grid, canvas=display,
                                       model_colors=pipeline.get_model_colors())
            jpeg = broadcaster.update_frame(
                display, results=results, threshold=ALERT_SENSITIVITY_THRESHOLD,
                multi_box=multi_box, model_colors=pipeline.get_model_colors())
            history.append(jpeg, time.time(), (pipeline.stream_generation, img.shape[:2]),
                           has_person=bool(pipeline.preview_boxes))

            if sum(results) >= ALERT_SENSITIVITY_THRESHOLD:
                current_epoch = datetime.now().timestamp()
                time_delta = current_epoch - last_alert
                if time_delta > MIN_ALERT_INTERVAL:
                    if time_delta / MIN_ALERT_INTERVAL < ALERT_COOLDOWN_THRESHOLD:
                        time.sleep(ALERT_COOLDOWN)
                    elif jpeg is not None and sender.submit(history.snapshot()):
                        print(f"Alert queued. Score: {sum(results)}")
                        last_alert = current_epoch
                        torch.manual_seed(random.randint(1, 3000000))

            if frame_callback:
                frame_callback(display)

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
