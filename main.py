import time
import random
from collections import Counter
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
    DEBUG_MODE,
    debug_print,
    config,
    MODELS_CONFIG,
)
from health import RuntimeHealth
from alert_dispatch import AlertDispatcher
from trajectory import pane_bounds
from camera import get_camera_manager
from alerts import initialize_bot, AlertMediaSender
from alert_media import MediaConfig, AlertHistory, draw_person_zoom
from streamer import start_preview_server, get_broadcaster
from model_loader import get_model_pipeline

def main(frame_callback=None):
    """
    Main function to run the object detection and alerting system.
    """
    health = RuntimeHealth(config.get('health', {}),
                           [key for key, value in MODELS_CONFIG.items() if value.get('enabled', True)])
    # Start live preview HTTP server (port 8080)
    start_preview_server(host="0.0.0.0", port=8080)
    broadcaster = get_broadcaster()

    # Get singleton camera manager
    camera = get_camera_manager()
    camera.start()

    # Get dynamic model pipeline
    pipeline = get_model_pipeline()

    health.update(models=[d.key for d in pipeline.detectors], device=str(pipeline.device))

    media = MediaConfig(**{"history_frames": pipeline.tracking_config.history_frames, **ALERT_MEDIA_CONFIG})
    history = AlertHistory(media)
    bot = initialize_bot()
    sender = AlertMediaSender(bot, media, health=health)
    dispatcher = AlertDispatcher(MIN_ALERT_INTERVAL, ALERT_COOLDOWN_THRESHOLD,
                                 int(media.history_max_mb * 1024 * 1024))

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

            health.frame(img, pipeline, results)

            # Explain alert qualification without logging images or credentials.
            if DEBUG_MODE:
                reasons = Counter(track.motion_reason
                                  for tracker in pipeline.trackers.values()
                                  for track in tracker.tracks if track.missed == 0)
                models = Counter(model for _, model, _ in pipeline.preview_boxes)
                debug_print(f"[DETECTION] models={dict(models)} qualified_boxes={len(multi_box)} "
                      f"score={sum(results):.3f}/{ALERT_SENSITIVITY_THRESHOLD:g} "
                      f"tracking={dict(reasons)}")

            # Overlay copies only: raw pixels remain untouched for the next inference.
            display = img.copy()
            pipeline.draw_trajectories(display)
            grid = pipeline.tracking_config if pipeline.tracking_config.enabled else None
            display = draw_person_zoom(img, pipeline.preview_boxes, media, grid, canvas=display,
                                       model_colors=pipeline.get_model_colors())
            jpeg = broadcaster.update_frame(
                display, results=results, threshold=ALERT_SENSITIVITY_THRESHOLD,
                multi_box=multi_box, model_colors=pipeline.get_model_colors(),
                confirmed_panes=pipeline.get_confirmed_panes())
            source = (pipeline.stream_generation, img.shape[:2])
            dispatcher.set_source(source)
            # Tracking clips contain confirmed motion, not unqualified wall candidates.
            evidence_boxes = multi_box if pipeline.tracking_config.enabled else pipeline.preview_boxes
            history.append(jpeg, time.time(), source,
                           has_person=bool(evidence_boxes),
                           person_centroids=tuple(((max(0, min(img.shape[1], b[0])) + max(0, min(img.shape[1], b[2])))/(2*img.shape[1]),
                                                   (max(0, min(img.shape[0], b[1])) + max(0, min(img.shape[0], b[3])))/(2*img.shape[0]))
                                                  for b, *_ in evidence_boxes))

            if pipeline.tracking_config.enabled:
                h, w = img.shape[:2]
                candidates = [(pipeline.pane_scores.get(pane, 0), (x1/w, y1/h, x2/w, y2/h))
                              for pane, x1, y1, x2, y2 in pane_bounds(img.shape, pipeline.tracking_config)]
            else:
                candidates = [(sum(results), (0., 0., 1., 1.))]
            # Every qualifying pane is retained, including events during the interval.
            # Highest score wins simultaneous first arrival; waiting panes remain first.
            for score, focus in sorted(candidates, key=lambda item: item[0], reverse=True):
                if score >= ALERT_SENSITIVITY_THRESHOLD and score > 0:
                    dispatcher.offer(focus, score, history.snapshot(focus))
            event = dispatcher.dispatch(datetime.now().timestamp(), sender)
            if event is not None:
                health.update(last_alert_queued_at=time.time(), alert_focus=event.focus)
                print(f"Alert queued. Score: {event.score:.3f}; focus={event.focus}", flush=True)
                torch.manual_seed(random.randint(1, 3000000))
            debug_print(f"[ALERT] pending_panes={len(dispatcher.pending)} "
                        f"pending_bytes={dispatcher.size_bytes} interval={MIN_ALERT_INTERVAL}")

            if frame_callback:
                frame_callback(display)

            # Thermal yield delay between frames to prevent continuous 100% duty cycle heat saturation
            if INTER_FRAME_DELAY > 0:
                time.sleep(INTER_FRAME_DELAY)

    except KeyboardInterrupt:
        print("Program interrupted by user.")
    except Exception as e:
        health.update(fatal_error=type(e).__name__)
        print(f"An error occurred: {type(e).__name__}", flush=True)
        raise
    finally:
        # Stop camera (only releases if no other consumers)
        camera.stop()
        print("Main program exiting.")

if __name__ == "__main__":
    main()
