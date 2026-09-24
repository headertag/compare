"""Offline GPU validation; never starts the camera, preview server or alert bot.

Example: python scripts/validate_trajectory_video.py --video sample.avi --output validation
Use --controls to freeze pane 8 and show pane 9 for only one frame.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from model_loader import ModelPipeline
from trajectory import TrackingConfig, pane_bounds
from alert_media import MediaConfig, AlertHistory, draw_person_zoom, encode_alert_video


def build_mosaic(frame, frozen, width, height, controls=False, first_frame=False):
    """Fit sources into the same dynamic grid as tracking, without stretching."""
    mosaic = np.zeros((height, width, 3), dtype=np.uint8)
    for pane, x1, y1, x2, y2 in pane_bounds(mosaic.shape, TrackingConfig(rows=3, columns=3)):
        source = frame if pane % 2 == 0 else cv2.flip(frame, 1)
        if controls and pane == 7:
            source = frozen
        elif controls and pane == 8:
            if not first_frame:
                continue
            source = frozen
        h, w = source.shape[:2]
        scale = min((x2 - x1) / w, (y2 - y1) / h)
        target_w, target_h = max(1, round(w * scale)), max(1, round(h * scale))
        tile = cv2.resize(source, (target_w, target_h))
        left = x1 + (x2 - x1 - target_w) // 2
        top = y1 + (y2 - y1 - target_h) // 2
        mosaic[top:top + target_h, left:left + target_w] = tile
    return mosaic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--models', nargs='+', default=['yolo11n.pt', 'yolov8n.pt'])
    parser.add_argument('--execution-mode', choices=['sequential', 'parallel'], default='sequential')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--controls', action='store_true')
    parser.add_argument('--zoom', action='store_true', help='Render magnifiers and export a 60-frame Telegram-style clip')
    args = parser.parse_args()
    if args.frames < 1 or args.stride < 1:
        parser.error('frames and stride must be positive')
    if min(args.width, args.height) < 6 or args.width % 2 or args.height % 2:
        parser.error('width and height must be even and at least 6 for MP4 output')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(rows=3, columns=3, min_movement_frames=3,
                  movement_pixels=1.5, high_threshold=.3, low_threshold=.1,
                  score_multiplier=1.5, history_frames=12, reset_gap_seconds=60)
    pipeline = ModelPipeline(models_config={f'model{i}': dict(type='ultralytics', name=name,
                             confidence_threshold=.15, weight=1) for i, name in enumerate(args.models)},
                             device=torch.device(args.device), tracking_config=config)
    if len(pipeline.detectors) != len(args.models):
        raise RuntimeError('Validation detector failed to load')
    inference_calls = {d.key: 0 for d in pipeline.detectors}
    subthreshold_detections = {d.key: 0 for d in pipeline.detectors}
    for detector in pipeline.detectors:
        original_run = detector.run
        def counted_run(img, scores, boxes, key=detector.key, run=original_run,
                        threshold=detector.confidence_threshold, weight=detector.weight):
            inference_calls[key] += 1
            result = run(img, scores, boxes)
            subthreshold_detections[key] += sum(score / weight <= threshold for score in scores)
            return result
        detector.run = counted_run
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open {args.video}')
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    writer = cv2.VideoWriter(str(output / 'trajectory-3x3.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                             fps / args.stride, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError('Video output could not be opened')
    stats = {str(i): dict(observed=0, qualified=0, qualified_frames=0, boosted_score_max=0.) for i in range(9)}
    frames = 0
    frozen = None
    media = MediaConfig()
    alert_history = AlertHistory(media)
    timings = []
    stage_timings = []
    try:
        while frames < args.frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frozen is None:
                frozen = frame.copy()
            mosaic = build_mosaic(frame, frozen, args.width, args.height,
                                  controls=args.controls, first_frame=frames == 0)
            start = time.perf_counter()
            scores, boxes = pipeline.run_inference(mosaic, args.execution_mode)
            timings.append(time.perf_counter() - start)
            stage_timings.append(pipeline.last_timings.copy())
            qualified_panes = set()
            for (pane, _), tracker in pipeline.trackers.items():
                entry = stats[str(pane)]
                current = [t for t in tracker.tracks if t.missed == 0]
                qualified = [t for t in current if t.score_eligible]
                entry['observed'] += len(current)
                entry['qualified'] += len(qualified)
                if qualified:
                    qualified_panes.add(pane)
                entry['boosted_score_max'] = max(entry['boosted_score_max'], pipeline.pane_scores[pane])
            for pane in qualified_panes:
                stats[str(pane)]['qualified_frames'] += 1
            raw = mosaic.copy() if args.zoom else None
            pipeline.draw_trajectories(mosaic)
            if args.zoom:
                mosaic = draw_person_zoom(raw, pipeline.preview_boxes, media, pipeline.tracking_config, canvas=mosaic,
                                          model_colors=pipeline.get_model_colors())
                ok, encoded = cv2.imencode('.jpg', mosaic, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    alert_history.append(encoded.tobytes(), frames * args.stride / fps, mosaic.shape[:2],
                                         has_person=bool(pipeline.preview_boxes))
            writer.write(mosaic)
            frames += 1
            if frames % 30 == 0:
                print(f'Processed {frames} mosaics; score={sum(scores):.3f}', flush=True)
            for _ in range(args.stride - 1):
                capture.grab()
    finally:
        capture.release()
        writer.release()
    if args.zoom and alert_history.snapshot():
        snapshot = alert_history.snapshot()
        encode_alert_video(snapshot, output / 'alert-history.mp4', media.playback_fps)
        (output / 'zoom-preview.jpg').write_bytes(snapshot[-1].jpeg)
    report = dict(frames=frames, width=args.width, height=args.height, device=str(pipeline.device),
                  gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None,
                  source=args.video, models=args.models, execution_mode=args.execution_mode,
                  tracking=config, controls=args.controls, zoom=args.zoom, alert_history_frames=len(alert_history.snapshot()), inference_calls=inference_calls,
                  confidence_thresholds={d.key: d.confidence_threshold for d in pipeline.detectors},
                  subthreshold_detections=subthreshold_detections,
                  steady_inference_ms=float(np.mean([t['inference_seconds'] for t in stage_timings[1:]])) * 1000 if len(stage_timings) > 1 else None,
                  steady_tracking_ms=float(np.mean([t['tracking_seconds'] for t in stage_timings[1:]])) * 1000 if len(stage_timings) > 1 else None,
                  seconds_per_mosaic=float(np.mean(timings)) if timings else None,
                  steady_seconds_per_mosaic=float(np.mean(timings[1:])) if len(timings) > 1 else None,
                  panes=stats)
    (output / 'validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    assert not any(subthreshold_detections.values()), 'Detector emitted below its original confidence threshold'
    assert all(count == frames for count in inference_calls.values()), 'Expected one full-frame inference per model per frame'
    if args.controls:
        assert frames > 3, 'Too few frames'
        assert stats['7']['observed'] > 0, 'Frozen control must contain a detected person'
        assert stats['8']['observed'] > 0, 'Transient control must contain a detected person'
        assert stats['7']['qualified'] == stats['8']['qualified'] == 0, 'Stationary/transient control qualified'
        assert all(stats[str(i)]['qualified'] > 0 for i in range(7)), 'Moving panes did not qualify'


if __name__ == '__main__':
    main()
