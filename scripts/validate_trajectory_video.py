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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--models', nargs='+', default=['yolo11n.pt', 'yolov8n.pt'])
    parser.add_argument('--execution-mode', choices=['sequential', 'parallel'], default='sequential')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--controls', action='store_true')
    args = parser.parse_args()
    if args.frames < 1 or args.stride < 1:
        parser.error('frames and stride must be positive')
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
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open {args.video}')
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    writer = cv2.VideoWriter(str(output / 'trajectory-3x3.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                             fps / args.stride, (1152, 864))
    if not writer.isOpened():
        raise RuntimeError('Video output could not be opened')
    stats = {str(i): dict(observed=0, qualified=0, qualified_frames=0, boosted_score_max=0.) for i in range(9)}
    frames = 0
    frozen = None
    timings = []
    try:
        while frames < args.frames:
            ok, frame = capture.read()
            if not ok:
                break
            tile = cv2.resize(frame, (384, 288))
            if frozen is None:
                frozen = tile.copy()
            tiles = [tile.copy() if i % 2 == 0 else cv2.flip(tile, 1) for i in range(9)]
            if args.controls:
                tiles[7] = frozen.copy()
                tiles[8] = frozen.copy() if frames == 0 else np.zeros_like(tile)
            mosaic = np.vstack([np.hstack(tiles[r:r + 3]) for r in (0, 3, 6)])
            start = time.perf_counter()
            scores, boxes = pipeline.run_inference(mosaic, args.execution_mode)
            timings.append(time.perf_counter() - start)
            qualified_panes = set()
            for (pane, _), tracker in pipeline.trackers.items():
                entry = stats[str(pane)]
                current = [t for t in tracker.tracks if t.missed == 0]
                qualified = [t for t in current if t.movement_frames >= 3 and t.score > .15]
                entry['observed'] += len(current)
                entry['qualified'] += len(qualified)
                if qualified:
                    qualified_panes.add(pane)
                entry['boosted_score_max'] = max(entry['boosted_score_max'], pipeline.pane_scores[pane])
            for pane in qualified_panes:
                stats[str(pane)]['qualified_frames'] += 1
            pipeline.draw_trajectories(mosaic)
            writer.write(mosaic)
            frames += 1
            if frames % 30 == 0:
                print(f'Processed {frames} mosaics; score={sum(scores):.3f}', flush=True)
            for _ in range(args.stride - 1):
                capture.grab()
    finally:
        capture.release()
        writer.release()
    report = dict(frames=frames, device=str(pipeline.device),
                  gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None,
                  source=args.video, models=args.models, execution_mode=args.execution_mode,
                  tracking=config, controls=args.controls,
                  seconds_per_mosaic=float(np.mean(timings)) if timings else None,
                  steady_seconds_per_mosaic=float(np.mean(timings[1:])) if len(timings) > 1 else None,
                  panes=stats)
    (output / 'validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if args.controls:
        assert frames > 3, 'Too few frames'
        assert stats['7']['observed'] > 0, 'Frozen control must contain a detected person'
        assert stats['8']['observed'] > 0, 'Transient control must contain a detected person'
        assert stats['7']['qualified'] == stats['8']['qualified'] == 0, 'Stationary/transient control qualified'
        assert all(stats[str(i)]['qualified'] > 0 for i in range(7)), 'Moving panes did not qualify'


if __name__ == '__main__':
    main()
