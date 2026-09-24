"""Display-only person magnifiers and a bounded, immutable alert-video buffer."""
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import math
import cv2
import numpy as np

from trajectory import iou, pane_bounds, TrackingConfig


@dataclass(frozen=True)
class MediaConfig:
    zoom_enabled: bool = True
    zoom_factor: float = 6.0
    zoom_max_pane_fraction: float = 1.0
    max_zoom_per_pane: int = 0
    video_enabled: bool = True
    history_frames: int = 60
    history_max_mb: float = 64.0
    playback_fps: float = 5.0

    def __post_init__(self):
        for name in ('zoom_enabled', 'video_enabled'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'alert_media.{name} must be boolean')
        for name in ('history_frames', 'max_zoom_per_pane'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == 'max_zoom_per_pane' else 1):
                raise ValueError(f'alert_media.{name} must be a valid integer')
        for name in ('zoom_factor', 'zoom_max_pane_fraction', 'history_max_mb', 'playback_fps'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'alert_media.{name} must be finite and positive')
        if self.zoom_factor < 1 or self.zoom_max_pane_fraction > 1 or self.playback_fps > 30:
            raise ValueError('Invalid alert_media zoom factor, pane fraction or playback fps')


def draw_person_zoom(raw, boxes, config, grid=None, canvas=None):
    """Magnify accepted detections from raw pixels, never from another inset.

Deduplicate overlapping model boxes and keep the magnifiers in their source
pane. Insets are display cues, not additional detections or alert evidence.
"""
    result = raw.copy() if canvas is None else canvas.copy()
    if not config.zoom_enabled:
        return result
    panes = list(pane_bounds(raw.shape, grid or TrackingConfig()))
    grouped = {pane: [] for pane, *_ in panes}
    height, width = raw.shape[:2]
    for box, model in boxes:
        box = np.asarray(box, float)
        if box.shape != (4,) or not np.isfinite(box).all():
            continue
        x1, y1, x2, y2 = box
        x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
        y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        cx, cy = (x1+x2)/2, (y1+y2)/2
        for pane, px1, py1, px2, py2 in panes:
            if px1 <= cx < px2 and py1 <= cy < py2:
                clipped = [max(x1, px1), max(y1, py1), min(x2, px2), min(y2, py2)]
                if not any(iou(clipped, previous) >= .5 for previous in grouped[pane]):
                    grouped[pane].append(clipped)
                break
    for pane, px1, py1, px2, py2 in panes:
        # The live broadcaster uses a 36px status banner across the top row.
        safe_top = min(py2, 36) if py1 == 0 else py1
        placed = []
        for box in grouped[pane][:config.max_zoom_per_pane or None]:
            x1, y1, x2, y2 = map(int, box)
            crop = raw[y1:y2, x1:x2]
            if not crop.size:
                continue
            factor = min(config.zoom_factor,
                         (px2-px1)*config.zoom_max_pane_fraction/crop.shape[1],
                         (py2-py1)*config.zoom_max_pane_fraction/crop.shape[0],
                         (py2-safe_top)/crop.shape[0])
            if factor <= 1:
                continue  # Already large enough to see.
            w, h = max(1, round(crop.shape[1]*factor)), max(1, round(crop.shape[0]*factor))
            # Prefer above/alongside the source; select the least overlapping inset.
            options = [(x1, y1-h-8), (x2+8, y1), (x1-w-8, y1), (x1, y2+8)]
            rectangles = []
            for left, top in options:
                left = max(px1, min(px2-w, left))
                top = max(safe_top, min(py2-h, top))
                rectangles.append([left, top, left+w, top+h])
            target = min(rectangles, key=lambda b: sum(iou(b, p) for p in placed) + iou(b, box))
            left, top, right, bottom = target
            placed.append(target)
            color = (255, 210, 60)
            cv2.line(result, ((x1+x2)//2, (y1+y2)//2), ((left+right)//2, (top+bottom)//2), color, 1)
            result[top:bottom, left:right] = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
            # All decoration is clipped to the same pane, too.
            pane_img = result[py1:py2, px1:px2]
            cv2.rectangle(pane_img, (left-px1, top-py1), (right-px1-1, bottom-py1-1), color, 2)
            cv2.rectangle(pane_img, (x1-px1, y1-py1), (x2-px1-1, y2-py1-1), color, 1)
            cv2.putText(pane_img, f'{factor:.1f}x person candidate', (left-px1+3, top-py1+14),
                        cv2.FONT_HERSHEY_SIMPLEX, .38, color, 1)
    return result


@dataclass(frozen=True)
class HistoryFrame:
    jpeg: bytes
    timestamp: float


class AlertHistory:
    def __init__(self, config):
        self.config = config
        self.frames = deque()
        self.size_bytes = 0
        self.source = None

    def append(self, jpeg, timestamp, source):
        if source != self.source:
            self.frames.clear()
            self.size_bytes = 0
            self.source = source
        if not jpeg:
            return
        frame = HistoryFrame(bytes(jpeg), timestamp)
        self.frames.append(frame)
        self.size_bytes += len(frame.jpeg)
        while self.frames and (len(self.frames) > self.config.history_frames or
                               self.size_bytes > self.config.history_max_mb*1024*1024):
            self.size_bytes -= len(self.frames.popleft().jpeg)

    def snapshot(self):
        return tuple(self.frames)


def encode_alert_video(frames, path, fps):
    """Encode chronologically in a worker, with bounded decode memory and CPU use."""
    import imageio_ffmpeg
    if not frames:
        raise ValueError('No alert frames available')
    first = cv2.imdecode(np.frombuffer(frames[0].jpeg, np.uint8), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError('Invalid alert JPEG')
    height, width = first.shape[:2]
    even_w, even_h = width + width%2, height + height%2
    writer = imageio_ffmpeg.write_frames(str(path), (even_w, even_h), fps=fps,
        codec='libx264', pix_fmt_in='bgr24', pix_fmt_out='yuv420p', macro_block_size=1,
        ffmpeg_log_level='error', output_params=['-preset', 'veryfast', '-crf', '23',
                                              '-threads', '2', '-movflags', '+faststart'])
    writer.send(None)
    try:
        for entry in frames:
            image = cv2.imdecode(np.frombuffer(entry.jpeg, np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (height, width):
                raise ValueError('Invalid/mismatched alert frame')
            image = cv2.copyMakeBorder(image, 0, height%2, 0, width%2, cv2.BORDER_CONSTANT)
            writer.send(np.ascontiguousarray(image))
    finally:
        writer.close()
    if not Path(path).is_file() or not 0 < Path(path).stat().st_size < 49_000_000:
        raise ValueError('Alert video missing or exceeds upload budget')
