"""Pane-local ByteTrack association and observed-motion qualification.

Each detector has its own tracker in each pane: ensemble votes must never count
as extra video frames. Coordinates and history remain local to that pane.
"""
from collections import deque
from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class TrackingConfig:
    rows: int = 1
    columns: int = 1
    min_movement_frames: int = 0
    movement_pixels: float = 2.0
    movement_box_fraction: float = 0.01
    low_threshold: float = 0.1
    high_threshold: float = 0.4
    match_iou: float = 0.2
    low_match_iou: float = 0.3
    max_lost_frames: int = 15
    history_frames: int = 60
    reset_gap_seconds: float = 5.0
    draw_history: bool = True
    score_multiplier: float = 1.5

    def __post_init__(self):
        for name in ('rows', 'columns', 'min_movement_frames', 'max_lost_frames', 'history_frames'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f'tracking.{name} must be an integer')
            if value < (1 if name in ('rows', 'columns', 'history_frames') else 0):
                raise ValueError(f'tracking.{name} is out of range')
        for name in ('movement_pixels', 'movement_box_fraction', 'reset_gap_seconds'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'tracking.{name} must be finite and nonnegative')
        if not math.isfinite(self.score_multiplier) or self.score_multiplier < 1:
            raise ValueError('tracking.score_multiplier must be finite and >= 1')
        if self.reset_gap_seconds == 0:
            raise ValueError('tracking.reset_gap_seconds must be positive')
        if not 0 <= self.low_threshold <= self.high_threshold <= 1:
            raise ValueError('tracking thresholds must satisfy 0 <= low <= high <= 1')
        for name in ('match_iou', 'low_match_iou'):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f'tracking.{name} must be in (0, 1]')

    @property
    def enabled(self):
        return self.min_movement_frames > 0


def pane_bounds(shape, config):
    """Partition actual frame dimensions; no fixed resolution or aspect ratio."""
    height, width = shape[:2]
    if config.rows > height or config.columns > width:
        raise ValueError('Tracking grid cannot have more panes than image pixels')
    for row in range(config.rows):
        for col in range(config.columns):
            yield (row * config.columns + col,
                   col * width // config.columns, row * height // config.rows,
                   (col + 1) * width // config.columns, (row + 1) * height // config.rows)


def measurement(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])


class Track:
    """Eight-state constant-velocity Kalman filter (cx, cy, w, h, velocities)."""
    def __init__(self, track_id, box, score, config, frame):
        self.id = track_id
        self.mean = np.r_[measurement(box), np.zeros(4)]
        self.covariance = np.diag([10.] * 4 + [100.] * 4)
        self.history = deque(maxlen=config.history_frames)
        self.history.append((frame, tuple(box)))
        self.last_box = np.array(box, dtype=float)
        self.score = score
        self.missed = 0
        self.movement_frames = 0
        self.observations = 1

    @property
    def predicted_box(self):
        cx, cy, w, h = self.mean[:4]
        w, h = max(w, 1), max(h, 1)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def predict(self):
        transition = np.eye(8)
        transition[:4, 4:] = np.eye(4)
        self.mean = transition @ self.mean
        self.covariance = transition @ self.covariance @ transition.T + np.diag([1.] * 4 + [.25] * 4)

    def update(self, box, score, config, frame):
        observed = measurement(box)
        previous = measurement(self.last_box)
        threshold = max(config.movement_pixels,
                        config.movement_box_fraction * np.linalg.norm(previous[2:]))
        # Only consecutive measured motion qualifies; predictions and gaps never do.
        moved = np.linalg.norm(observed[:2] - previous[:2]) > threshold
        self.movement_frames = self.movement_frames + 1 if moved and self.missed == 0 else 0
        innovation = self.covariance[:4, :4] + np.eye(4) * 4
        gain = np.linalg.solve(innovation, self.covariance[:4, :]).T
        self.mean += gain @ (observed - self.mean[:4])
        residual = np.eye(8)
        residual[:, :4] -= gain
        self.covariance = residual @ self.covariance @ residual.T + 4 * gain @ gain.T
        self.last_box = np.array(box, dtype=float)
        self.score = score
        self.history.append((frame, tuple(box)))
        self.missed = 0
        self.observations += 1


def iou(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    start = np.maximum(a[:2], b[:2])
    end = np.minimum(a[2:], b[2:])
    intersection = np.prod(np.maximum(end - start, 0))
    union = np.prod(np.maximum(a[2:] - a[:2], 0)) + np.prod(np.maximum(b[2:] - b[:2], 0)) - intersection
    return float(intersection / union) if union > 0 else 0.0


def associate(tracks, detections, threshold):
    """Gated Hungarian matching, with explicit unmatched choices per track."""
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))
    costs = np.array([[1 - iou(t.predicted_box, d[0]) for d in detections] for t in tracks])
    valid = costs <= 1 - threshold
    # Large unmatched cost maximizes valid match cardinality before minimizing cost.
    penalty = len(tracks) + len(detections) + 1
    augmented = np.full((len(tracks), len(detections) + len(tracks)), float(penalty))
    augmented[:, :len(detections)] = np.where(valid, costs, penalty * 3)
    rows, cols = linear_sum_assignment(augmented)
    matches = [(int(r), int(c)) for r, c in zip(rows, cols) if c < len(detections) and valid[r, c]]
    return (matches, [r for r in range(len(tracks)) if r not in {m[0] for m in matches}],
            [c for c in range(len(detections)) if c not in {m[1] for m in matches}])


class ByteTracker:
    """High-confidence association, then low-confidence recovery of active tracks."""
    def __init__(self, config):
        self.config = config
        self.tracks = []
        self.frame = 0
        self.next_id = 1

    def update(self, detections):
        self.frame += 1
        cfg = self.config
        high = [(np.asarray(b, float), s) for b, s in detections if s >= cfg.high_threshold]
        low = [(np.asarray(b, float), s) for b, s in detections if cfg.low_threshold <= s < cfg.high_threshold]
        for track in self.tracks:
            track.predict()
        matches, unmatched, unused = associate(self.tracks, high, cfg.match_iou)
        observed = []
        for ti, di in matches:
            track = self.tracks[ti]
            track.update(*high[di], cfg, self.frame)
            observed.append(track)
        # Low scores can maintain an active trajectory but cannot create/revive one.
        active = [self.tracks[i] for i in unmatched if self.tracks[i].missed == 0]
        matches_low, _, _ = associate(active, low, cfg.low_match_iou)
        for ti, di in matches_low:
            track = active[ti]
            track.update(*low[di], cfg, self.frame)
            observed.append(track)
        for track in self.tracks:
            if track not in observed:
                track.missed += 1
                track.movement_frames = 0
        self.tracks = [t for t in self.tracks if t.missed <= cfg.max_lost_frames]
        for di in unused:
            track = Track(self.next_id, *high[di], cfg, self.frame)
            self.next_id += 1
            self.tracks.append(track)
            observed.append(track)
        return observed
