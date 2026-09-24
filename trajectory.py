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
        predict_tracks([self])

    def update(self, box, score, config, frame):
        update_tracks([self], [(box, score)], config, frame)


_TRANSITION = np.eye(8)
_TRANSITION[:4, 4:] = np.eye(4)
_PROCESS_NOISE = np.diag([1.] * 4 + [.25] * 4)
_MEASUREMENT_NOISE = np.eye(4) * 4


def predict_tracks(tracks):
    if not tracks:
        return
    means = np.array([t.mean for t in tracks]) @ _TRANSITION.T
    covariances = np.array([t.covariance for t in tracks])
    covariances = _TRANSITION @ covariances @ _TRANSITION.T + _PROCESS_NOISE
    for track, mean, covariance in zip(tracks, means, covariances):
        track.mean, track.covariance = mean, covariance


def update_tracks(tracks, detections, config, frame):
    """Batch small Kalman solves; only history bookkeeping needs Python per track."""
    if not tracks:
        return
    boxes = np.array([d[0] for d in detections], dtype=float)
    observed = np.concatenate(((boxes[:, :2] + boxes[:, 2:]) / 2,
                               boxes[:, 2:] - boxes[:, :2]), axis=1)
    means = np.array([t.mean for t in tracks])
    covariances = np.array([t.covariance for t in tracks])
    innovation = covariances[:, :4, :4] + _MEASUREMENT_NOISE
    gain = np.linalg.solve(innovation, covariances[:, :4, :]).transpose(0, 2, 1)
    means += (gain @ (observed - means[:, :4])[..., None])[..., 0]
    residual = np.broadcast_to(np.eye(8), covariances.shape).copy()
    residual[:, :, :4] -= gain
    covariances = (residual @ covariances @ residual.transpose(0, 2, 1) +
                   4 * gain @ gain.transpose(0, 2, 1))
    for index, track in enumerate(tracks):
        previous = track.last_box
        threshold = max(config.movement_pixels, config.movement_box_fraction *
                        math.hypot(previous[2] - previous[0], previous[3] - previous[1]))
        distance = math.hypot(observed[index, 0] - (previous[0] + previous[2]) / 2,
                              observed[index, 1] - (previous[1] + previous[3]) / 2)
        # Only consecutive measured motion qualifies; predictions and gaps never do.
        track.movement_frames = track.movement_frames + 1 if distance > threshold and track.missed == 0 else 0
        track.mean, track.covariance = means[index], covariances[index]
        track.last_box = boxes[index]
        track.score = detections[index][1]
        track.history.append((frame, tuple(boxes[index])))
        track.missed = 0
        track.observations += 1


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
    # Compute every pairwise overlap together instead of a Python loop per pair.
    predicted = np.array([t.predicted_box for t in tracks])
    observed = np.array([d[0] for d in detections])
    overlap = np.maximum(0, np.minimum(predicted[:, None, 2:], observed[None, :, 2:]) -
                         np.maximum(predicted[:, None, :2], observed[None, :, :2]))
    intersection = overlap.prod(axis=2)
    track_area = np.maximum(0, predicted[:, 2:] - predicted[:, :2]).prod(axis=1)
    detection_area = np.maximum(0, observed[:, 2:] - observed[:, :2]).prod(axis=1)
    union = track_area[:, None] + detection_area[None, :] - intersection
    costs = 1 - np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
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
        predict_tracks(self.tracks)
        matches, unmatched, unused = associate(self.tracks, high, cfg.match_iou)
        observed = []
        measurements = []
        for ti, di in matches:
            track = self.tracks[ti]
            observed.append(track)
            measurements.append(high[di])
        # Low scores can maintain an active trajectory but cannot create/revive one.
        active = [self.tracks[i] for i in unmatched if self.tracks[i].missed == 0]
        matches_low, _, _ = associate(active, low, cfg.low_match_iou)
        for ti, di in matches_low:
            track = active[ti]
            observed.append(track)
            measurements.append(low[di])
        update_tracks(observed, measurements, cfg, self.frame)
        observed_ids = {track.id for track in observed}
        for track in self.tracks:
            if track.id not in observed_ids:
                track.missed += 1
                track.movement_frames = 0
        self.tracks = [t for t in self.tracks if t.missed <= cfg.max_lost_frames]
        for di in unused:
            track = Track(self.next_id, *high[di], cfg, self.frame)
            self.next_id += 1
            self.tracks.append(track)
            observed.append(track)
        return observed
