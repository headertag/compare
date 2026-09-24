"""Adversarial motion checks: detector geometry is not visual movement."""
import cv2
import numpy as np
import pytest
import torch
from model_loader import BaseDetector, ModelPipeline
from trajectory import TrackingConfig, ByteTracker, VisualMotion, pane_bounds, qualify_track


def textured_frame():
    rng = np.random.default_rng(19)
    return rng.integers(30, 150, (120, 160, 3), dtype=np.uint8)


def test_large_oscillating_boxes_never_qualify():
    cfg = TrackingConfig(min_movement_frames=3)
    tracker = ByteTracker(cfg)
    for x in [30, 34, 30, 34, 30, 34] * 4:
        track = tracker.update([([x, 20, x + 30, 80], .9)])[0]
        assert track.movement_frames < 3


def test_slow_consistent_trajectory_accumulates():
    cfg = TrackingConfig(min_movement_frames=3)
    tracker = ByteTracker(cfg)
    for x in np.arange(30., 34.5, .5):
        track = tracker.update([([x, 20, x + 30, 80], .9)])[0]
    assert track.movement_frames >= 3  # No individual shift exceeded the old 2px cutoff


def test_image_difference_ignores_box_motion_and_uniform_light_change():
    cfg = TrackingConfig(min_movement_frames=3)
    motion = VisualMotion(cfg)
    image = textured_frame()
    panes = list(pane_bounds(image.shape, cfg))
    motion.update(image, panes)
    motion.update(image.copy(), panes)
    assert motion.fraction(0, [34, 20, 64, 80], [30, 20, 60, 80]) == 0
    motion.update((image.astype(np.int16) + 25).astype(np.uint8), panes)
    assert motion.fraction(0, [34, 20, 64, 80], [30, 20, 60, 80]) == 0


def test_image_motion_remains_pane_local():
    cfg = TrackingConfig(rows=1, columns=2, min_movement_frames=3)
    motion = VisualMotion(cfg)
    before = np.zeros((100, 200, 3), np.uint8)
    after = before.copy()
    after[20:80, 20:60] = 255
    panes = list(pane_bounds(before.shape, cfg))
    motion.update(before, panes)
    motion.update(after, panes)
    assert motion.fraction(0, [20, 20, 60, 80], [20, 20, 60, 80]) > .5
    assert motion.fraction(1, [20, 20, 60, 80], [20, 20, 60, 80]) == 0


class CandidateDetector(BaseDetector):
    def __init__(self):
        super().__init__('candidate', dict(confidence_threshold=.5), torch.device('cpu'))
        self.box = [20, 20, 50, 80]
        self.calls = 0

    def run(self, image, scores, boxes):
        self.calls += 1
        scores.append(.6)
        boxes.append((self.box, self.key))


def pipeline():
    result = ModelPipeline({}, torch.device('cpu'), tracking_config=dict(min_movement_frames=3))
    result.detectors = [CandidateDetector()]
    return result


@pytest.mark.parametrize('path', [list(range(20, 65, 3)), [20, 24, 20, 24] * 6])
def test_static_scene_with_drifting_or_jittering_boxes_never_scores(path):
    p = pipeline()
    image = textured_frame()
    for x in path:
        p.detectors[0].box = [x, 20, x + 30, 80]
        scores, boxes = p.run_inference(image.copy())
        assert scores == [] and boxes == []
        assert not p.trackers[(0, 'candidate')].tracks[0].score_eligible
    assert p.detectors[0].calls == len(path)


def test_accepted_translation_gets_boost_and_stops():
    p = pipeline()
    results = []
    for x in range(20, 44, 3):
        image = np.zeros((120, 160, 3), np.uint8)
        image[20:80, x:x + 30] = 220
        p.detectors[0].box = [x, 20, x + 30, 80]
        scores, _ = p.run_inference(image)
        results.append(scores)
    assert results[0] == []
    assert results[-1] == pytest.approx([.9])  # Raw .6 already passed the original .5 cutoff
    assert p.run_inference(image.copy())[0] == []
    # Detector drifts over that stopped object: geometry alone must not revive it.
    for x in range(44, 56, 3):
        p.detectors[0].box = [x, 20, x + 30, 80]
        assert p.run_inference(image.copy())[0] == []


def test_visual_evidence_required_on_multiple_frames():
    cfg = TrackingConfig(min_movement_frames=3)
    tracker = ByteTracker(cfg)
    for x in (20, 23, 26, 29):
        track = tracker.update([([x, 20, x + 30, 80], .6)])[0]
        assert not qualify_track(track, cfg, .5, 0.)
    assert not qualify_track(track, cfg, .5, .2)  # One flash cannot supply a trajectory


def test_other_motion_in_same_pane_does_not_activate_static_region():
    cfg = TrackingConfig(min_movement_frames=3)
    motion = VisualMotion(cfg)
    before = np.zeros((100, 200, 3), np.uint8)
    after = before.copy()
    after[20:80, 20:100] = 255
    panes = list(pane_bounds(before.shape, cfg))
    motion.update(before, panes)
    motion.update(after, panes)
    assert motion.fraction(0, [140, 20, 180, 80], [140, 20, 180, 80]) == 0


def test_small_pixel_noise_cannot_verify_box_drift():
    cfg = TrackingConfig(min_movement_frames=3)
    motion = VisualMotion(cfg)
    image = textured_frame()
    rng = np.random.default_rng(2)
    noisy = (image.astype(int) + rng.integers(-4, 5, image.shape)).astype(np.uint8)
    panes = list(pane_bounds(image.shape, cfg))
    motion.update(image, panes)
    motion.update(noisy, panes)
    assert motion.fraction(0, [34, 20, 64, 80], [30, 20, 60, 80]) == 0


def test_subpixel_motion_can_pass_full_pipeline():
    p = pipeline()
    for x in np.arange(20., 24.5, .5):
        image = np.zeros((120, 160, 3), np.uint8)
        image[20:80, int(x):int(x) + 30] = 220
        p.detectors[0].box = [x, 20, x + 30, 80]
        scores, _ = p.run_inference(image)
    assert scores == pytest.approx([.9])


def test_verified_movement_cannot_rescue_subthreshold_confidence():
    cfg = TrackingConfig(min_movement_frames=3, score_multiplier=10, require_visual_motion=False)
    tracker = ByteTracker(cfg)
    for x in (20, 23, 26, 29):
        track = tracker.update([([x, 20, x + 30, 80], .6)])[0]
    assert track.movement_frames >= 3
    assert not qualify_track(track, cfg, .8, 1.)
    assert track.motion_reason == 'below original confidence threshold'
