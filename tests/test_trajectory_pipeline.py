import numpy as np
import pytest
import torch
from model_loader import BaseDetector, ModelPipeline


class ScriptedDetector(BaseDetector):
    def __init__(self, key='fake', weight=1):
        super().__init__(key, dict(confidence_threshold=.15, weight=weight), torch.device('cpu'))
        self.x = 10
        self.score = .8
        self.panes = None
        self.calls = 0

    def run(self, img, results, boxes):
        self.calls += 1
        self.last_shape = img.shape
        from trajectory import pane_bounds, TrackingConfig
        for pane, x1, y1, _, _ in pane_bounds(img.shape, TrackingConfig(rows=3, columns=3)):
            if self.panes is None or pane in self.panes:
                results.append(self.score * self.weight)
                boxes.append(([x1 + self.x, y1 + 10, x1 + self.x + 30, y1 + 60], self.key))
                if not self.collect_all:
                    break


def make_pipeline(**options):
    cfg = dict(rows=3, columns=3, min_movement_frames=2, movement_pixels=1,
               movement_box_fraction=0, net_displacement_box_fraction=0,
               require_visual_motion=False, score_multiplier=1.5)
    cfg.update(options)
    pipeline = ModelPipeline({}, torch.device('cpu'), tracking_config=cfg)
    pipeline.detectors = [ScriptedDetector()]
    frame = np.zeros((300, 300, 3), np.uint8)
    for i in range(9):
        frame[i // 3 * 100:(i // 3 + 1) * 100, i % 3 * 100:(i % 3 + 1) * 100] = i
    return pipeline, frame


@pytest.mark.parametrize('mode', ['sequential', 'parallel'])
def test_nine_panes_boost_once_per_model_and_keep_coordinates_local(mode):
    pipeline, frame = make_pipeline()
    pipeline.detectors.append(ScriptedDetector('second', .5))
    for x in (10, 13, 16):
        for d in pipeline.detectors:
            d.x = x
        scores, boxes = pipeline.run_inference(frame, mode)
        if x < 16:
            assert scores == [] and boxes == []
    assert scores == pytest.approx([1.2, .6])
    assert len(boxes) == 18
    assert all(d.calls == 3 and d.last_shape == frame.shape for d in pipeline.detectors)
    assert len(pipeline.trackers) == 18
    assert boxes[-1][0] == [216, 210, 246, 260]
    for tracker in pipeline.trackers.values():
        assert len(tracker.tracks[0].history) == 3
        assert tracker.tracks[0].history[0][1] == (10, 10, 40, 60)
    pipeline.draw_trajectories(frame.copy())


def test_models_in_unrelated_panes_do_not_combine_votes():
    pipeline, frame = make_pipeline()
    pipeline.detectors[0].panes = {0}
    second = ScriptedDetector('other')
    second.panes = {8}
    pipeline.detectors.append(second)
    for x in (10, 13, 16):
        for d in pipeline.detectors:
            d.x = x
        scores, _ = pipeline.run_inference(frame)
    assert sum(scores) == pytest.approx(1.2)


def test_pane_change_needs_new_history():
    pipeline, frame = make_pipeline()
    detector = pipeline.detectors[0]
    detector.panes = {0}
    for x in (10, 13, 16):
        detector.x = x
        pipeline.run_inference(frame)
    detector.panes = {1}
    assert pipeline.run_inference(frame)[0] == []
    assert pipeline.trackers[(1, 'fake')].tracks[0].movement_frames == 0


def test_zero_disables_grid_history_and_boost():
    pipeline, frame = make_pipeline(min_movement_frames=0)
    scores, boxes = pipeline.run_inference(frame)
    assert scores == [.8] and len(boxes) == 1
    assert pipeline.trackers == {}
    assert pipeline.detectors[0].calls == 1
    assert pipeline.detectors[0].candidate_threshold == .15


def test_low_confidence_candidates_can_extend_and_boost():
    pipeline, frame = make_pipeline()
    detector = pipeline.detectors[0]
    pipeline.run_inference(frame)
    detector.score = .2
    for x in (13, 16):
        detector.x = x
        scores, _ = pipeline.run_inference(frame)
    assert scores == pytest.approx([.3])
    detector.x = 19
    detector.score = .12
    assert pipeline.run_inference(frame)[0] == pytest.approx([.18])  # Boost crosses .15 floor


def test_resize_and_idle_gap_reset_history():
    pipeline, frame = make_pipeline()
    detector = pipeline.detectors[0]
    for x in (10, 13, 16):
        detector.x = x
        pipeline.run_inference(frame)
    pipeline._last_frame_time -= 10
    assert pipeline.run_inference(frame)[0] == []
    assert pipeline.run_inference(frame[:270])[0] == []
    pipeline.reset_tracking()
    assert pipeline.trackers == {}


def test_parallel_detector_failure_propagates():
    pipeline, frame = make_pipeline()
    def fail(*args):
        raise RuntimeError('inference failed')
    pipeline.detectors[0].run = fail
    with pytest.raises(RuntimeError, match='inference failed'):
        pipeline.run_inference(frame, 'parallel')


def test_crowd_does_not_multiply_model_weight():
    pipeline, frame = make_pipeline()
    detector = pipeline.detectors[0]
    original = detector.run
    def crowd(img, scores, boxes):
        original(img, scores, boxes)
        for pane in range(9):
            x1, y1 = pane % 3 * 100, pane // 3 * 100
            scores.append(.7)
            boxes.append(([x1 + detector.x + 40, y1 + 10, x1 + detector.x + 65, y1 + 60], detector.key))
    detector.run = crowd
    for x in (10, 13, 16):
        detector.x = x
        scores, boxes = pipeline.run_inference(frame)
    assert scores == pytest.approx([1.2])
    assert len(boxes) == 18


def test_invalid_candidates_are_dropped_and_history_is_clipped():
    pipeline, frame = make_pipeline()
    detector = pipeline.detectors[0]
    def invalid(img, scores, boxes):
        for box in ([float('nan'), 0, 20, 20], [30, 0, 10, 20], [-10, -10, 110, 110]):
            scores.append(.8)
            boxes.append((box, detector.key))
    detector.run = invalid
    pipeline.run_inference(frame)
    assert sum(len(t.tracks) for t in pipeline.trackers.values()) == 1
    assert pipeline.trackers[(0, 'fake')].tracks[0].history[0][1] == (0, 0, 100, 100)


def test_live_resolution_change_rebuilds_local_tracking():
    pipeline, _ = make_pipeline()
    detector = pipeline.detectors[0]
    for width, height in [(1920, 1080), (3840, 2160), (1280, 720)]:
        frame = np.zeros((height, width, 3), np.uint8)
        for step, x in enumerate((10, 13, 16)):
            detector.x = x
            scores, boxes = pipeline.run_inference(frame)
            if step == 0:
                assert scores == []
                assert all(len(t.tracks[0].history) == 1 for t in pipeline.trackers.values())
        assert len(boxes) == 9
        assert boxes[-1][0] == [16 + 2 * width // 3, 10 + 2 * height // 3,
                                46 + 2 * width // 3, 60 + 2 * height // 3]
        assert scores == pytest.approx([1.2])


def test_widescreen_demo_preserves_source_proportions():
    from scripts.validate_trajectory_video import build_mosaic
    source = np.full((300, 400, 3), 255, np.uint8)
    mosaic = build_mosaic(source, source, 1920, 1080)
    assert mosaic.shape == (1080, 1920, 3)
    # 4:3 footage fits as 480x360 within each 640x360 pane, with 80px side bars.
    assert np.all(mosaic[:360, :80] == 0)
    assert np.all(mosaic[:360, 80:560] == 255)
    assert np.all(mosaic[:360, 560:640] == 0)
    assert np.array_equal(mosaic[:360, :640], mosaic[720:, 1280:])


@pytest.mark.parametrize('mode', ['sequential', 'parallel'])
@pytest.mark.parametrize('grid', [(1, 1), (3, 3), (4, 5)])
def test_inference_count_independent_of_grid_size(mode, grid):
    pipeline, frame = make_pipeline(rows=grid[0], columns=grid[1])
    pipeline.detectors.append(ScriptedDetector('second'))
    for _ in range(4):
        pipeline.run_inference(frame, mode)
    assert all(d.calls == 4 and d.last_shape == frame.shape for d in pipeline.detectors)


@pytest.mark.parametrize('box,pane,local', [
    ([90, 10, 110, 60], 1, (0, 10, 10, 60)),
    ([85, 10, 105, 60], 0, (85, 10, 100, 60)),
    ([10, 90, 60, 110], 3, (10, 0, 60, 10)),
])
def test_seam_box_assigned_once_by_center(box, pane, local):
    pipeline, frame = make_pipeline()
    def detect(img, scores, boxes):
        scores.append(.8)
        boxes.append((box, 'fake'))
    pipeline.detectors[0].run = detect
    pipeline.run_inference(frame)
    assert sum(len(t.tracks) for t in pipeline.trackers.values()) == 1
    assert pipeline.trackers[(pane, 'fake')].tracks[0].history[0][1] == local
