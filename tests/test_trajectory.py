import numpy as np
import pytest
from trajectory import TrackingConfig, ByteTracker, Track, associate, pane_bounds


def det(x=10, score=.8, y=10):
    return ([x, y, x + 30, y + 60], score)


def config(**kwargs):
    return TrackingConfig(**dict(min_movement_frames=2, movement_pixels=1,
                                  movement_box_fraction=0, **kwargs))


def test_moving_then_stationary_and_gap():
    tracker = ByteTracker(config())
    first = tracker.update([det()])[0]
    assert first.movement_frames == 0
    assert tracker.update([det(13)])[0].movement_frames == 1
    assert tracker.update([det(16)])[0].movement_frames == 2
    assert tracker.update([det(16)])[0].movement_frames == 0
    tracker.update([det(19)])
    assert tracker.update([]) == []
    assert first.movement_frames == 0
    recovered = tracker.update([det(22)])[0]
    assert recovered.id == first.id
    assert recovered.movement_frames == 0


def test_low_confidence_recovery_and_no_low_births():
    tracker = ByteTracker(config())
    assert tracker.update([det(score=.2)]) == []
    original = tracker.update([det()])[0]
    low = tracker.update([det(13, .2)])[0]
    assert original.id == low.id and low.movement_frames == 1
    assert tracker.update([det(16, .2)])[0].movement_frames == 2
    tracker.update([])
    assert tracker.update([det(19, .2)]) == []


def test_jitter_does_not_qualify():
    tracker = ByteTracker(config())
    for x in [10, 10.5, 10, 10.5] * 5:
        assert tracker.update([det(x)])[0].movement_frames == 0


def test_history_bounded_and_expiry():
    tracker = ByteTracker(config(history_frames=3, max_lost_frames=1))
    for x in range(10, 30, 3):
        track = tracker.update([det(x)])[0]
    assert len(track.history) == 3
    assert track.history[-1][1] == tuple(det(28)[0])
    tracker.update([])
    tracker.update([])
    assert tracker.tracks == []
    assert tracker.update([det(28)])[0].id != track.id


def test_hungarian_preserves_two_identities_when_input_order_changes():
    tracker = ByteTracker(config())
    tracks = tracker.update([det(10), det(80)])
    ids = [t.id for t in tracks]
    tracks = tracker.update([det(77), det(13)])
    assert {t.id: t.last_box[0] for t in tracks} == {ids[0]: 13, ids[1]: 77}


def test_gated_hungarian_does_not_steal_valid_match():
    cfg = config()
    tracks = [Track(1, det(10)[0], .8, cfg, 0), Track(2, det(200)[0], .8, cfg, 0)]
    matches, unmatched, unused = associate(tracks, [det(12)], .2)
    assert matches == [(0, 0)] and unmatched == [1] and unused == []


def test_nondivisible_grid_covers_every_pixel_once():
    pixels = np.zeros((101, 103), int)
    for _, x1, y1, x2, y2 in pane_bounds(pixels.shape, config(rows=3, columns=3)):
        pixels[y1:y2, x1:x2] += 1
    assert np.all(pixels == 1)


@pytest.mark.parametrize('bad', [dict(rows=0), dict(columns=1.5), dict(min_movement_frames=-1),
                                 dict(low_threshold=.8, high_threshold=.3), dict(match_iou=0),
                                 dict(history_frames=0), dict(score_multiplier=.5),
                                 dict(reset_gap_seconds=0), dict(movement_pixels=float('nan'))])
def test_invalid_configuration(bad):
    with pytest.raises(ValueError):
        TrackingConfig(**bad)


@pytest.mark.parametrize('width,height', [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)])
def test_dynamic_widescreen_grid(width, height):
    panes = list(pane_bounds((height, width, 3), config(rows=3, columns=3)))
    assert len(panes) == 9
    assert sum((x2 - x1) * (y2 - y1) for _, x1, y1, x2, y2 in panes) == width * height
    for pane, x1, y1, x2, y2 in panes:
        row, col = divmod(pane, 3)
        assert (x1, x2) == (col * width // 3, (col + 1) * width // 3)
        assert (y1, y2) == (row * height // 3, (row + 1) * height // 3)
        assert abs((x2 - x1) - width / 3) < 1
        assert abs((y2 - y1) - height / 3) < 1
