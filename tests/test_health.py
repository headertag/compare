import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np
from health import RuntimeHealth, health_issues, notification_plan
from trajectory import TrackingConfig


def state(now=1000):
    return dict(started_at=0, written_at=now, frame_at=now, models=['yolo'],
                expected_models=['yolo'], last_detection_at=now, last_qualified_at=now,
                pane_changed_at={'1': now})


def test_stale_service_and_processing_detected_independently():
    s = state()
    assert not health_issues(s, {}, 1000)
    s['frame_at'] = 700
    assert 'frames' in health_issues(s, {}, 1000)
    s['written_at'] = 700
    assert 'heartbeat' in health_issues(s, {}, 1000)


def test_startup_grace_missing_models_stalled_worker_and_errors():
    s = state();s.update(started_at=950, frame_at=None, models=None)
    assert not health_issues(s, {}, 1000)
    s.update(started_at=0, models=[], media_busy_since=700, media_error='Upload failed')
    issues = health_issues(s, {}, 1000)
    assert {'models', 'frames', 'media_stalled', 'media_error'} <= set(issues)


def test_frozen_pane_and_detection_inactivity_are_advisories():
    s = state(30000);s['pane_changed_at'] = {'1': 29000, '2': 30000}
    s['last_detection_at'] = None
    issues = health_issues(s, {}, 30000)
    assert 'pane_1' in issues and 'pane_2' not in issues and 'no_detections' in issues
    assert 'no_detections' not in health_issues(s, {'no_detection_seconds': 0}, 30000)


def test_notification_dedup_repeat_recovery_and_failed_delivery_retry():
    issues = {'frames': 'stale frames'}
    previous = {'issues': issues, 'sent_at': 1000}
    assert notification_plan(issues, previous, 1001, 3600) is None
    assert notification_plan(issues, previous, 4600, 3600)
    assert 'Recovered' in notification_plan({}, previous, 1001, 3600)
    assert notification_plan(issues, {}, 1001, 3600)  # No persisted successful send
    assert notification_plan({}, {}, 1001, 3600) is None


def test_runtime_records_only_raw_changes_and_skips_unused_panes(tmp_path):
    health = RuntimeHealth({'ignored_panes': [2]}, ['yolo'], tmp_path/'health.json')
    pipeline = SimpleNamespace(preview_boxes=[], pane_scores={},
                               tracking_config=TrackingConfig(rows=1, columns=2))
    raw = np.full((270, 960, 3), 80, np.uint8)
    health.last_write = 0
    health.frame(raw, pipeline, [])
    first = health.state['pane_changed_at'].copy()
    assert set(first) == {'1'}
    noisy = raw.copy();noisy[::2, ::2] += 1
    health.last_write = 0;health.frame(noisy, pipeline, [])
    assert health.state['pane_changed_at'] == first
    noisy[50:100, 50:100] = 200
    health.last_write = 0;health.frame(noisy, pipeline, [])
    assert health.state['pane_changed_at']['1'] >= first['1']
    assert json.loads((tmp_path/'health.json').read_text())['frames'] == 3


def test_health_settings_reject_invalid_thresholds():
    import pytest
    from health import validate_health_settings
    for settings in ({'enabled': 'false'}, {'stale_seconds': -1},
                     {'repeat_seconds': 0}, {'ignored_panes': [0]}):
        with pytest.raises(ValueError): validate_health_settings(settings)
    assert validate_health_settings({'no_detection_seconds': 0})


def test_media_success_clears_previous_health_failure():
    from test_alert_media import sender_without_thread, jpeg
    from alert_media import HistoryFrame
    sender = sender_without_thread(MagicMock())
    sender.health = MagicMock()
    def encode(frames, path, fps): path.write_bytes(b'video')
    with patch('alert_media.encode_alert_video', side_effect=encode):
        sender.send_snapshot((HistoryFrame(jpeg(), 0),))
    status = sender.health.update.call_args.kwargs
    assert status['media_error'] is None and status['last_delivery_recipients'] == 2
