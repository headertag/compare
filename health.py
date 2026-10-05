"""Small, credential-free runtime heartbeat for the independent health checker."""
import json
import os
import threading
import time
from pathlib import Path


def validate_health_settings(settings):
    if not isinstance(settings, dict):
        raise ValueError('health must be a mapping')
    import math
    if not isinstance(settings.get('enabled', True), bool):
        raise ValueError('health.enabled must be boolean')
    for key in ('stale_seconds', 'startup_grace_seconds', 'media_timeout_seconds',
                'pane_stale_seconds', 'no_detection_seconds', 'repeat_seconds'):
        if key in settings:
            value = settings[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (value == 0 and key != 'no_detection_seconds'):
                raise ValueError(f'health.{key} must be a positive number (no_detection_seconds permits 0)')
    panes = settings.get('ignored_panes', [])
    if not isinstance(panes, list) or any(isinstance(p, bool) or not isinstance(p, int) or p < 1 for p in panes):
        raise ValueError('health.ignored_panes must be a list of positive 1-based pane numbers')
    return settings


class RuntimeHealth:
    def __init__(self, settings=None, expected_models=(), path='.runtime/health.json'):
        self.settings = validate_health_settings(settings or {})
        self.path = Path(path)
        self.lock = threading.RLock()
        self.last_write = 0
        self.last_summary = 0
        self.references = {}
        now = time.time()
        self.state = dict(started_at=now, pid=os.getpid(), frame_at=None, frames=0,
                          expected_models=list(expected_models), models=None,
                          last_detection_at=None, last_qualified_at=None,
                          last_alert_queued_at=None, last_delivery_at=None,
                          media_busy_since=None, media_error=None, fatal_error=None,
                          pane_changed_at={}, score=0, detections=0, detection_frames=0, qualified_frames=0)
        self.update()

    def update(self, **values):
        with self.lock:
            self.state.update(values)
            self.state['written_at'] = time.time()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix('.tmp')
            temp.write_text(json.dumps(self.state))
            os.replace(temp, self.path)
            self.last_write = time.time()

    def frame(self, raw, pipeline, scores):
        now = time.time()
        with self.lock:
            self.state['frame_at'] = now
            self.state['frames'] += 1
            self.state['score'] = float(sum(scores))
            self.state['detections'] = len(pipeline.preview_boxes)
            if pipeline.preview_boxes:
                self.state['last_detection_at'] = now
                self.state['detection_frames'] += 1
            if any(pipeline.pane_scores.values()):
                self.state['last_qualified_at'] = now
                self.state['qualified_frames'] += 1
            if now - self.last_write < 5:
                return
            # Inspect raw camera panes, before timestamps, boxes or other app overlays.
            import cv2
            import numpy as np
            from trajectory import pane_bounds
            ignored = self.settings.get('ignored_panes', [])
            for pane, x1, y1, x2, y2 in pane_bounds(raw.shape, pipeline.tracking_config):
                key = str(pane+1)
                if pane+1 in ignored:
                    continue
                crop = raw[y1:y2, x1:x2]
                small = cv2.GaussianBlur(cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),
                                                   (480, 270)), (5, 5), 0)
                previous = self.references.get(key)
                changed = previous is None
                if previous is not None:
                    delta = cv2.absdiff(small, previous)
                    changed = np.count_nonzero(delta > 12) >= 8 or float(delta.mean()) > 1.0
                if changed:
                    self.references[key] = small
                    self.state['pane_changed_at'][key] = now
            self.update()
            if now-self.last_summary >= 60:
                print(f"[HEALTH] frames={self.state['frames']} detection_frames={self.state['detection_frames']} "
                      f"qualified_frames={self.state['qualified_frames']} score={self.state['score']:.3f} "
                      f"models={self.state['models']} last_delivery={self.state['last_delivery_at']}", flush=True)
                self.last_summary = now


def health_issues(state, settings, now=None):
    now = time.time() if now is None else now
    if not state:
        return {'heartbeat': 'No application health heartbeat is available.'}
    issues = {}
    started = state.get('started_at', now)
    age = now-started
    grace = settings.get('startup_grace_seconds', 180)
    stale = settings.get('stale_seconds', 120)
    if age < grace and state.get('frame_at') is None:
        return issues
    if now-state.get('written_at', 0) > stale:
        issues['heartbeat'] = 'Application heartbeat stopped updating; service may be stopped or hung.'
    if now-(state.get('frame_at') or started) > stale:
        issues['frames'] = 'No completed inference frames recently; camera capture or inference may be stalled.'
    expected = set(state.get('expected_models', []))
    loaded = state.get('models')
    if loaded is not None and (not loaded or expected-set(loaded)):
        issues['models'] = 'One or more configured detection models failed to load.'
    if state.get('fatal_error'):
        issues['fatal'] = 'Application reported a fatal processing error: '+state['fatal_error']
    busy = state.get('media_busy_since')
    if busy and now-busy > settings.get('media_timeout_seconds', 180):
        issues['media_stalled'] = 'Alert encoding/upload has not completed; media worker may be stuck.'
    if state.get('media_error'):
        issues['media_error'] = 'Alert media failure: '+state['media_error']
    if 'heartbeat' not in issues and 'frames' not in issues:
        for pane, changed in state.get('pane_changed_at', {}).items():
            if now-changed > settings.get('pane_stale_seconds', 600):
                issues['pane_'+pane] = f'Pane {pane} appears unchanged for at least {settings.get("pane_stale_seconds", 600)//60} minutes. Check its camera/NVR timestamp; a quiet scene is also possible.'
        quiet = settings.get('no_detection_seconds', 21600)
        if quiet > 0 and now-(state.get('last_detection_at') or started) > quiet:
            issues['no_detections'] = f'No person detections for at least {quiet//3600} hours while inference is running. This may be quiet activity or a detection/source problem.'
    return issues


def notification_plan(issues, previous, now, repeat_seconds):
    """Return changed/repeat failures and recoveries; persist only after delivery."""
    old = previous.get('issues', {})
    changed = set(issues) != set(old)
    due = bool(issues) and now-previous.get('sent_at', 0) >= repeat_seconds
    if not changed and not due:
        return None
    parts = []
    if issues:
        parts.append('Camera monitor health warning:\n'+'\n'.join('- '+v for v in issues.values()))
    resolved = set(old)-set(issues)
    if resolved:
        parts.append('Recovered checks: '+', '.join(sorted(resolved)))
    return '\n\n'.join(parts) or None
