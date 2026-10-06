"""Offline, seeded 12-frame reproduction of cross-pane alert suppression.

Procedural images and explicitly scripted detector observations; no camera, model
weight download, Telegram request, production configuration, or service mutation.
The harness runs the application's real main loop and tracking/history code.
"""
from pathlib import Path
import argparse
from contextlib import ExitStack, redirect_stdout
import io
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main as application
from model_loader import BaseDetector, ModelPipeline
from alert_media import AlertHistory, MediaConfig

SEED = 20261005
WIDTH, HEIGHT = 1280, 720
PW, PH = WIDTH // 4, HEIGHT // 4
TRACKING = dict(rows=4, columns=4, min_movement_frames=2, score_multiplier=3,
                movement_pixels=2., movement_box_fraction=.01, low_threshold=.1,
                high_threshold=.4, match_iou=.01, low_match_iou=.01,
                max_lost_frames=15, history_frames=60, reset_gap_seconds=5.,
                draw_history=False)
CASES = ('wall_then_person', 'person_only', 'static_wall', 'fixed_box_lighting',
         'simultaneous', 'short_interval_control', 'late_wall_only', 'flickering_wall')


def fixtures(seed=SEED, case='wall_then_person'):
    rng = np.random.default_rng(seed)
    wall, person = map(int, rng.choice(16, 2, replace=False))
    backgrounds = []
    scenes = []
    for pane in range(16):
        kind = ('office', 'park', 'home')[int(rng.integers(3))]
        scenes.append(kind)
        tile = np.full((PH, PW, 3), (170, 150, 120), np.uint8)
        cv2.rectangle(tile, (0, 110), (PW, PH), (75, 105, 80), -1)
        if kind == 'office':
            cv2.rectangle(tile, (30, 22), (280, 145), (130, 135, 145), -1)
            for y in range(32, 125, 24):
                for x in range(40, 270, 30):
                    cv2.rectangle(tile, (x, y), (x+16, y+15), (80, 65, 40), -1)
        elif kind == 'park':
            cv2.rectangle(tile, (0, 125), (PW, 145), (165, 170, 175), -1)
            for x in rng.integers(20, 300, 5):
                cv2.rectangle(tile, (int(x)-3, 65), (int(x)+3, 120), (40, 65, 80), -1)
                cv2.circle(tile, (int(x), 55), 22, (40, 95, 40), -1)
        else:
            cv2.rectangle(tile, (45, 60), (270, 145), (150, 165, 185), -1)
            cv2.fillPoly(tile, [np.array([[25, 60], [155, 10], [290, 60]])], (65, 75, 95))
            cv2.rectangle(tile, (150, 85), (180, 145), (50, 65, 85), -1)
            cv2.rectangle(tile, (75, 80), (115, 110), (100, 70, 40), -1)
        cv2.putText(tile, f'{pane+1}: {kind}', (5, 17), 0, .4, (255, 255, 255), 1)
        backgrounds.append(tile)
    person_start = 0 if case == 'simultaneous' else 3
    frames, observations, truth = [], [], []
    person_color = tuple(map(int, rng.integers(40, 240, 3)))
    for frame in range(12):
        tiles = [tile.copy() for tile in backgrounds]
        dets, people = [], []
        if case != 'person_only':
            # Stationary masonry. Local alternating IR/shadow changes are NOT a person.
            tile = tiles[wall]
            cv2.rectangle(tile, (65, 32), (250, 165), (110, 110, 110), -1)
            for y in range(40, 164, 15):
                cv2.line(tile, (65, y), (250, y), (65, 65, 65), 1)
            if case != 'static_wall':
                shade = 55 if frame % 2 else 155
                cv2.rectangle(tile, (80, 55), (230, 130), (shade, shade, shade), -1)
            drift = max(0, frame-8) if case == 'late_wall_only' else frame
            x = 85 + (0 if case in ('fixed_box_lighting', 'flickering_wall') else drift*3)
            ox, oy = wall % 4 * PW, wall // 4 * PH
            if case != 'flickering_wall' or frame % 2 == 0:
                dets.append(dict(pane=wall+1, box=[ox+x, oy+40, ox+x+65, oy+155],
                             confidence=.72, truth='wall_false_positive'))
        if case != 'late_wall_only' and person_start <= frame < 9:
            x = 55 + (frame-person_start)*8
            tile = tiles[person]
            color = person_color
            cv2.circle(tile, (x+16, 61), 10, (155, 180, 205), -1)
            cv2.rectangle(tile, (x+5, 73), (x+27, 112), color, -1)
            cv2.line(tile, (x+10, 110), (x, 146), (30, 30, 30), 7)
            cv2.line(tile, (x+23, 110), (x+34, 146), (30, 30, 30), 7)
            ox, oy = person % 4 * PW, person // 4 * PH
            box = [ox+x-4, oy+50, ox+x+39, oy+150]
            people.append(dict(pane=person+1, box=box))
            dets.append(dict(pane=person+1, box=box, confidence=.94, truth='person'))
        frames.append(np.vstack([np.hstack(tiles[r*4:r*4+4]) for r in range(4)]))
        observations.append(dets)
        truth.append(people)
    return frames, observations, dict(seed=seed, wall_pane=wall+1, person_pane=person+1,
                                     scene_types=scenes, ground_truth=truth)


class FixtureDetector(BaseDetector):
    def __init__(self, observations):
        super().__init__('fixture', dict(confidence_threshold=.6, weight=5), torch.device('cpu'))
        self.observations, self.calls = observations, 0

    def run(self, img, scores, boxes):
        for entry in (self.observations[self.calls] if self.calls < 12 else []):
            if entry['confidence'] > self.candidate_threshold:
                scores.append(entry['confidence']*self.weight)
                boxes.append((entry['box'], self.key))
        self.calls += 1


def reproduce(case='wall_then_person', seed=SEED, output=None, drain_pending=False):
    frames, observations, metadata = fixtures(seed, case)
    pipeline = ModelPipeline({}, torch.device('cpu'), tracking_config=TRACKING)
    detector = FixtureDetector(observations)
    pipeline.detectors = [detector]
    state = SimpleNamespace(index=-1)
    records, submissions, histories, dispatchers = [], [], [], []
    total_frames = 410 if drain_pending else 12
    from alert_dispatch import AlertDispatcher
    def dispatcher_factory(*args):
        dispatcher = AlertDispatcher(*args)
        dispatchers.append(dispatcher)
        return dispatcher
    step_seconds = 1 / 3.2
    interval_seconds = .6 if case == 'short_interval_control' else 120

    class Clock:
        @staticmethod
        def now():
            return SimpleNamespace(timestamp=lambda: 10000+state.index*step_seconds)

    class Camera:
        def start(self): pass
        def stop(self): pass
        def get_frame(self):
            state.index += 1
            assert state.index < total_frames, 'main loop did not terminate after fixture'
            return frames[min(state.index, 11)].copy()

    class Health:
        def __init__(self, *a, **kw): pass
        def frame(self, *a): pass
        def update(self, **kw): pass

    class Sender:
        def __init__(self, *a, **kw): pass
        def submit(self, snapshot):
            assert snapshot
            submissions.append(dict(frame=state.index, focus=list(snapshot[0].focus_box),
                                    clip_frames=len(snapshot)))
            return True

    def history_factory(config):
        history = AlertHistory(config)
        histories.append(history)
        return history

    def encode(frame, **kw):
        ok, jpeg = cv2.imencode('.jpg', frame)
        assert ok
        return jpeg.tobytes()

    def observe(frame):
        records.append(dict(frame=state.index, seconds=state.index*step_seconds,
                            accepted=len(pipeline.preview_boxes),
                            pane_scores={str(p+1):float(s) for p,s in pipeline.pane_scores.items() if s},
                            focus=list(pipeline.get_alert_focus()),
                            tracks=[dict(pane=p+1, reason=t.motion_reason, confidence=t.score,
                                         visual_fraction=t.visual_fraction)
                                    for (p,_),tracker in pipeline.trackers.items()
                                    for t in tracker.tracks if t.missed == 0]))
        if state.index == total_frames-1: raise KeyboardInterrupt

    overrides = dict(get_camera_manager=lambda:Camera(), get_model_pipeline=lambda:pipeline,
                     RuntimeHealth=Health, start_preview_server=lambda **kw:None,
                     get_broadcaster=lambda:SimpleNamespace(update_frame=encode),
                     initialize_bot=lambda:None, AlertMediaSender=Sender,
                     AlertHistory=history_factory, AlertDispatcher=dispatcher_factory, datetime=Clock, DEBUG_MODE=False,
                     EXECUTION_MODE='sequential', INTER_FRAME_DELAY=0,
                     ALERT_SENSITIVITY_THRESHOLD=2.3, MIN_ALERT_INTERVAL=interval_seconds,
                     ALERT_COOLDOWN_THRESHOLD=1, ALERT_COOLDOWN=0,
                     ALERT_MEDIA_CONFIG=dict(debug_mode=False, zoom_enabled=False))
    with ExitStack() as stack, redirect_stdout(io.StringIO()):
        for name,value in overrides.items():stack.enter_context(patch.object(application,name,value))
        stack.enter_context(patch.object(application.time,'sleep',lambda _:None))
        stack.enter_context(patch('model_loader.DEBUG_MODE',False))
        stack.enter_context(patch('model_loader.debug_print',lambda *a:None))
        stack.enter_context(patch('socket.create_connection', side_effect=AssertionError('Network forbidden in synthetic tests')))
        stack.enter_context(patch('cv2.VideoCapture', side_effect=AssertionError('Camera forbidden in synthetic tests')))
        stack.enter_context(torch.random.fork_rng(devices=[]))
        application.main(frame_callback=observe)
    assert detector.calls == total_frames
    wp = metadata['wall_pane']-1
    focus = (wp%4/4, wp//4/4, (wp%4+1)/4, (wp//4+1)/4)
    wall_clip = histories[0].snapshot(focus)
    result = dict(case=case, **metadata, step_seconds=step_seconds, interval_seconds=interval_seconds, frames=records,
                  submissions=submissions, wall_history_frames=len(wall_clip),
                  pending_focus=[list(k) for k in dispatchers[0].pending],
                  detector_calls=detector.calls, observations=observations,
                  detector_is_scripted=True)
    if output:
        folder=Path(output)/case;folder.mkdir(parents=True,exist_ok=True)
        for i,frame in enumerate(frames):cv2.imwrite(str(folder/f'frame-{i:02}.png'),frame)
        sheet=np.vstack([np.hstack([cv2.resize(f,(320,180)) for f in frames[r*3:r*3+3]]) for r in range(4)])
        cv2.imwrite(str(folder/'contact-sheet.png'),sheet)
        (folder/'result.json').write_text(json.dumps(result,indent=2))
        from PIL import Image
        gif=[Image.fromarray(cv2.cvtColor(cv2.resize(f,(960,540)),cv2.COLOR_BGR2RGB)) for f in frames]
        gif[0].save(folder/'sequence.gif',save_all=True,append_images=gif[1:],duration=650,loop=0)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    summary=[]
    for case in CASES:
        r=reproduce(case,output=args.output)
        summary.append({k:r[k] for k in ('case','wall_pane','person_pane','submissions','wall_history_frames')})
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
