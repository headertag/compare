import io
import threading
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest
from alert_media import MediaConfig, AlertHistory, HistoryFrame, draw_person_zoom, encode_alert_video
from alerts import AlertMediaSender
from trajectory import TrackingConfig


def jpeg(value=0, width=96, height=64):
    ok, data = cv2.imencode('.jpg', np.full((height, width, 3), value, np.uint8))
    assert ok
    return data.tobytes()


def test_zoom_uses_raw_pixels_without_mutating_inference_input():
    raw = np.zeros((180, 240, 3), np.uint8)
    raw[80:100, 90:100] = [0, 255, 0]
    saved = raw.copy()
    canvas = np.zeros_like(raw)
    result = draw_person_zoom(raw, [([90, 80, 100, 100], 'a')], MediaConfig(), canvas=canvas)
    assert np.array_equal(raw, saved)
    assert np.count_nonzero(canvas) == 0
    assert np.count_nonzero(np.all(result == [0, 255, 0], axis=2)) > 200


def test_cross_model_zoom_duplicates_are_suppressed():
    raw = np.full((180, 240, 3), 100, np.uint8)
    a = ([90, 80, 100, 100], 'a')
    b = ([90, 80, 100, 100], 'a')
    assert np.array_equal(draw_person_zoom(raw, [a], MediaConfig()),
                          draw_person_zoom(raw, [a, b], MediaConfig()))


def test_zoom_never_spills_into_another_camera_pane():
    raw = np.full((180, 400, 3), 100, np.uint8)
    result = draw_person_zoom(raw, [([190, 80, 230, 100], 'a')], MediaConfig(),
                              TrackingConfig(rows=1, columns=2))
    assert np.array_equal(result[:, :200], raw[:, :200])
    assert not np.array_equal(result[:, 200:], raw[:, 200:])


def test_sixty_frame_history_is_chronological_bounded_and_immutable():
    history = AlertHistory(MediaConfig())
    for i in range(65):
        history.append(jpeg(i), i, 'camera')
    snapshot = history.snapshot()
    assert [f.timestamp for f in snapshot] == list(range(5, 65))
    history.append(jpeg(100), 100, 'new-camera')
    assert len(history.snapshot()) == 1
    assert len(snapshot) == 60 and snapshot[-1].timestamp == 64


def test_history_byte_budget_and_invalid_encode():
    data = jpeg()
    history = AlertHistory(MediaConfig(history_max_mb=len(data)*2/1024/1024))
    for i in range(5):
        history.append(data, i, 'a')
    assert len(history.snapshot()) == 2
    assert history.size_bytes == len(data)*2
    history.append(None, 6, 'a')
    assert len(history.snapshot()) == 2


def test_real_h264_clip_has_all_sixty_frames(tmp_path):
    frames = tuple(HistoryFrame(jpeg(i*3, width=95, height=63), i) for i in range(60))
    path = tmp_path / 'alert.mp4'
    encode_alert_video(frames, path, 5)
    video = cv2.VideoCapture(str(path))
    decoded = []
    while True:
        ok, image = video.read()
        if not ok:
            break
        decoded.append(image)
    video.release()
    assert len(decoded) == 60
    assert decoded[0].shape[:2] == (64, 96)
    assert decoded[-1].mean() > decoded[0].mean() + 100
    blob = path.read_bytes()
    assert b'avc1' in blob  # H.264, not OpenCV's unsupported-in-some-clients mp4v
    assert blob.index(b'moov') < blob.index(b'mdat')  # Streamable header


def sender_without_thread(bot, config=None):
    sender = object.__new__(AlertMediaSender)
    sender.bot = bot
    sender.config = config or MediaConfig()
    sender.chat_ids = [1, 2]
    return sender


def test_video_encoding_failure_falls_back_to_independent_photo_streams():
    bot = MagicMock()
    sender = sender_without_thread(bot)
    frames = (HistoryFrame(jpeg(10), 0), HistoryFrame(jpeg(20), 1))
    sent = []
    bot.sendPhoto.side_effect = lambda chat, photo, **kw: sent.append((chat, photo.read()))
    with patch('alert_media.encode_alert_video', side_effect=RuntimeError('no encoder')):
        sender.send_snapshot(frames)
    assert sent == [(1, frames[-1].jpeg), (2, frames[-1].jpeg)]
    bot.sendVideo.assert_not_called()
    assert all("caption" not in call.kwargs for call in bot.sendPhoto.call_args_list)


def test_video_upload_failure_falls_back_only_for_failed_recipient(tmp_path):
    bot = MagicMock()
    sender = sender_without_thread(bot)
    def encode(frames, path, fps):
        path.write_bytes(b'video')
    videos = []
    photos = []
    def upload(chat, clip, **kwargs):
        videos.append((chat, clip.read()))
        if chat == 1:
            raise RuntimeError('upload failed')
    bot.sendVideo.side_effect = upload
    bot.sendPhoto.side_effect = lambda chat, photo, **kw: photos.append(chat)
    with patch('alert_media.encode_alert_video', side_effect=encode):
        sender.send_snapshot((HistoryFrame(jpeg(), 0), HistoryFrame(jpeg(), 1)))
    assert videos == [(1, b'video'), (2, b'video')]
    assert all("caption" not in call.kwargs for call in bot.sendVideo.call_args_list)
    assert photos == [1]


def test_submission_remains_nonblocking_with_a_busy_encoder():
    started, release = threading.Event(), threading.Event()
    def blocked(self, frames):
        started.set()
        release.wait(5)
    with patch.object(AlertMediaSender, 'send_snapshot', blocked):
        sender = AlertMediaSender(MagicMock(), MediaConfig(), chat_ids=[])
        frame = (HistoryFrame(jpeg(), 0),)
        try:
            assert sender.submit(frame)
            assert started.wait(2)
            assert sender.submit(frame)
            assert not sender.submit(frame)  # Bounded queue; returns without waiting
        finally:
            release.set()
            sender.queue.join()


def test_preview_encoded_jpeg_is_the_same_one_saved_and_buffered(tmp_path):
    from streamer import PreviewBroadcaster
    broadcaster = PreviewBroadcaster()
    broadcaster.shm_path = str(tmp_path / 'preview.jpg')
    raw = np.zeros((180, 240, 3), np.uint8)
    raw[80:100, 90:100] = [0, 255, 0]
    display = draw_person_zoom(raw, [([90, 80, 100, 100], 'a')], MediaConfig())
    encoded = broadcaster.update_frame(display)
    assert encoded == broadcaster.get_jpeg() == (tmp_path / 'preview.jpg').read_bytes()
    assert not np.array_equal(cv2.imdecode(np.frombuffer(encoded, np.uint8), 1), raw)


def test_default_zoom_doubles_previous_dimensions_when_pane_has_room():
    raw = np.zeros((1080, 1920, 3), np.uint8)
    box = ([200, 200, 220, 240], 'person')
    original_resize = cv2.resize
    with patch('alert_media.cv2.resize', wraps=original_resize) as resize:
        draw_person_zoom(raw, [box], MediaConfig(), TrackingConfig(rows=3, columns=3))
    assert resize.call_args.args[1] == (120, 240)


def test_empty_observations_are_omitted_but_age_out_old_people():
    history = AlertHistory(MediaConfig(history_frames=3))
    history.append(jpeg(10), 0, 'a', has_person=True)
    history.append(jpeg(20), 1, 'a', has_person=False)
    history.append(jpeg(30), 2, 'a', has_person=True)
    assert [f.timestamp for f in history.snapshot()] == [0, 2]
    assert history.size_bytes == len(jpeg(10)) + len(jpeg(30))
    history.append(jpeg(40), 3, 'a', has_person=False)
    assert [f.timestamp for f in history.snapshot()] == [2]
    history.append(jpeg(), 4, 'new-source', has_person=False)
    assert history.snapshot() == () and history.size_bytes == 0


def test_single_person_frame_sends_one_fps_video(tmp_path):
    bot = MagicMock()
    sender = sender_without_thread(bot)
    received = []
    def upload(chat, clip, **kwargs):
        path = tmp_path / f'{chat}.mp4'
        path.write_bytes(clip.read())
        cap = cv2.VideoCapture(str(path))
        received.append((cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)))
        cap.release()
    bot.sendVideo.side_effect = upload
    sender.send_snapshot((HistoryFrame(jpeg(), 0),))
    assert received == [(1, 1), (1, 1)]
    bot.sendPhoto.assert_not_called()


def test_zoom_labels_keep_each_models_raw_confidence_and_color():
    raw = np.full((1080, 1920, 3), 100, np.uint8)
    detections = [([200, 200, 250, 280], 'frcnn', .87),
                  ([200, 200, 250, 280], 'yolo', .63)]
    colors = {'frcnn': (0, 0, 255), 'yolo': (50, 205, 50)}
    with patch('alert_media.cv2.putText', wraps=cv2.putText) as text, \
         patch('alert_media.cv2.resize', wraps=cv2.resize) as resize:
        draw_person_zoom(raw, detections, MediaConfig(), model_colors=colors)
    assert resize.call_count == 1
    calls = {call.args[1]: call.args for call in text.call_args_list}
    for model, percentage in [('frcnn', '87%'), ('yolo', '63%')]:
        assert calls[model][5] == calls[percentage][5] == colors[model]
        assert calls[model][2][0] < calls[percentage][2][0]
        assert calls[model][2][1] == calls[percentage][2][1]
        assert calls[model][4] >= .7


def test_debug_mode_defaults_on_and_can_hide_only_inset_labels():
    assert MediaConfig().debug_mode is True
    with pytest.raises(ValueError, match='debug_mode must be boolean'):
        MediaConfig(debug_mode='false')
    raw = np.full((540, 960, 3), 100, np.uint8)
    with patch('alert_media.cv2.putText', wraps=cv2.putText) as text, \
         patch('alert_media.cv2.resize', wraps=cv2.resize) as resize:
        result = draw_person_zoom(raw, [([200, 200, 250, 280], 'yolo', .9)],
                                  MediaConfig(debug_mode=False))
    text.assert_not_called()
    assert resize.call_count == 1
    assert not np.array_equal(result, raw)
