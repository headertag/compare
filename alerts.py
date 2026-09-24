import telepot
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_IDS

def initialize_bot():
    """Initializes and returns the Telegram bot object."""
    return telepot.Bot(TELEGRAM_TOKEN)

def send_alert(bot, image_path='ALERT.jpg'):
    """Sends an image alert to the configured Telegram chats."""
    with open(image_path, 'rb') as alert_image:
        for chat_id in TELEGRAM_CHAT_IDS:
            try:
                bot.sendPhoto(chat_id, alert_image)
                alert_image.seek(0)
            except Exception as e:
                print(f"Failed to send photo to {chat_id}: {e}")


class AlertMediaSender:
    """One active encoding/upload and one queued snapshot; never blocks inference."""
    def __init__(self, bot, media_config, chat_ids=None):
        import queue
        import threading
        self.bot = bot
        self.config = media_config
        self.chat_ids = list(TELEGRAM_CHAT_IDS if chat_ids is None else chat_ids)
        self.queue = queue.Queue(maxsize=1)
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def submit(self, frames):
        import queue
        if not frames:
            return False
        try:
            self.queue.put_nowait(tuple(frames))
            return True
        except queue.Full:
            print('Alert media queue busy; inference continues without queuing another clip.')
            return False

    def _run(self):
        while True:
            frames = self.queue.get()
            try:
                self.send_snapshot(frames)
            except Exception as exc:
                print(f'Alert media failed: {exc}')
            finally:
                self.queue.task_done()

    def send_snapshot(self, frames):
        import io
        import tempfile
        from pathlib import Path
        from alert_media import encode_alert_video
        # Unique files and immutable JPEG bytes avoid concurrent ALERT.jpg races.
        with tempfile.TemporaryDirectory(prefix='compare-alert-') as directory:
            video = Path(directory) / 'trajectory.mp4'
            caption = (f'Person alert: {len(frames)} processed frames, '
                       f'{max(0, frames[-1].timestamp-frames[0].timestamp):.1f}s captured; '
                       f'playback {self.config.playback_fps:g} fps. Insets show confidence-qualified candidates.')
            have_video = False
            if self.config.video_enabled and len(frames) > 1:
                try:
                    encode_alert_video(frames, video, self.config.playback_fps)
                    have_video = True
                except Exception as exc:
                    print(f'Video encoding failed; using magnified snapshot: {exc}')
            for chat_id in self.chat_ids:
                try:
                    if have_video:
                        try:
                            with video.open('rb') as clip:
                                self.bot.sendVideo(chat_id, clip, caption=caption, supports_streaming=True)
                            print(f'Telegram video delivered: {len(frames)} frames, {video.stat().st_size} bytes.', flush=True)
                            continue
                        except Exception as exc:
                            print(f'Video upload failed for {chat_id}; using snapshot: {exc}')
                    image = io.BytesIO(frames[-1].jpeg)
                    image.name = 'person-alert.jpg'
                    self.bot.sendPhoto(chat_id, image)
                except Exception as exc:
                    print(f'Failed to send alert to {chat_id}: {exc}')
