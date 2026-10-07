import time
import telepot
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_IDS, debug_print

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
    """One active encoding/upload; busy submissions are discarded without buffering."""
    def __init__(self, bot, media_config, chat_ids=None, health=None):
        import queue
        import threading
        self.health = health
        self.bot = bot
        self.config = media_config
        self.chat_ids = list(TELEGRAM_CHAT_IDS if chat_ids is None else chat_ids)
        self.queue = queue.Queue(maxsize=1)
        self.slot = threading.BoundedSemaphore(1)
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def submit(self, frames):
        import queue
        if not frames or not self.slot.acquire(blocking=False):
            return False
        try:
            self.queue.put_nowait(tuple(frames))
            debug_print(f"[MEDIA] queued frames={len(frames)} fps={self.config.playback_fps}")
            return True
        except queue.Full:
            self.slot.release()
            print('Alert media queue busy; inference continues without queuing another clip.')
            return False

    def _run(self):
        while True:
            frames = self.queue.get()
            try:
                self._health(media_busy_since=time.time())
                self.send_snapshot(frames)
            except Exception as exc:
                self._health(media_error='Worker failed: '+type(exc).__name__)
                print(f'Alert media failed: {type(exc).__name__}', flush=True)
            finally:
                self._health(media_busy_since=None)
                self.slot.release()
                self.queue.task_done()

    def _health(self, **values):
        if getattr(self, "health", None) is not None:
            self.health.update(**values)

    def send_snapshot(self, frames):
        import io
        import tempfile
        from pathlib import Path
        from alert_media import encode_alert_video
        # Unique files and immutable JPEG bytes avoid concurrent ALERT.jpg races.
        with tempfile.TemporaryDirectory(prefix='compare-alert-') as directory:
            video = Path(directory) / 'trajectory.mp4'
            failures = []
            delivered = 0
            have_video = False
            if self.config.video_enabled and frames:
                try:
                    debug_print(f"[MEDIA] encoding frames={len(frames)} fps={self.config.playback_fps}")
                    encode_alert_video(frames, video, self.config.playback_fps)
                    have_video = True
                except Exception as exc:
                    failures.append('Video encoding failed: '+type(exc).__name__)
                    print(f'Video encoding failed; using magnified snapshot: {type(exc).__name__}', flush=True)
            for chat_id in self.chat_ids:
                try:
                    if have_video:
                        try:
                            with video.open('rb') as clip:
                                self.bot.sendVideo(chat_id, clip, supports_streaming=True)
                            delivered += 1
                            print(f'Telegram video delivered: {len(frames)} frames, {video.stat().st_size} bytes.', flush=True)
                            continue
                        except Exception as exc:
                            failures.append('Video upload failed: '+type(exc).__name__)
                            print(f'Video upload failed; using snapshot: {type(exc).__name__}', flush=True)
                    image = io.BytesIO(frames[-1].jpeg)
                    image.name = 'person-alert.jpg'
                    self.bot.sendPhoto(chat_id, image)
                    delivered += 1
                    print('[MEDIA] photo delivered', flush=True)
                except Exception as exc:
                    failures.append('Alert delivery failed: '+type(exc).__name__)
                    print(f'Alert delivery failed: {type(exc).__name__}', flush=True)

            status = {'media_error': '; '.join(sorted(set(failures))) or None,
                      'last_delivery_recipients': delivered, 'last_clip_frames': len(frames)}
            if delivered:
                status['last_delivery_at'] = time.time()
            self._health(**status)
