"""Hard process deadlines for media encoding and Telegram file uploads."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


class MediaDeliveryError(Exception):
    def __init__(self, description, *, retryable=False, retry_after=0):
        super().__init__(description)
        self.retryable = retryable
        self.retry_after = retry_after


def run_bounded(command, payload, timeout):
    if timeout <= 0:
        raise TimeoutError('Media delivery deadline reached')
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, start_new_session=os.name == 'posix')
    try:
        output, _ = process.communicate(payload, timeout=timeout)
    except BaseException:
        if os.name == 'posix':
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
        else:
            process.kill()
        process.communicate()
        raise
    if process.returncode:
        raise MediaDeliveryError('Media subprocess exited unsuccessfully')
    return output


class BoundedTelegramBot:
    def __init__(self, token):
        self.token = token

    def _send(self, method, chat, media, timeout_seconds):
        import tempfile
        # BytesIO photos and file-backed videos use the same isolated transport.
        with tempfile.TemporaryDirectory(prefix='compare-upload-') as folder:
            name = Path(folder)/('media.mp4' if method == 'sendVideo' else 'media.jpg')
            source = getattr(media, 'name', None)
            if isinstance(source, str) and Path(source).is_file():
                name = Path(source)
            else:
                name.write_bytes(media.read())
            request = dict(method=method, token=self.token, chat=chat, path=str(name))
            result = json.loads(run_bounded(
                [sys.executable, str(Path(__file__).resolve()), 'upload'],
                json.dumps(request).encode(), timeout_seconds))
            if not result['ok']:
                code = result.get('code')
                reason = result.get('description', 'Telegram transport failed')
                raise MediaDeliveryError(f'Telegram {code}: {reason}' if code else reason,
                                         retryable=result.get('retryable', False),
                                         retry_after=result.get('retry_after', 0))

    def sendVideo(self, chat, media, *, supports_streaming=True, timeout_seconds=20):
        return self._send('sendVideo', chat, media, timeout_seconds)

    def sendPhoto(self, chat, media, *, timeout_seconds=20):
        return self._send('sendPhoto', chat, media, timeout_seconds)


def upload_worker():
    import re
    import telepot
    import urllib3
    request = json.load(sys.stdin)
    # No hidden library retry loop: the parent owns attempts and backoff.
    telepot.api._pools['default'] = urllib3.PoolManager(retries=False)
    cls, options = telepot.api._onetime_pool_spec
    telepot.api._onetime_pool_spec = (cls, {**options, 'retries': False})
    bot = telepot.Bot(request['token'])
    try:
        with open(request['path'], 'rb') as media:
            if request['method'] == 'sendVideo':
                bot.sendVideo(request['chat'], media, supports_streaming=True)
            else:
                bot.sendPhoto(request['chat'], media)
        result = {'ok': True}
    except telepot.exception.TelegramError as exc:
        code = exc.error_code
        description = str(exc.description).replace(request['token'], '[TOKEN]').replace(str(request['chat']), '[RECIPIENT]')
        description = re.sub(r'https?://\S+', '[URL]', description)[:200]
        result = dict(ok=False, code=code, description=description,
                      retryable=code == 429 or code >= 500,
                      retry_after=exc.json.get('parameters', {}).get('retry_after', 0))
    except Exception as exc:
        result = dict(ok=False, description=type(exc).__name__,
                      retryable=isinstance(exc, (OSError, urllib3.exceptions.HTTPError)))
    print(json.dumps(result))


if __name__ == '__main__':
    if sys.argv[1] == 'upload':
        upload_worker()
    elif sys.argv[1] == 'encode':
        import pickle
        from alert_media import encode_alert_video
        # Only trusted parent-created payloads are read on this private stdin pipe.
        frames, path, fps = pickle.load(sys.stdin.buffer)
        encode_alert_video(frames, path, fps)
