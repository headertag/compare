import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import Mock, patch

import pytest
from alert_media import MediaConfig, HistoryFrame, encode_alert_video
from alerts import AlertMediaSender
from media_transport import BoundedTelegramBot, MediaDeliveryError, run_bounded, upload_worker


class FakeClock:
    def __init__(self): self.now=0.;self.waits=[]
    def monotonic(self): return self.now
    def sleep(self,delay): self.waits.append(delay);self.now+=delay


def retry_sender():
    sender=object.__new__(AlertMediaSender);sender.config=MediaConfig()
    return sender


def test_transient_retries_backoff_then_success_without_fallback():
    clock=FakeClock();sender=retry_sender()
    action=Mock(side_effect=[MediaDeliveryError('Telegram 503',retryable=True),TimeoutError(), 'ok'])
    with patch('alerts.time.monotonic',clock.monotonic),patch('alerts.time.sleep',clock.sleep):
        assert sender._retry_upload(action,60)=='ok'
    assert clock.waits==[1,2] and action.call_count==3
    assert all(c.args==(20,) for c in action.call_args_list)


def test_deadline_cuts_attempt_and_prevents_further_retries():
    clock=FakeClock();sender=retry_sender();limits=[]
    def stall(timeout):
        limits.append(timeout);clock.now+=timeout;raise TimeoutError()
    with patch('alerts.time.monotonic',clock.monotonic),patch('alerts.time.sleep',clock.sleep):
        with pytest.raises(TimeoutError):sender._retry_upload(stall,25)
    assert limits==[20,4] and clock.now==25


def test_permanent_failure_is_not_retried_and_retry_after_respects_deadline():
    sender=retry_sender();clock=FakeClock()
    with patch('alerts.time.monotonic',clock.monotonic),patch('alerts.time.sleep',clock.sleep):
        permanent=Mock(side_effect=MediaDeliveryError('Telegram 403'))
        with pytest.raises(MediaDeliveryError):sender._retry_upload(permanent,60)
        assert permanent.call_count==1 and not clock.waits
        limited=Mock(side_effect=MediaDeliveryError('Telegram 429',retryable=True,retry_after=90))
        with pytest.raises(TimeoutError):sender._retry_upload(limited,60)
        assert limited.call_count==1 and not clock.waits


def test_bounded_process_is_killed_on_deadline(tmp_path):
    pidfile=tmp_path/'pid'
    script=f'import os,time;open({str(pidfile)!r},"w").write(str(os.getpid()));time.sleep(30)'
    started=time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):run_bounded([sys.executable,'-c',script],b'',.5)
    assert time.monotonic()-started<3
    if os.name=='posix':
        with pytest.raises(ProcessLookupError):os.kill(int(pidfile.read_text()),0)


def test_transport_passes_credentials_only_on_stdin_and_bounds_attempt():
    bot=BoundedTelegramBot('private-token');photo=io.BytesIO(b'jpeg')
    with patch('media_transport.run_bounded',return_value=b'{"ok":true}') as worker:
        bot.sendPhoto(123,photo,timeout_seconds=7)
    command,payload,timeout=worker.call_args.args
    assert 'private-token' not in str(command) and timeout==7
    assert json.loads(payload)['token']=='private-token'


def test_worker_redacts_errors_and_disables_hidden_upload_retries(monkeypatch,capsys,tmp_path):
    import telepot
    file=tmp_path/'video';file.write_bytes(b'video')
    token='secret-token'
    req=dict(token=token,chat=123,path=str(file),method='sendVideo')
    monkeypatch.setattr(sys,'stdin',io.StringIO(json.dumps(req)))
    bot=Mock();bot.sendVideo.side_effect=telepot.exception.TelegramError(
        'failure '+token+' 123 https://api.invalid/secret',503,{'parameters':{}})
    cls,opts=telepot.api._onetime_pool_spec
    with patch('telepot.Bot',return_value=bot),patch.dict(telepot.api._pools),patch.object(telepot.api,'_onetime_pool_spec',(cls,opts.copy())):
        upload_worker()
        assert telepot.api._onetime_pool_spec[1]['retries'] is False
    raw=capsys.readouterr().out;result=json.loads(raw)
    assert result['retryable'] and result['code']==503
    assert token not in raw and '123' not in raw and 'https://' not in raw


def test_encoding_deadline_path_produces_real_multiframe_video(tmp_path):
    import cv2
    import numpy as np
    images=[]
    for n in [10,200]:
        ok,jpeg=cv2.imencode('.jpg',np.full((64,96,3),n,np.uint8));assert ok
        images.append(HistoryFrame(jpeg.tobytes(),n))
    file=tmp_path/'clip.mp4'
    encode_alert_video(tuple(images),file,1,timeout_seconds=10)
    video=cv2.VideoCapture(str(file))
    assert video.read()[0] and video.read()[0] and not video.read()[0]
    video.release()


def test_alert_total_budget_shared_between_recipients_including_fallback():
    clock=FakeClock();sender=retry_sender();sender.chat_ids=[1,2];sender.bot=Mock();sender.health=Mock()
    def stall(*args,timeout_seconds,**kw):
        clock.now+=timeout_seconds;raise TimeoutError()
    sender.bot.sendVideo.side_effect=stall;sender.bot.sendPhoto.side_effect=stall
    def encode(frames,path,fps,**kw):clock.now+=2;path.write_bytes(b'video')
    with patch('alerts.time.monotonic',clock.monotonic),patch('alerts.time.sleep',clock.sleep),patch('alert_media.encode_alert_video',side_effect=encode):
        sender.send_snapshot((HistoryFrame(b'jpeg',0),))
    assert clock.now<=60
    assert {c.args[0] for c in sender.bot.sendVideo.call_args_list}=={1,2}
    assert sender.health.update.call_args.kwargs['last_delivery_recipients']==0
