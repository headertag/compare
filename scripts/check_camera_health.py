#!/usr/bin/env python3
"""Run independently of inference (systemd timer); reuse existing Telegram config."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from health import health_issues, notification_plan, validate_health_settings, apply_service_state


def send_message(token, chat, text):
    data = json.dumps({'chat_id': chat, 'text': text[:4000]}).encode()
    request = urllib.request.Request('https://api.telegram.org/bot'+token+'/sendMessage',
                                     data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=20) as response:
        result = json.load(response)
    if not result.get('ok'):
        raise RuntimeError('Telegram rejected health notification')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', action='store_true', help='Send an explicit installation test')
    parser.add_argument('--dry-run', action='store_true', help='Show current checks without sending')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path('config.yaml').read_text())
    settings = validate_health_settings(cfg.get('health', {}))
    if not settings.get('enabled', True):
        return
    runtime = Path('.runtime'); runtime.mkdir(exist_ok=True)
    with (runtime/'health-check.lock').open('w') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if args.test or args.dry_run:
                raise SystemExit("Health check already running; retry shortly.")
            return
        try: state = json.loads((runtime/'health.json').read_text())
        except (OSError, ValueError): state = None
        now = time.time()
        issues = health_issues(state, settings, now)
        try:
            result = subprocess.run(['systemctl', 'show', 'camera-alert.service',
                                     '-p', 'ActiveState', '-p', 'MainPID',
                                     '-p', 'ActiveEnterTimestampMonotonic'],
                                    capture_output=True, text=True, timeout=5)
            if result.returncode:
                issues['service_check'] = 'Unable to query the camera service state.'
            else:
                status = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
                issues = apply_service_state(issues, state, status, time.monotonic(),
                                             settings.get('startup_grace_seconds', 180))
        except (OSError, subprocess.TimeoutExpired):
            issues['service_check'] = 'Unable to query the camera service state.'
        if args.dry_run:
            print(json.dumps(issues, indent=2)); return
        path = runtime/'health-notifications.json'
        try: previous = json.loads(path.read_text())
        except (OSError, ValueError): previous = {}
        # Independent state per recipient: a failed chat remains eligible for retry.
        for index, chat in enumerate(cfg['telegram']['chat_ids']):
            key = str(chat)
            text = ('Camera monitor: health notifications are now enabled. This is an installation test.'
                    if args.test else notification_plan(issues, previous.get(key, {}), now,
                                                        settings.get('repeat_seconds', 3600)))
            if not text: continue
            try:
                send_message(cfg['telegram']['token'], chat, text)
            except Exception as exc:
                print(f'Health notification delivery failed for recipient {index+1}: {type(exc).__name__}', flush=True)
                continue
            print(f'Health notification delivered to recipient {index+1}.', flush=True)
            if not args.test:
                previous[key] = {'issues': issues, 'sent_at': now}
                temp = path.with_suffix('.tmp')
                temp.write_text(json.dumps(previous)); os.chmod(temp, 0o600); os.replace(temp, path)


if __name__ == '__main__':
    main()
