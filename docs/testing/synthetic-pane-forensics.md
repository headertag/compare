# Synthetic alert regression tests

Run `PYTHONPATH=. python -m pytest tests/test_synthetic_pane_forensics.py tests/test_alert_dispatch.py tests/test_alert_media.py -q`.

Fixtures use 12 procedural 4×4 camera-wall frames with office, park and home
scenes, scripted person and wall observations, and seeded random pane placement.
They exercise real tracking, history and main-loop decisions without accessing
live cameras or Telegram. They do not measure YOLO classification accuracy.

## v1.0.3 hotfix policy

Send the strongest fresh qualified event when ready. Discard events during the
global cooldown and require fresh qualifying input after it ends. Do not replay
other panes, a previously reported identity, or an event rejected by a busy media
worker. A still-active identity may alert again from a fresh frame after cooldown.

The extended 410-frame synthetic test verifies that a person present only during
cooldown does not cause a later alert. Unit tests verify the exact interval
boundary, same-track fresh events, simultaneous pane selection, busy-worker drops,
and bounded history. Existing controls verify flicker alone does not qualify and
unqualified wall frames are excluded from tracking clips.

Earlier experiments intentionally retained events across cooldown. That policy
was rejected by the user and is superseded by this hotfix. Qualified clip history
is still retrospective, but it never generates an alert without fresh evidence.

## October 11 clip-history regression

The qualified-only history change in e5f2056 discarded the earlier observations
needed to establish motion. A new alert at first qualification therefore commonly
encoded one frame. The offline first-person test reproduced this (one frame versus
three expected) before the fix. History now records observed track identities and
selects prior observations of only the identities qualified on the trigger frame.
An encoding test decodes the selected multi-frame MP4. Tests exclude unrelated
tracks in the same pane, other panes, missed observations and old source history.
Alert timing, cooldown discard policy, model cutoffs and motion rules are unchanged.
