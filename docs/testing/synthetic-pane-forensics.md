# Synthetic cross-pane alert investigation

These are offline regression tests covering the reproduced failure and its repair.
No live camera input, production access, network calls, or Telegram sends are used.

## Reproduce

From the repository with its dependencies installed:

```sh
PYTHONPATH=. python -m pytest tests/test_synthetic_pane_forensics.py -q
PYTHONPATH=. python scripts/synthetic_pane_forensics.py --output /tmp/compare-synthetic
```

The generator makes twelve 1280×720 frames per scenario, partitioned into a 4×4
camera wall. Seeded randomness selects different office, park and home scenes,
the person/wall panes, and the person's appearance. Scenes remain spatially
consistent for meaningful tracking. Three seeds test pane independence.

## What is real versus controlled

Images are procedural illustrations, not photographs. Detector boxes and
confidence scores are explicitly scripted: .72 for the false-positive wall and
.94 for the person. This tests what the application does *after* a detector emits
those observations. It does not measure YOLO accuracy or prove a particular
production frame produced a wall detection.

The real ModelPipeline, ByteTracker, visual-motion calculation, AlertHistory and
main loop execute. Camera, model weights, Telegram sender, server and health
writes are replaced locally. A recording sender captures actual submissions from
the main loop. Camera opening and network connection helpers are guarded against
use. Main-loop time advances at 3.2 FPS; the normal alert interval is 120 seconds.
The short-interval control uses 0.6 seconds, explicitly only in that control.

## Findings before the repair

1. **Global alert suppression reproduced.** Wall drift plus local brightness
   changes qualifies the .72 false positive at frame 3. The person qualifies at
   frame 5 with score 14.1 versus the wall's 10.8, and wins pane selection. However,
   `main.py` uses one global `last_alert`, so no person notification is submitted.
   Removing the wall observation lets exactly the same person event send. The
   higher-confidence detection was not discarded: its notification was blocked.
2. **Motion qualification can accept non-person changes.** Static pixels with
   drifting boxes fail. Changing illumination with a fixed box fails. Combined
   drift and nonuniform local illumination changes pass. This supports a false
   positive mechanism, without claiming all static walls pass or YOLO will emit
   these scripted boxes. The visual gate measures changed pixels, not whether
   they belong to a moving person.
3. **History is candidate-based.** An unqualified, stationary wall candidate
   remains in all twelve history frames. In `late_wall_only`, box drift starts
   late and the final frame qualifies; the actual sender receives all twelve
   frames, none containing a ground-truth person. Filtering to the correct pane
   alone does not require motion qualification or a matching qualified identity
   on every retained frame.
4. **Higher score selection works when eligible.** The simultaneous-person case
   selects the person, and the short-interval control can send it after the wall.

The regression tests now require the other pane's event to remain pending, then
exercise 410 continuously processed synthetic frames to verify it actually sends
after 120 seconds even though the person has left. A fixed box appearing on
alternate frames is also tested: it never qualifies or enters clip history.
The late-wall case now retains only its one qualified frame, not all twelve.
The fixed implementation retains qualified pane events behind a global interval,
prioritizes unserved panes, bounds pending JPEG memory, and uses only qualified
boxes as tracking clip evidence. These tests do not claim to eliminate every
model false positive; drift plus nonuniform lighting remains an adversarial case.

## Remaining limitations

History now requires qualification in the selected pane, but is not filtered to
a single track identity. Multiple genuine tracks in one pane can share a clip.
One pending clip per pane coalesces activity; it is not a durable event archive.
Stronger visual-motion discrimination requires additional adversarial fixtures
(shadows, IR changes, camera movement), not simply lowering thresholds.

These experiments establish software mechanisms under controlled inputs. They
cannot retroactively identify the exact observations in an unrecorded incident.

## Delayed duplicate regression

The dispatcher records successfully queued track identities. More qualified
frames for those identities cannot re-create a pending event. New tracks remain
eligible, and sender queue rejection does not mark tracks submitted. Identity
memory is pruned with active/lost tracker state and cleared on source changes.
The offline main-loop regression asserts that the already-sent wall track does
not remain pending while the as-yet-unreported person does.
