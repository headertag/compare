"""Bounded, fair pending pane events behind the shared notification interval."""
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class PendingAlert:
    focus: tuple
    score: float
    frames: tuple
    event_ids: frozenset = frozenset()


class AlertDispatcher:
    def __init__(self, interval, multiplier=1, max_bytes=64*1024*1024):
        self.interval = max(0, interval)
        self.multiplier = max(1, multiplier)
        self.max_bytes = max_bytes
        self.pending = OrderedDict()
        self.last_sent = None
        self.source = None
        self.served = {}
        self.submitted_ids = set()

    def set_source(self, source):
        if self.source != source:
            self.pending.clear()
            self.served.clear()
            self.submitted_ids.clear()
            self.source = source

    def retain_active_ids(self, active_ids):
        # Keep deduplication bounded while preserving identities in pending clips.
        pending_ids = {i for event in self.pending.values() for i in event.event_ids}
        self.submitted_ids.intersection_update(set(active_ids) | pending_ids)

    def offer(self, focus, score, frames, event_ids=None):
        if not frames:
            return
        ids = frozenset(event_ids or ())
        if event_ids is not None and not ids.difference(self.submitted_ids):
            return
        key = tuple(focus)
        if key in self.pending:
            ids = ids | self.pending[key].event_ids
        # Replacement preserves arrival order: persistent panes cannot jump ahead.
        self.pending[key] = PendingAlert(key, score, tuple(frames), ids)
        self._bound_memory()

    @property
    def size_bytes(self):
        # Snapshots from multiple panes often share the same immutable JPEG bytes.
        unique = {id(f.jpeg): f.jpeg for event in self.pending.values() for f in event.frames}
        return sum(map(len, unique.values()))

    def _bound_memory(self):
        while self.size_bytes > self.max_bytes and self.pending:
            # Preserve each pane's latest qualified frame before dropping an event.
            key = max(self.pending, key=lambda k: len(self.pending[k].frames))
            event = self.pending[key]
            if len(event.frames) > 1:
                self.pending[key] = PendingAlert(key, event.score, event.frames[1:], event.event_ids)
            else:
                self.pending.pop(key)
                print('Pending alert dropped: configured media memory budget too small.', flush=True)

    def dispatch(self, now, sender):
        if not self.pending:
            return None
        if self.last_sent is not None:
            elapsed = now-self.last_sent
            if elapsed <= self.interval or elapsed < self.interval*self.multiplier:
                return None
        # A repeat from an already served pane cannot starve an unseen pane.
        key = min(self.pending, key=lambda k: self.served.get(k, float("-inf")))
        event = self.pending[key]
        if not sender.submit(event.frames):
            return None  # Retry without losing the event or consuming the interval.
        self.pending.pop(key)
        self.last_sent = now
        self.served[key] = now
        self.submitted_ids.update(event.event_ids)
        return event
