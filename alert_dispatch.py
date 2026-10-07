"""Select a fresh frame event; never replay suppressed alerts after cooldown."""
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

    def set_source(self, source):
        if self.source != source:
            self.pending.clear()
            self.source = source

    def offer(self, focus, score, frames, event_ids=None):
        if not frames:
            return
        ids = frozenset(event_ids or ())
        key = tuple(focus)
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
        # Offers belong exclusively to this processed frame. Discard all of them
        # even during cooldown, when busy, and after choosing one winning pane.
        candidates = tuple(self.pending.values())
        self.pending.clear()
        if not candidates:
            return None
        if self.last_sent is not None:
            elapsed = now-self.last_sent
            if elapsed < self.interval*self.multiplier:
                return None
        event = max(candidates, key=lambda event: event.score)
        if not sender.submit(event.frames):
            return None
        self.last_sent = now
        return event
