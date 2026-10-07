from unittest.mock import Mock
from alert_dispatch import AlertDispatcher
from alert_media import HistoryFrame


def frame(value=b'abc'):
    return (HistoryFrame(value, 1),)


def test_cooldown_discards_events_and_no_replay_when_it_expires():
    d=AlertDispatcher(120);sender=Mock();sender.submit.return_value=True
    d.offer((1,),9,frame());assert d.dispatch(0,sender)
    for t in (1,60,119.9):
        d.offer((2,),12,frame());assert d.dispatch(t,sender) is None
        assert not d.pending
    assert d.dispatch(120,sender) is None
    assert d.dispatch(240,sender) is None
    assert sender.submit.call_count==1


def test_fresh_detection_at_cooldown_end_can_alert_even_with_same_track_id():
    d=AlertDispatcher(120);sender=Mock();sender.submit.return_value=True
    identity=(0,'yolo',1)
    d.offer((1,),9,frame(),[identity]);assert d.dispatch(0,sender)
    d.offer((1,),10,frame(b'fresh'),[identity])
    assert d.dispatch(120,sender).frames[0].jpeg==b'fresh'


def test_busy_sender_drops_event_and_requires_a_fresh_offer():
    d=AlertDispatcher(120);sender=Mock();sender.submit.return_value=False
    d.offer((1,),9,frame());assert d.dispatch(0,sender) is None
    sender.submit.return_value=True
    assert d.dispatch(1,sender) is None
    d.offer((2,),9,frame(b'new'));assert d.dispatch(2,sender)


def test_best_current_pane_wins_and_other_panes_are_not_buffered():
    d=AlertDispatcher(0);sender=Mock();sender.submit.return_value=True
    d.offer((1,),5,frame());d.offer((2,),9,frame())
    assert d.dispatch(0,sender).focus==(2,)
    assert d.dispatch(1,sender) is None


def test_memory_bound_keeps_latest_frame_and_source_reset_drops_candidates():
    d=AlertDispatcher(120,max_bytes=6)
    old,new=HistoryFrame(b'aaaa',1),HistoryFrame(b'bbbb',2)
    d.offer((1,),5,(old,new));assert d.size_bytes<=6
    assert d.pending[(1,)].frames==(new,)
    d.set_source('new');assert not d.pending


def test_multiplier_gate_discards_until_fresh_eligible_frame():
    d=AlertDispatcher(10,2);sender=Mock();sender.submit.return_value=True
    d.offer((1,),9,frame());d.dispatch(0,sender)
    d.offer((2,),9,frame());assert d.dispatch(15,sender) is None
    assert d.dispatch(20,sender) is None
    d.offer((2,),9,frame());assert d.dispatch(20,sender)
