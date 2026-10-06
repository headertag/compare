from unittest.mock import Mock
from alert_dispatch import AlertDispatcher
from alert_media import HistoryFrame


def frame(value=b'abc'):
    return (HistoryFrame(value, 1),)


def test_repeats_cannot_starve_unserved_panes_and_busy_sender_does_not_lose_events():
    d=AlertDispatcher(120); sender=Mock();sender.submit.return_value=True
    a,b=(0,0,.5,1),(.5,0,1,1)
    d.offer(a,5,frame()); assert d.dispatch(0,sender).focus == a
    d.offer(a,6,frame()); d.offer(b,9,frame())
    assert d.dispatch(120,sender) is None
    sender.submit.return_value=False
    assert d.dispatch(121,sender) is None and len(d.pending)==2
    sender.submit.return_value=True
    assert d.dispatch(122,sender).focus==b
    assert d.dispatch(243,sender).focus==a


def test_simultaneous_events_retain_offered_priority_and_source_changes_clear_pending():
    d=AlertDispatcher(0); sender=Mock();sender.submit.return_value=True
    d.set_source('one');d.offer((1,),9,frame());d.offer((2,),5,frame())
    assert d.dispatch(1,sender).focus==(1,)
    d.set_source('two');assert not d.pending


def test_memory_is_bounded_and_latest_qualified_frame_is_preserved():
    d=AlertDispatcher(120,max_bytes=6)
    old,new=HistoryFrame(b'aaaa',1),HistoryFrame(b'bbbb',2)
    d.offer((1,),5,(old,new))
    assert d.size_bytes<=6 and d.pending[(1,)].frames==(new,)


def test_multiplier_extends_interval_without_blocking_or_dividing_by_zero():
    d=AlertDispatcher(10,2);sender=Mock();sender.submit.return_value=True
    d.offer((1,),9,frame());d.dispatch(0,sender)
    d.offer((2,),9,frame());assert d.dispatch(15,sender) is None
    assert d.dispatch(20,sender).focus==(2,)
    immediate=AlertDispatcher(0);immediate.offer((1,),9,frame())
    assert immediate.dispatch(0,sender) is not None


def test_submitted_track_cannot_requeue_a_delayed_duplicate():
    d=AlertDispatcher(120);sender=Mock();sender.submit.return_value=True
    focus=(0,0,1,1);identity=(0,'yolo',1)
    d.offer(focus,9,frame(),[identity]);assert d.dispatch(0,sender)
    for now in (1,2,60,119,121,240):
        d.offer(focus,10,frame(b'new pixels'),[identity])
        assert d.dispatch(now,sender) is None
    assert not d.pending and sender.submit.call_count==1


def test_new_track_same_pane_is_retained_while_old_track_does_not_overwrite_it():
    d=AlertDispatcher(120);sender=Mock();sender.submit.return_value=True
    focus=(0,0,1,1);old=(0,'yolo',1);new=(0,'yolo',2)
    d.offer(focus,9,frame(),[old]);d.dispatch(0,sender)
    d.offer(focus,10,frame(b'new person'),[new])
    d.offer(focus,11,frame(b'old person'),[old])
    assert d.dispatch(121,sender).frames[0].jpeg==b'new person'
    d.offer(focus,12,frame(),[new]);assert not d.pending


def test_failed_submit_does_not_consume_track_and_dedup_memory_is_bounded():
    d=AlertDispatcher(120);sender=Mock();sender.submit.return_value=False
    identity=(0,'yolo',1)
    d.offer((0,),9,frame(),[identity]);assert not d.dispatch(0,sender)
    assert not d.submitted_ids
    sender.submit.return_value=True;assert d.dispatch(1,sender)
    d.retain_active_ids([identity]);assert identity in d.submitted_ids
    d.retain_active_ids([]);assert not d.submitted_ids
