"""Offline regression tests for cross-pane alert retention and qualified history."""
import numpy as np
import pytest
from scripts.synthetic_pane_forensics import fixtures, reproduce, SEED


def focus(pane):
    p = pane-1
    return [p%4/4, p//4/4, (p%4+1)/4, (p//4+1)/4]


def test_fixture_is_seeded_twelve_frame_widescreen_grid():
    a, observations, meta = fixtures()
    b, observations2, meta2 = fixtures()
    assert len(a) == 12 and all(f.shape == (720, 1280, 3) for f in a)
    assert all(np.array_equal(x,y) for x,y in zip(a,b))
    assert observations == observations2 and meta == meta2
    assert set(meta['scene_types']) == {'office','park','home'}
    assert meta['wall_pane'] != meta['person_pane']
    assert all(person['pane'] != meta['wall_pane'] for people in meta['ground_truth'] for person in people)


@pytest.mark.parametrize('seed', [SEED, 17, 90210])
def test_global_interval_retains_other_panes_real_person_event(seed):
    bad = reproduce('wall_then_person',seed)
    control = reproduce('person_only',seed)
    person = str(bad['person_pane'])
    # The person is detected and qualifies identically, despite the false positive.
    assert [f['pane_scores'].get(person,0) for f in bad['frames']] == [
        f['pane_scores'].get(person,0) for f in control['frames']]
    qualified = [f for f in bad['frames'] if f['pane_scores'].get(person,0)>0]
    assert qualified
    assert all(f['focus'] == focus(bad['person_pane']) for f in qualified)
    assert all(f['pane_scores'][person] > f['pane_scores'][str(bad['wall_pane'])] for f in qualified)
    assert len(bad['submissions']) == 1
    assert bad['submissions'][0]['focus'] == focus(bad['wall_pane'])
    assert control['submissions'][0]['focus'] == focus(control['person_pane'])
    assert focus(bad['person_pane']) in bad['pending_focus']
    # All qualified person events land in the global 120s gate after the wall alert.
    wall_time = bad['submissions'][0]['frame']*bad['step_seconds']
    assert all(0 < f['seconds']-wall_time <= 120 for f in qualified)
    assert bad['detector_calls'] == control['detector_calls'] == 12


@pytest.mark.parametrize('case', ['static_wall','fixed_box_lighting','flickering_wall'])
def test_false_positive_needs_both_box_drift_and_pixel_change(case):
    result = reproduce(case)
    assert not any(f['pane_scores'].get(str(result['wall_pane']),0) for f in result['frames'])
    assert result['submissions'][0]['focus'] == focus(result['person_pane'])


def test_higher_scoring_person_wins_when_interval_is_open():
    result = reproduce('simultaneous')
    assert result['submissions'][0]['focus'] == focus(result['person_pane'])
    both = [f for f in result['frames'] if len(f['pane_scores']) == 2]
    assert both and all(f['focus'] == focus(result['person_pane']) for f in both)


def test_same_observations_send_person_with_short_interval_control():
    result = reproduce('short_interval_control')
    assert result['submissions'][0]['focus'] == focus(result['wall_pane'])
    assert any(s['focus'] == focus(result['person_pane']) for s in result['submissions'][1:])


def test_unqualified_false_positive_is_excluded_from_history():
    result = reproduce('static_wall')
    assert result['wall_history_frames'] == 0
    assert not any(f['pane_scores'].get(str(result['wall_pane']),0) for f in result['frames'])
    # With nonuniform lighting plus drift, that same wall can actually be sent.
    sent = reproduce('late_wall_only')
    assert any(s['focus'] == focus(sent['wall_pane']) and s['clip_frames']==1
               for s in sent['submissions'])


def test_main_loop_delivers_saved_person_after_interval_without_new_detection():
    result = reproduce('wall_then_person', drain_pending=True)
    sent = result['submissions']
    assert len(sent) == 2
    assert sent[0]['focus'] == focus(result['wall_pane'])
    assert sent[1]['focus'] == focus(result['person_pane'])
    assert sent[1]['frame'] > 12  # Person has left; the saved event still delivers.
    assert sent[1]['clip_frames'] > 0
    assert (sent[1]['frame']-sent[0]['frame'])*result['step_seconds'] > 120
