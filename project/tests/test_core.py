import numpy as np
from catchbox.core import BallState, Config, BoxMetrics, crossing, sample_throws


def test_predict_descending_crossing():
    t,p = crossing(BallState(0,np.array([0.,0.,1.]),np.array([.5,0.,1.])),.3)
    assert t > 1/9.81
    np.testing.assert_allclose(p,[.5*t,0,.3])
    assert crossing(BallState(0,np.zeros(3),np.zeros(3)),1) is None

def test_seed_and_ballistic_targets():
    a = sample_throws(42,10,'mixed',[.6,0,.4])
    assert a == sample_throws(42,10,'mixed',[.6,0,.4])
    assert a != sample_throws(43,10,'mixed',[.6,0,.4])
    for throw in a:
        t,p = crossing(BallState(0,np.array(throw.position),np.array(throw.velocity)),.4)
        assert abs(p[0]-.6) <= .09 and abs(p[1]) <= .07

def test_retention_requires_continuous_full_second():
    m=BoxMetrics(Config())
    m.update(0,np.array([0,0,.1]))
    m.update(.1,np.array([0,0,-.03]))
    m.update(.7,np.array([0,0,-.08]))
    assert m.entered and not m.retained
    m.update(.8,np.array([0,0,-.01]))  # ball not fully inside: restart timer
    m.update(.9,np.array([0,0,-.08]))
    m.update(1.89,np.array([0,0,-.08]))
    assert not m.retained
    m.update(1.9,np.array([0,0,-.08]))
    assert m.result()['outcome']=='catch'

def test_rim_and_bounceout_are_independent():
    m=BoxMetrics(Config())
    m.update(0,np.array([0,0,.1]))
    m.update(.1,np.array([0,0,-.01]),rim=True,contact=True)
    m.update(.2,np.array([0,0,.1]))
    assert m.result()['outcome']=='entered_bounced_out' and m.rim_hit
    n=BoxMetrics(Config())
    n.update(0,np.array([.105,0,.1]))
    n.update(.1,np.array([.105,0,-.01]),rim=True)
    assert n.result()['outcome']=='rim_hit_no_entry'

def test_retention_tolerates_small_contact_jitter_and_bounce_return():
    m=BoxMetrics(Config(box_width=.1,box_depth=.1,box_height=.05))
    m.update(0.,np.array([0.,0.,.1]))
    m.update(.1,np.array([0.,0.,-.01]))
    m.update(.2,np.array([0.,0.,.03]))
    m.update(.3,np.array([.0305,0.,-.03]))
    assert m.result()['outcome']=='entered_not_retained'
    m.update(1.3,np.array([.0305,0.,-.03]))
    assert m.retained and m.escaped
    n=BoxMetrics(Config(box_width=.1,box_depth=.1,box_height=.05))
    n.update(0.,np.array([0.,0.,.1]))
    n.update(.1,np.array([0.,0.,-.01]))
    n.update(.2,np.array([0.,0.,-.03]))
    n.update(.8,np.array([.032,0.,-.03]))
    n.update(1.3,np.array([0.,0.,-.03]))
    assert not n.retained


def test_lobs_have_upward_arc_and_lateral_travel():
    for throw in sample_throws(271828,100,'lob',[.6,0,.4]):
        v=np.array(throw.velocity)
        t,p=crossing(BallState(0,np.array(throw.position),v),.4)
        assert .8 <= np.linalg.norm(v[:2]) <= 1.2
        assert 1.6 <= v[2] <= 2.2
        assert v[0] < 0 and t > v[2]/9.81
        assert np.linalg.norm(p[:2]-np.array(throw.position[:2])) > .2
