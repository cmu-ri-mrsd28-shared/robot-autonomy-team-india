import numpy as np
from catchbox.core import Config
from catchbox.experiments import Retention,setup,throws_for,rollout

def test_retention_resets_after_loss():
    r=Retention();r.update(0,True);r.update(.8,True);r.update(.9,False)
    r.update(1,True);r.update(1.9,True);assert not r.success
    r.update(2,True);assert r.success

def test_paddle_contact_and_replay():
    sim=setup('paddle',Config(duration=3))
    throw=throws_for(sim,42,1,'drop')[0]
    a,_=rollout(sim,throw,'stationary');b,_=rollout(sim,throw,'stationary')
    a.pop('wall_s');b.pop('wall_s');assert a==b
    assert a['contact'] and a['first_bounce_height_m'] is not None
    assert sim.model.geom('paddle').size[0]==.075

def test_stock_gripper_and_delayed_dynamic_jaws():
    sim=setup('gripper',Config(duration=3))
    assert sim.box_id is None
    throw=throws_for(sim,42,1,'lob',True)[0]
    _,early=rollout(sim,throw,'parked',opening=.08,closure_offset=-.18,delay=0,record=True)
    _,late=rollout(sim,throw,'parked',opening=.08,closure_offset=-.18,delay=.2,record=True)
    assert early[20]['jaw_aperture']<late[20]['jaw_aperture']
    assert np.max(np.abs(np.array(early[-1]['qpos'][:7])-sim.home))<.01
