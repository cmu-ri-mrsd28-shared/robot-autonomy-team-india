import numpy as np
from catchbox.core import Config, Throw
from catchbox.sim import Simulation

def test_vertical_catch_replays_exactly_and_robot_is_dynamic():
    sim=Simulation(Config())
    throw=Throw((sim.nominal_opening+[.03,0,.4]).tolist(),[0,0,0],'drop')
    a,ta=sim.rollout(throw,trace=True)
    b,tb=sim.rollout(throw,trace=True)
    assert a==b and ta==tb
    assert a['retained'] and a['entered']
    assert a['ik_error_m']<.001
    assert np.linalg.norm(np.array(ta[-1]['q'])-np.array(ta[0]['q']))>.01
    assert sim.model.nv==15  # seven arm + two finger + six free-ball DOFs
