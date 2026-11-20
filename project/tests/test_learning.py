import numpy as np
from catchbox.core import BallState,Config
from catchbox.learning.trajectory import Predictor,varied_throw,freeze_bank,in_workspace,clamp_workspace
import pytest
from catchbox.learning.environment import CatchCore,impact_follow_offset

def test_predictor_uses_acceleration_and_rejects_contact():
    s=BallState(0,np.array([0.,0.,1.]),np.array([1.,0.,1.]),acceleration=np.array([.2,0,-8.]))
    p=Predictor().predict(s,.4)
    assert p.valid and p.velocity[2]<0
    np.testing.assert_allclose(p.position,s.position+s.velocity*p.time_to_contact+.5*s.acceleration*p.time_to_contact**2)
    assert not Predictor().predict(s,.4,True).valid

def test_launch_diversity_and_required_movement():
    rng=np.random.default_rng(12);rows=[varied_throw(rng,[.6,0,.4]) for _ in range(500)]
    angles=[m['angle_deg'] for _,m in rows];distance=[m['distance_m'] for _,m in rows]
    assert min(angles)<-70 and max(angles)>70 and min(distance)<.6 and max(distance)>1.4
    for throw,meta in rows:
        s=BallState(0,np.array(throw.position),np.array(throw.velocity),acceleration=np.array([0,0,-9.81]))
        pred=Predictor().predict(s,.4)
        np.testing.assert_allclose(pred.position,meta['target'],atol=1e-10)
        assert meta['required_movement_m']>=.065

def test_new_box_and_deterministic_zero_residual_rollout():
    assert Config().box_width==.05 and Config().box_depth==.05
    env=CatchCore('box',seed=0);throw=env.throw;meta=env.meta
    results=[]
    for _ in range(2):
        env.reset(throw,meta)
        for k in range(200):
            obs,rew,done,trunc,info=env.step(np.zeros(7))
            assert np.isfinite(obs).all()
            if done or trunc:break
        results.append(info['result'])
    assert results[0]==results[1]
    assert results[0]['arm_movement_m']>.04
    np.testing.assert_allclose(env.sim.read().acceleration,env.sim.data.qacc[env.sim.ball_v:env.sim.ball_v+3])

def test_workspace_bounds_and_outside_landing_rejected():
    center=np.array([.6,0.,.4])
    assert in_workspace(center+np.array([.5,0,0]),center)
    assert not in_workspace(center+np.array([.51,0,0]),center)
    np.testing.assert_allclose(clamp_workspace(center+np.array([.6,.8,0]),center),center+np.array([.3,.4,0]))
    env=CatchCore('box',seed=0)
    throw,meta=varied_throw(np.random.default_rng(2),env.start_pos+np.array([1.,0,0]))
    with pytest.raises(ValueError,match='outside the 50 cm workspace'):
        env.reset(throw,meta)

def test_frozen_throws_land_in_workspace():
    env=CatchCore('box',seed=0)
    bank=freeze_bank(env.sim,3,12)
    assert all(in_workspace(item['meta']['target'],bank['workspace_center']) for item in bank['throws'])

def test_impact_follow_is_bounded_and_matches_velocity_at_impact():
    v=np.array([.3,-.2,-.4]);dt=1e-5
    assert np.all(impact_follow_offset(-.121,v)==0)
    assert np.all(impact_follow_offset(.121,v)==0)
    np.testing.assert_allclose((impact_follow_offset(dt,v)-impact_follow_offset(-dt,v))/(2*dt),v,rtol=.001)
    assert np.max(np.abs(impact_follow_offset(-.06,np.array([10.,-10.,10.]))))<=.04

def test_impact_matching_rollout_reaches_reward_window():
    env=CatchCore('box',seed=0);env.velocity_match=True
    seen=False
    for _ in range(100):
        _,_,done,trunc,info=env.step(np.zeros(7))
        seen |= 'preimpact_velocity' in info['reward_terms']
        if done or trunc:break
    assert seen

def test_printed_box_has_real_mesh_mass_and_held_handle():
    env=CatchCore('box',seed=0,printed_box=True)
    sim=env.sim
    assert sim.model.body_mass[sim.box_id] == pytest.approx(.25)
    assert sim.cfg.box_width == pytest.approx(.12)
    assert sim.cfg.box_height == pytest.approx(.09)
    assert sim.model.geom('printed_box_visual').id >= 0
    np.testing.assert_allclose(sim.nominal_opening-sim.nominal_grip,[.105,0,.05],atol=1e-5)
    np.testing.assert_allclose(sim.data.qpos[7:9],[.025,-.025],atol=1e-5)
