"""State-based residual catching environment. Physics and metrics run at 1 kHz."""
from dataclasses import asdict
import numpy as np
import mujoco
from catchbox.core import Config,Throw,BoxMetrics
from catchbox.sim import Simulation
from catchbox.experiments import Retention,setup,rotation_for_normal
from .trajectory import Predictor,varied_throw,in_workspace,clamp_workspace,launch_prediction

def impact_follow_offset(time_from_impact, velocity):
    """Bounded anticipation and gentle follow-through around predicted impact."""
    t=float(time_from_impact); window=.12
    if t < -window or t > window:return np.zeros(3)
    if t < 0:
        u=(t+window)/window
        weight=u*u*(3-2*u)
        offset=np.asarray(velocity)*t*weight
    else:
        offset=np.asarray(velocity)*t*(1-t/window)**2
    return np.clip(offset,-.04,.04)

class CatchCore:
    dt=.02
    action_dim=7
    def __init__(self,tool='box',seed=0,training=False,total_steps=150000,record=False,printed_box=False):
        self.tool=tool;self.cfg=Config(duration=3.5,control_dt=self.dt,box_width=.12 if printed_box else .10 if tool=='box' else .05,box_depth=.12 if printed_box else .10 if tool=='box' else .05,box_height=.09 if printed_box else .05 if tool=='box' else .12,printed_box=printed_box)
        self.sim=Simulation(self.cfg) if tool=='box' else setup(tool,self.cfg)
        self.predictor=Predictor();self.rng=np.random.default_rng(seed)
        self.training=training;self.total_steps=total_steps;self.steps_seen=0;self.record=record
        self.soft_landing=False
        self.velocity_match=False
        self.reset()
    def reset(self,throw=None,meta=None,seed=None):
        if seed is not None:self.rng=np.random.default_rng(seed)
        s=self.sim;s.reset();d=s.data
        self.start_pos=d.site_xpos[s.site].copy();self.start_q=d.qpos[:7].copy()
        self.workspace_center=d.site_xpos[s.grip_site].copy()
        self.level=min(1.,.2+.8*self.steps_seen/(.5*self.total_steps)) if self.training else 1.
        self.height=s.nominal_opening[2]+(self.cfg.radius if self.tool=='paddle' else 0)
        self.rejected=0
        if throw is None:
            for _ in range(100):
                origin=s.nominal_opening.copy();origin[2]=self.height
                candidate,metadata=varied_throw(self.rng,origin,self.level)
                pt=np.array(metadata['target']);pt[2]-=self.cfg.radius if self.tool=='paddle' else 0
                if not in_workspace(metadata['target'],self.workspace_center):
                    self.rejected+=1;continue
                release_prediction=launch_prediction(s,candidate,self.height)
                if not release_prediction.valid or not in_workspace(release_prediction.position,self.workspace_center):
                    self.rejected+=1;continue
                q,e=s.inverse_kinematics(pt)
                move=np.max(abs(q-self.start_q))*1.875/1.5
                if e<=.003 and move+.04<metadata['flight_s'] and metadata['speed_m_s']<=5:
                    throw,meta=candidate,metadata;break
                self.rejected+=1
            else:raise RuntimeError('No feasible training launch')
        self.throw=throw;self.meta=meta or {}
        d.qpos[s.ball_q:s.ball_q+3]=throw.position;d.qvel[s.ball_v:s.ball_v+3]=throw.velocity
        if self.tool=='gripper':d.qpos[7:9]=[.04,-.04]
        mujoco.mj_forward(s.model,d)
        pred=self.predictor.predict(s.read(),self.height)
        if not pred.valid:raise ValueError('Launch has no descending intercept')
        if not in_workspace(pred.position,self.workspace_center):raise ValueError('Predicted landing point is outside the 50 cm workspace')
        self.flight=pred.time_to_contact;self.target=pred.position.copy()
        self.target[2]-=self.cfg.radius if self.tool=='paddle' else 0
        self.goal_q,_=s.inverse_kinematics(self.target)
        self.move_time=max(.12,np.max(abs(self.goal_q-self.start_q))*1.875/1.5)
        self.metrics=BoxMetrics(self.cfg);self.retention=Retention();self.contact=False;self.rim=False
        self.entered=False;self.escaped=False;self.finger_strike=False;self.first_contact=None
        self.previous_local=None;self.first_bounce=0.;self.bounce_done=False;self.was_touch=False
        self.residual=np.zeros(7);self.last_qcmd=self.start_q.copy();self.last_vcmd=np.zeros(7)
        self.closed_aperture=.08;self.max_movement=0.;self.entry_speed=None;self.intercept_error=None
        self.return_sum=0.;self.trace=[];self.reward_terms={};self.done=False
        self.tool_geoms=s.box_geoms if self.tool=='box' else ({s.model.geom('paddle').id} if self.tool=='paddle' else set())
        self.fingers=[{i for i in range(s.model.ngeom) if 'finger1' in s.model.geom(i).name},
                      {i for i in range(s.model.ngeom) if 'finger2' in s.model.geom(i).name}]
        self.pred=pred
        return self.observation()
    def local_state(self):
        s=self.sim;d=s.data
        r=d.site_xmat[s.site].reshape(3,3);p=d.site_xpos[s.site]
        mujoco.mj_jacSite(s.model,d,s.jacp,s.jacr,s.site)
        velocity=s.jacp@d.qvel
        return r.T@(s.read().position-p),r.T@(s.read().velocity-velocity),velocity
    def observation(self):
        s=self.sim;d=s.data;ball=s.read();local,relative,opening_v=self.local_state()
        pred=self.predictor.predict(ball,self.height,self.contact)
        if pred.valid:self.pred=pred
        # Finite contact accelerations remain observable but are not flight-extrapolated.
        acceleration=np.clip(ball.acceleration,-60,60)
        obs=np.r_[d.qpos[:9]/np.pi,d.qvel[:9]/3,ball.position,ball.velocity/5,acceleration/30,
                  d.site_xpos[s.site],d.site_xmat[s.site],opening_v/3,
                  self.pred.position,self.pred.velocity/5,np.clip(self.pred.time_to_contact,0,2),
                  float(pred.valid),float(self.contact),self.residual,float(d.time)/self.cfg.duration,
                  min(self.retention.maximum,1)]
        return np.asarray(np.clip(obs,-10,10),dtype=np.float32)
    def step(self,action):
        if self.done:raise RuntimeError('Reset after episode termination')
        a=np.asarray(action,dtype=float)
        if a.shape!=(7,) or not np.isfinite(a).all():raise ValueError('Expected finite 7D action')
        a=np.clip(a,-1,1);old_action=self.residual.copy();self.residual=.8*self.residual+.2*a
        s=self.sim;d=s.data;m=s.model;c=self.cfg;t=float(d.time)
        pred=self.predictor.predict(s.read(),self.height,self.contact)
        if pred.valid:
            self.pred=pred;self.flight=t+pred.time_to_contact
            self.target=pred.position.copy();self.target[2]-=c.radius if self.tool=='paddle' else 0
        pos=self.target.copy();r=np.eye(3)
        # Correction bounds: cm-level intercept trim, tilt, stroke, matching horizontal motion.
        pos+=self.residual[:3]*np.array([.018,.018,.035])
        onset=self.flight-.07
        u=np.clip((t-onset)/.24,0,1);smooth=10*u**3-15*u**4+6*u**5
        if self.tool=='paddle':
            pos[2]-=(.06+.04*self.residual[5])*smooth
        elif self.tool=='box':pos[2]-=(.02+.02*self.residual[5])*smooth
        if self.contact and self.tool in ('box','paddle'):
            local,relative,opening_v=self.local_state();ball=s.read()
            if self.tool=='paddle':
                pos[:2]=ball.position[:2]+(.04+.03*self.residual[6])*ball.velocity[:2]+.015*self.residual[:2]
                tilt=np.clip(-3*local[:2]-.15*relative[:2]+.18*self.residual[3:5],-.3,.3)
                r=rotation_for_normal([*tilt,1])
            else:
                # Short post-entry horizontal following, smoothly expires after 200 ms.
                follow=max(0.,1-(t-self.first_contact)/.2)
                pos[:2]+=(.04+.04*self.residual[6])*self.pred.velocity[:2]*follow
                r=rotation_for_normal([*(.08*self.residual[3:5]),1])
        else:r=rotation_for_normal([*(.08*self.residual[3:5]),1])
        if self.tool=='gripper':r=s.data.site_xmat[s.site].reshape(3,3).copy()
        if self.tool=='box' and self.velocity_match:
            pos+=impact_follow_offset(t-self.flight,self.pred.velocity)
        pos[:2]=np.clip(pos[:2],s.nominal_opening[:2]-.23,s.nominal_opening[:2]+.23)
        pos[2]=np.clip(pos[2],s.nominal_opening[2]-.13,s.nominal_opening[2]+.08)
        pos=clamp_workspace(pos,self.workspace_center)
        q,_=s.inverse_kinematics(pos,r)
        progress=np.clip(t/self.move_time,0,1);blend=10*progress**3-15*progress**4+6*progress**5
        desired=self.start_q+(q-self.start_q)*blend
        speed_limit=6.0 if self.tool=='box' else 1.5
        qcmd=self.last_qcmd+np.clip(desired-self.last_qcmd,-speed_limit*self.dt,speed_limit*self.dt)
        vcmd=(qcmd-self.last_qcmd)/self.dt
        acmd=np.clip((vcmd-self.last_vcmd)/self.dt,-12,12)
        old_retained=self.success
        old_entered=self.entered
        old_escaped=self.escaped
        for k in range(round(self.dt/c.timestep)):
            fraction=(k+1)*c.timestep/self.dt
            qt=self.last_qcmd+(qcmd-self.last_qcmd)*fraction
            s.torque(qt,vcmd,acmd)
            if self.tool=='gripper':
                if d.time>=self.flight-.12+.06*self.residual[5]+.03:
                    self.closed_aperture=max(.034,self.closed_aperture-.10*c.timestep)
                d.ctrl[7:9]=[self.closed_aperture/2,-self.closed_aperture/2]
            mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
            self._metrics()
        self.last_qcmd=qcmd;self.last_vcmd=vcmd;self.steps_seen+=1
        if not np.isfinite(d.qpos).all() or any(w.number for w in d.warning):raise RuntimeError('MuJoCo numerical failure')
        local,relative,_=self.local_state()
        error=float(np.linalg.norm(d.site_xpos[s.site]-self.target))
        track=float(np.exp(-error**2/.015**2)) if not self.contact else 0.
        stable=float(np.exp(-np.dot(local[:2],local[:2])/.025**2)*np.exp(-np.dot(relative,relative)/2)) if self.contact else 0.
        terms=dict(interception=.15*track,retention=.4*float(self.inside),stability=.08*stable,
                   action=-.002*float(np.dot(a,a)),smoothness=-.01*float(np.dot(a-old_action,a-old_action)),
                   success=20.*float(self.success and not old_retained))
        if self.soft_landing and self.tool=='box':
            entered_now=self.entered and not old_entered
            speed=self.entry_speed if self.entry_speed is not None else float(np.linalg.norm(relative))
            terms.update(entry=3.*float(entered_now),
                         gentle_entry=3.*float(entered_now)*float(np.exp(-speed**2/4.)),
                         escape=-3.*float(self.escaped and not old_escaped),
                         retention_success=10.*float(self.success and not old_retained))
        if self.velocity_match and self.tool=='box' and not self.contact:
            time_left=self.flight-float(d.time)
            near_time=0. <= time_left <= .12
            near_target=np.linalg.norm(local[:2]) < .08
            if near_time and near_target:
                rel_speed=np.linalg.norm(relative)
                terms['preimpact_velocity']=.08*float(np.exp(-rel_speed**2/4.))
        reward=sum(terms.values());self.return_sum+=reward;self.reward_terms=terms
        # Terminate genuinely unrecoverable misses, never just a transient loss of contact.
        ball=s.read();lost=bool(ball.position[2]<.04 or np.linalg.norm(ball.position-d.site_xpos[s.site])>2.5)
        terminated=bool(self.success or lost);truncated=bool(d.time>=c.duration-1e-8)
        self.done=terminated or truncated
        if self.record:self.trace.append(dict(t=float(d.time),qpos=d.qpos.tolist(),ball=ball.position.tolist(),opening=d.site_xpos[s.site].tolist()))
        return self.observation(),float(reward),terminated,truncated,dict(result=self.result(),reward_terms=terms)
    @property
    def success(self):return self.metrics.retained if self.tool=='box' else self.retention.success
    def _metrics(self):
        s=self.sim;d=s.data;c=self.cfg;local,relative,opening_v=self.local_state()
        touch=False;both=[False,False];rim=False;finger=False
        for con in d.contact:
            if s.ball_geom not in (con.geom1,con.geom2):continue
            other=con.geom2 if con.geom1==s.ball_geom else con.geom1
            touch|=other in self.tool_geoms
            if self.tool=='box' and other in self.tool_geoms:
                cp=d.site_xmat[s.site].reshape(3,3).T@(con.pos-d.site_xpos[s.site]);rim|=cp[2]>-.012
            for j in range(2):both[j]|=other in self.fingers[j]
        finger=any(both)
        if self.tool=='gripper':touch=finger
        if touch and self.first_contact is None:self.first_contact=float(d.time);self.entry_speed=float(np.linalg.norm(relative))
        self.contact|=touch;self.rim|=rim
        self.max_movement=max(self.max_movement,float(np.linalg.norm(d.site_xpos[s.site]-self.start_pos)))
        if self.intercept_error is None and d.time>=self.flight:self.intercept_error=float(np.linalg.norm(d.site_xpos[s.site]-self.target))
        if self.tool=='box':
            self.metrics.update(float(d.time),local,rim,touch)
            self.entered=self.metrics.entered;self.escaped=self.metrics.escaped
            self.inside=bool(self.metrics.since is not None)
        elif self.tool=='paddle':
            self.inside=bool(self.contact and np.linalg.norm(local[:2])+c.radius<=.075 and c.radius-.003<=local[2]<=c.radius+.008)
            if self.first_contact is not None:
                if touch and not self.was_touch and d.time-self.first_contact>.01:self.bounce_done=True
                if not self.bounce_done:self.first_bounce=max(self.first_bounce,float(local[2]-c.radius))
            self.retention.update(float(d.time),self.inside)
        else:
            gap=abs(d.qpos[7])+abs(d.qpos[8])
            if self.previous_local is not None and self.previous_local[2]>.006>=local[2]:
                self.entered|=bool(abs(local[0])+c.radius<=gap/2+.001 and abs(local[1])<.012)
            self.finger_strike|=bool(finger and not self.entered)
            self.inside=bool(self.entered and all(both) and abs(local[2])<.035 and abs(local[1])<.025)
            self.escaped|=bool(self.contact and np.linalg.norm(local)>.1)
            self.retention.update(float(d.time),self.inside)
        self.previous_local=local.copy();self.was_touch=touch
    def result(self):
        outcome=self.metrics.result()['outcome'] if self.tool=='box' else ('catch' if self.success else 'escaped_after_contact' if self.escaped else 'contact_not_retained' if self.contact else 'miss')
        return dict(tool=self.tool,success=bool(self.success),outcome=outcome,entered=bool(self.entered),
                    contact=bool(self.contact),rim_hit=bool(self.rim),finger_strike=bool(self.finger_strike),
                    retention_s=float(self.metrics.max_retention if self.tool=='box' else self.retention.maximum),
                    first_bounce_m=self.first_bounce if self.tool=='paddle' else None,
                    first_bounce_complete=self.bounce_done,impact_relative_speed_m_s=self.entry_speed,
                    arm_movement_m=self.max_movement,intercept_error_m=self.intercept_error,
                    episode_return=self.return_sum,episode_duration_s=float(self.sim.data.time),
                    throw=asdict(self.throw),meta=self.meta,rejected_before_launch=self.rejected)

# Keep physics usable without installing ML dependencies; Gym is only needed to train.
def gym_env(tool='box',seed=0,training=False,total_steps=150000,printed_box=False):
    import gymnasium as gym
    class Adapter(gym.Env):
        metadata={'render_modes':[]}
        def __init__(self):
            self.core=CatchCore(tool,seed,training,total_steps,printed_box=printed_box)
            self.action_space=gym.spaces.Box(-1.,1.,shape=(7,),dtype=np.float32)
            self.observation_space=gym.spaces.Box(-10.,10.,shape=self.core.observation().shape,dtype=np.float32)
        def reset(self,seed=None,options=None):
            super().reset(seed=seed)
            return self.core.reset(seed=seed),{}
        def step(self,action):return self.core.step(action)
    return Adapter()
