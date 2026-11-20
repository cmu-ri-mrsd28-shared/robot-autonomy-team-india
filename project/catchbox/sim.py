"""robosuite model composition + native MuJoCo CPU dynamics (no renderer needed)."""
import time
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Panda
from robosuite.models.grippers import gripper_factory
from .core import BallState, RobotState, BoxMetrics, crossing

def vec(x):
    return ' '.join(str(float(v)) for v in x)

class Simulation:
    def __init__(self, cfg, tool="box"):
        self.tool = tool
        cfg.validate()
        self.cfg = cfg
        world = MujocoWorldBase()
        panda = Panda()
        panda.add_gripper(gripper_factory('PandaGripper'))
        world.merge(panda)
        option = world.root.find('option')
        option.set('timestep', str(cfg.timestep))
        option.set('gravity', '0 0 -9.81')
        option.set('integrator', 'implicitfast')
        ET.SubElement(world.worldbody, 'geom', name='floor', type='plane', size='3 3 .1', rgba='.12 .16 .2 1')
        ET.SubElement(world.worldbody, 'light', pos='0 0 3', dir='0 0 -1')
        # Determine mounting transform at the known home configuration. The box
        # is carried in front of the hand by an ideal rigid side bracket.
        initial = world.get_model(mode='mujoco')
        d = mujoco.MjData(initial)
        d.qpos[:7] = panda.init_qpos
        mujoco.mj_forward(initial, d)
        hand_name = panda.eef_name['right']
        hand_id = initial.body(hand_name).id
        rotation = d.xmat[hand_id].reshape(3,3).copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rotation.T.ravel())
        if cfg.printed_box and tool=='box':
            grip_initial=d.site_xpos[initial.site('gripper0_grip_site').id].copy()
            opening_initial=grip_initial+np.array([.105,0,.05])
            mount_pos=rotation.T@(opening_initial-d.xpos[hand_id])
        else:
            mount_pos = rotation.T @ np.array([0.17, 0, 0.02])
        hand = world.worldbody.find(f'.//body[@name="{hand_name}"]')
        box = ET.SubElement(hand, 'body', name='catch_box', pos=vec(mount_pos), quat=vec(quat))
        if tool == 'box' and cfg.printed_box:
            mesh_path=Path(__file__).resolve().parent.parent/'assets'/'armBox-inmeters.STL'
            ET.SubElement(world.asset,'mesh',name='printed_box_mesh',file=str(mesh_path))
            ET.SubElement(box,'geom',name='printed_box_visual',type='mesh',mesh='printed_box_mesh',mass='0',contype='0',conaffinity='0',rgba='.18 .65 .7 1',group='1')
            # STL's non-convex cavity cannot be used as a single MuJoCo mesh collider.
            # Five rectangular pieces approximate its 120 mm cavity and 90 mm depth.
            common=dict(type='box',friction=vec(cfg.friction),solref=vec(cfg.solref),solimp=vec(cfg.solimp),condim='4',group='0',rgba='.18 .65 .7 0')
            ET.SubElement(box,'geom',name='box_bottom',pos='0 0 -.095',size='.075 .075 .005',mass=str(cfg.box_mass*.2),**common)
            for name,pos,size in [('xp',[.0675,0,-.045],[.0075,.075,.045]),('xm',[-.0675,0,-.045],[.0075,.075,.045]),('yp',[0,.0675,-.045],[.06,.0075,.045]),('ym',[0,-.0675,-.045],[.06,.0075,.045])]:
                ET.SubElement(box,'geom',name='box_'+name,pos=vec(pos),size=vec(size),mass=str(cfg.box_mass*.16),**common)
            ET.SubElement(box,'geom',name='handle_mass',type='box',pos='-.105 0 -.05',size='.03 .025 .025',mass=str(cfg.box_mass*.16),contype='0',conaffinity='0',rgba='0 0 0 0',group='0')
            ET.SubElement(box,'site',name='opening',pos='0 0 0',size='.004',rgba='0 1 0 1')
        elif tool == 'box':
            w, dep, h, thick = cfg.box_width/2, cfg.box_depth/2, cfg.box_height, .006
            common = dict(type='box', friction=vec(cfg.friction), solref=vec(cfg.solref),
                          solimp=vec(cfg.solimp), condim='4', rgba='.18 .65 .7 1', mass=str(cfg.box_mass/5))
            ET.SubElement(box, 'geom', name='box_bottom', pos=vec([0,0,-thick/2]), size=vec([w+thick,dep+thick,thick/2]), **common)
            for name, pos, size in [
                ('xp',[w+thick/2,0,h/2],[thick/2,dep+thick,h/2]),
                ('xm',[-w-thick/2,0,h/2],[thick/2,dep+thick,h/2]),
                ('yp',[0,dep+thick/2,h/2],[w,thick/2,h/2]),
                ('ym',[0,-dep-thick/2,h/2],[w,thick/2,h/2])]:
                ET.SubElement(box, 'geom', name='box_'+name, pos=vec(pos), size=vec(size), **common)
            ET.SubElement(box, 'site', name='opening', pos=vec([0,0,h]), size='.004', rgba='0 1 0 1')
            # Visual bracket; rigid attachment idealization, not a grasp simulation.
            ET.SubElement(box, 'geom', name='bracket', type='box', pos='-.085 0 -.035',
                          size='.085 .012 .012', mass='.06', contype='0', conaffinity='0', rgba='.4 .4 .4 1')
        elif tool == 'paddle':
            ET.SubElement(box, 'geom', name='paddle', type='cylinder', size='.075 .005',
                          mass='.17', rgba='.8 .08 .08 1', friction=vec(cfg.friction),
                          solref=vec(cfg.solref), solimp=vec(cfg.solimp), condim='4')
            ET.SubElement(box, 'geom', name='handle', type='box', pos='-.09 0 0',
                          size='.055 .012 .006', mass='.03', rgba='.6 .4 .2 1')
            ET.SubElement(box, 'site', name='opening', pos='0 0 .005', size='.003')
        elif tool == 'gripper':
            # No catcher attachment: use the stock finger tip frame directly.
            hand.remove(box)
        else:
            raise ValueError(tool)
        ball = ET.SubElement(world.worldbody, 'body', name='ball', pos='0 0 2')
        ET.SubElement(ball, 'freejoint', name='ball_joint')
        ET.SubElement(ball, 'geom', name='ball_geom', type='sphere', size=str(cfg.radius), mass=str(cfg.mass),
                      rgba='1 .48 .05 1', friction=vec(cfg.friction), solref=vec(cfg.solref), solimp=vec(cfg.solimp), priority='1', condim='4')
        self.model = world.get_model(mode='mujoco')
        if tool=='box' and cfg.printed_box:
            for name in ['box_bottom','box_xp','box_xm','box_yp','box_ym','handle_mass']:
                self.model.geom_group[self.model.geom(name).id]=3
        # Change render groups AFTER mass/inertia inference (robosuite infers group 0).
        for name in ['floor','ball_geom'] + ([] if tool=='box' and cfg.printed_box else ['box_bottom','box_xp','box_xm','box_yp','box_ym','bracket'] if tool == 'box' else ['paddle','handle'] if tool == 'paddle' else []):
            self.model.geom_group[self.model.geom(name).id] = 1
        self.data = mujoco.MjData(self.model)
        self.ikdata = mujoco.MjData(self.model)
        self.home = panda.init_qpos.copy()
        self.site = self.model.site('gripper0_grip_site' if tool == 'gripper' else 'opening').id
        self.grip_site = self.model.site('gripper0_grip_site').id
        self.box_id = self.model.body('catch_box').id if tool != 'gripper' else None
        self.ball_id = self.model.body('ball').id
        self.ball_geom = self.model.geom('ball_geom').id
        self.ball_q = self.model.joint('ball_joint').qposadr[0]
        self.ball_v = self.model.joint('ball_joint').dofadr[0]
        self.box_geoms = {self.model.geom('box_'+n).id for n in ['bottom','xp','xm','yp','ym']} if tool == 'box' else set()
        self.jacp = np.zeros((3,self.model.nv))
        self.jacr = self.jacp.copy()
        self.reset()
        self.nominal_opening = self.data.site_xpos[self.site].copy()
        self.nominal_grip = self.data.site_xpos[self.grip_site].copy()

    def read(self):
        return BallState(float(self.data.time), self.data.xpos[self.ball_id].copy(),
                         self.data.qvel[self.ball_v:self.ball_v+3].copy(),
                         acceleration=self.data.qacc[self.ball_v:self.ball_v+3].copy())

    def robot_state(self):
        return RobotState(float(self.data.time), self.data.qpos[:7].copy(), self.data.qvel[:7].copy(),
                          self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3,3).copy())

    def torque(self, target, velocity=None, acceleration=None):
        # Computed-torque joint tracking, with actual MuJoCo motor saturation.
        # Model-based feedforward is simulation-only until a hardware controller exists.
        velocity = np.zeros(7) if velocity is None else velocity
        acceleration = np.zeros(7) if acceleration is None else acceleration
        mass = np.zeros((self.model.nv,self.model.nv))
        mujoco.mj_fullM(self.model,mass,self.data.qM)
        desired = acceleration+400*(target-self.data.qpos[:7])+40*(velocity-self.data.qvel[:7])
        tau = mass[:7,:7]@desired+self.data.qfrc_bias[:7]
        self.data.ctrl[:7] = np.clip(tau, self.model.actuator_ctrlrange[:7,0], self.model.actuator_ctrlrange[:7,1])
        self.data.ctrl[7:] = [.025,-.025] if self.tool=='box' and self.cfg.printed_box else 0

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = self.home
        if self.tool=='box' and self.cfg.printed_box:self.data.qpos[7:9]=[.025,-.025]
        self.data.qpos[self.ball_q:self.ball_q+3] = [0,0,3]
        mujoco.mj_forward(self.model, self.data)
        # Settle arm before release; ball reset after warmup.
        for _ in range(round(.3/self.cfg.timestep)):
            self.torque(self.home)
            mujoco.mj_step(self.model,self.data)
        self.data.time = 0

    def inverse_kinematics(self, position, rotation=None):
        rotation = np.eye(3) if rotation is None else rotation
        d, m = self.ikdata, self.model
        d.qpos[:] = self.data.qpos
        for _ in range(65):
            mujoco.mj_forward(m,d)
            r = d.site_xmat[self.site].reshape(3,3)
            orient = sum((np.cross(r[:,i],rotation[:,i]) for i in range(3)))*.5
            err = np.r_[position-d.site_xpos[self.site], orient]
            if np.linalg.norm(err) < 1e-5:
                break
            mujoco.mj_jacSite(m,d,self.jacp,self.jacr,self.site)
            jac = np.vstack([self.jacp[:,:7],self.jacr[:,:7]])
            dq = jac.T @ np.linalg.solve(jac@jac.T+np.eye(6)*1e-4, err)
            d.qpos[:7] += np.clip(dq,-.08,.08)
            d.qpos[:7] = np.clip(d.qpos[:7],m.jnt_range[:7,0]+.01,m.jnt_range[:7,1]-.01)
        return d.qpos[:7].copy(), float(np.linalg.norm(position-d.site_xpos[self.site]))

    def rollout(self, throw, viewer=None, trace=False, ball_source=None):
        self.reset()
        d,m,c = self.data,self.model,self.cfg
        d.qpos[self.ball_q:self.ball_q+3] = throw.position
        d.qvel[self.ball_v:self.ball_v+3] = throw.velocity
        mujoco.mj_forward(m,d)
        metrics = BoxMetrics(c)
        source = self if ball_source is None else ball_source
        prediction = crossing(source.read(), self.nominal_opening[2])
        target = self.nominal_opening.copy() if prediction is None else prediction[1]
        goal, ik_error = self.inverse_kinematics(target)
        startq = d.qpos[:7].copy()
        flight = 0 if prediction is None else prediction[0]
        # Quintic move ends before predicted crossing where possible; velocity-limited.
        move_time = max(.12, float(np.max(abs(goal-startq)))*1.875/1.5)
        rows = []
        wall_start = time.perf_counter()
        steps = round(c.duration/c.timestep)
        period = round(c.control_dt/c.timestep)
        intercept_error = None
        intercept_speed = None
        entry_speed = None
        held_before_crossing = False
        warnings_before = np.array([w.number for w in d.warning])
        for k in range(steps):
            s = np.clip(d.time/move_time,0,1)
            blend = 10*s**3-15*s**4+6*s**5
            speed = (30*s**2-60*s**3+30*s**4)/move_time
            accel = (60*s-180*s**2+120*s**3)/move_time**2
            self.torque(startq+(goal-startq)*blend,(goal-startq)*speed,(goal-startq)*accel)
            mujoco.mj_step(m,d)
            mujoco.mj_forward(m,d)
            rot = d.site_xmat[self.site].reshape(3,3)
            local = rot.T@(d.xpos[self.ball_id]-d.site_xpos[self.site])
            rim, contact = False, False
            for con in d.contact:
                if self.ball_geom not in (con.geom1,con.geom2):
                    continue
                other = con.geom2 if con.geom1 == self.ball_geom else con.geom1
                contact |= other != m.geom('floor').id
                if other in self.box_geoms:
                    cp = rot.T@(con.pos-d.site_xpos[self.site])
                    rim |= cp[2] > -.012
            was_entered = metrics.entered
            metrics.update(float(d.time),local,rim,contact)
            if metrics.entered and not was_entered:
                mujoco.mj_jacSite(m,d,self.jacp,self.jacr,self.site)
                entry_speed = float(np.linalg.norm(self.jacp@d.qvel))
            if intercept_error is None and d.time >= flight:
                intercept_error = float(np.linalg.norm(d.site_xpos[self.site]-target))
                mujoco.mj_jacSite(m,d,self.jacp,self.jacr,self.site)
                intercept_speed = float(np.linalg.norm(self.jacp@d.qvel))
                held_before_crossing = bool(d.time >= move_time and intercept_speed < .02 and intercept_error < .01)
            if k % period == 0:
                if trace:
                    rows.append(dict(t=float(d.time),ball=d.xpos[self.ball_id].tolist(),
                                     opening=d.site_xpos[self.site].tolist(),local=local.tolist(),q=d.qpos[:7].tolist(),qpos=d.qpos.tolist()))
                if viewer is not None:
                    if not viewer.is_running():
                        raise KeyboardInterrupt('Viewer closed')
                    viewer.sync()
                    time.sleep(max(0,d.time-(time.perf_counter()-wall_start)))
        if not np.isfinite(d.qpos).all() or np.any(np.array([w.number for w in d.warning]) > warnings_before):
            raise RuntimeError('MuJoCo numerical warning; discard rollout')
        return {**metrics.result(), 'predicted_crossing_s': flight, 'target':target.tolist(),
                'move_time_s':move_time, 'ik_error_m':ik_error,'intercept_error_m':intercept_error,
                'opening_speed_at_entry_m_s':entry_speed, 'opening_speed_at_crossing_m_s':intercept_speed,
                'held_before_crossing':held_before_crossing, 'physics_steps':steps}, rows
