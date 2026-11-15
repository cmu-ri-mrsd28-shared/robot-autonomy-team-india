"""Separate FrankaPy executor. NumPy only; never imports MuJoCo or legacy code."""
import json
import time
import numpy as np
if __package__:
    from .panda_model import Panda,QMIN,QMAX,rotvec,duration,validate_config
else:
    from panda_model import Panda,QMIN,QMAX,rotvec,duration,validate_config


def validate(plan):
    from hashlib import sha256
    cfg=validate_config(plan['config'])
    if plan.get('schema')!=1 or plan['config_hash']!=sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest():
        raise ValueError('Invalid plan/config fingerprint')
    q=np.asarray(plan['start_q'],dtype=float)
    if q.shape!=(7,) or not np.isfinite(q).all() or np.any(q<QMIN) or np.any(q>QMAX): raise ValueError('Invalid start joints')
    robot=Panda(cfg)
    for action in plan['actions']:
        if action['kind']=='move':
            a=np.asarray(action['start_q']); b=np.asarray(action['goal_q'])
            if (a.shape!=(7,) or b.shape!=(7,) or not np.isfinite([a,b]).all()
                    or np.max(abs(a-q))>.001 or np.any(b<QMIN) or np.any(b>QMAX)):
                raise ValueError('Invalid/discontinuous joint segment')
            t=float(action['duration'])
            if not np.isfinite(t) or t+1e-6<duration(a,b,cfg['motion']): raise ValueError('Unsafe segment timing')
            if 'target_tcp' in action:
                target=np.asarray(action['target_tcp']); actual=robot.fk(b)
                if np.linalg.norm(target[:3,3]-actual[:3,3])>.001 or np.linalg.norm(rotvec(target[:3,:3]@actual[:3,:3].T))>.01:
                    raise ValueError('Joint goal disagrees with planned TCP')
            q=b
        elif action['kind']=='grip':
            if not 0<=action['width']<=.08: raise ValueError('Invalid jaw width')
        elif action['kind']!='checkpoint': raise ValueError('Unknown action')
    return robot


class Executor:
    def __init__(self,arm,plan):
        self.arm,self.plan=arm,plan; self.robot=validate(plan)

    def pose_check(self,q):
        expected=self.robot.fk(q); actual=self.arm.get_pose()
        if (np.asarray(actual.translation).shape!=(3,) or np.asarray(actual.rotation).shape!=(3,3)
                or not np.isfinite(actual.translation).all() or not np.isfinite(actual.rotation).all()):
            raise RuntimeError('Invalid measured tool pose')
        error=np.linalg.norm(actual.translation-expected[:3,3])
        angle=np.linalg.norm(rotvec(actual.rotation@expected[:3,:3].T))
        if error>.005 or angle>.035:
            raise RuntimeError(f'Model/tool frame or endpoint mismatch: {error:.5f} m, {angle:.5f} rad')

    def joints(self):
        q=np.asarray(self.arm.get_joints(),dtype=float)
        if q.shape!=(7,) or not np.isfinite(q).all(): raise RuntimeError('Invalid measured joint feedback')
        return q

    def cancel(self):
        # A second cancel on a terminal goal can cause actionlib PREEMPTING/DONE errors.
        if self.arm._client.get_state() in (0,1,6,7):
            import rospy
            self.arm._client.cancel_goal()
            self.arm._client.wait_for_result(rospy.Duration.from_sec(2.))

    def execute(self):
        q=self.joints()
        if np.max(abs(q-self.plan['start_q']))>.02: raise RuntimeError('Robot does not match plan start; capture state and replan')
        self.pose_check(q)
        cfg=self.plan['config']
        try:
            for index,action in enumerate(self.plan['actions']):
                print(index,action['label'],flush=True)
                if action['kind']=='move':
                    if np.max(abs(self.joints()-action['start_q']))>.02:
                        raise RuntimeError('Measured joints disagree with segment start')
                    self.arm.goto_joints(action['goal_q'],duration=action['duration'],
                                         use_impedance=False,block=False,ignore_errors=False)
                    deadline=time.monotonic()+action['duration']+10
                    while not self.arm.is_skill_done(ignore_errors=False):
                        if time.monotonic()>deadline: raise TimeoutError('Joint motion timed out')
                        time.sleep(.01)
                    status=self.arm._get_current_franka_interface_status().franka_interface_status
                    if status.error_description: raise RuntimeError(status.error_description)
                    if self.arm._client.get_state()!=3: raise RuntimeError('Joint action did not succeed')
                    if np.max(abs(self.joints()-action['goal_q']))>.02:
                        raise RuntimeError('Joint target not reached')
                    self.pose_check(np.asarray(action['goal_q']))
                elif action['kind']=='grip':
                    closed=action['width']==0
                    self.arm.goto_gripper(action['width'],grasp=closed,speed=cfg['gripper']['speed'],
                                          force=cfg['gripper']['force'] if closed else 0.,
                                          epsilon_inner=.004,epsilon_outer=cfg['cube_size']+.004 if closed else .004,
                                          block=True,ignore_errors=False)
                    width=self.arm.get_gripper_width()
                    if not np.isfinite(width): raise RuntimeError('Invalid gripper width feedback')
                    if closed and not cfg['cube_size']-.012<width<cfg['cube_size']+.005:
                        raise RuntimeError('Grasp width inconsistent with the cube; stopping before lift')
                    if not closed and width<cfg['cube_size']+.01: raise RuntimeError('Jaw failed to open')
                else:
                    print('Nominal checkpoint only; inspect the physical arrangement:',action['locations'],flush=True)
        except BaseException:
            try: self.cancel()
            except Exception as error: print('Cancellation error:',error,flush=True)
            # Never release, reset, retry or drive home as error recovery.
            raise

