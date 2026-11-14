"""Public FrankaPy adapter and a no-ROS dry-run backend.

API reference: https://iamlab-cmu.github.io/frankapy/_modules/frankapy/franka_arm.html
"""

import time
import numpy as np


class DryRunBackend:
    def __init__(self, config):
        self.config = config
        self.xyz = np.asarray(config['preview_start_tool_xyz'],dtype=float)
        self.rotation = np.asarray(config['tool_rotation_EW'],dtype=float)
        self.width = config['gripper_open_width_m']
        self.trace = []

    def pose(self):
        return self.xyz.copy(), self.rotation.copy()

    def move(self, xyz, rotation, duration, label):
        self.xyz, self.rotation = np.array(xyz), np.array(rotation)
        self.trace.append({'operation':'goto_pose','label':label,'translation':self.xyz.tolist(),
                           'rotation':self.rotation.tolist(),'duration_s':float(duration)})

    def gripper(self, grasp, label):
        self.width = self.config['block_width_m'] if grasp else self.config['gripper_open_width_m']
        self.trace.append({'operation':'goto_gripper','label':label,'grasp':grasp,
            'commanded_width_m':self.config['gripper_grasp_width_m'] if grasp else self.width,
            'assumed_feedback_width_m':self.width})
        return self.width

    def pause(self, seconds):
        self.trace.append({'operation':'pause','duration_s':seconds})

    def check_width(self, grasp, label, commanded_width=None):
        self.trace.append({'operation':'gripper_feedback','label':label,'grasp':grasp,
                           'assumed_feedback_width_m':self.width})
        return self.width

    def stop(self):
        self.trace.append({'operation':'stop'})


class FrankaPyBackend:
    def __init__(self, config, arm=None, transform_cls=None, clock=time.monotonic, sleep=time.sleep):
        if arm is None:
            from frankapy import FrankaArm
            arm = FrankaArm(robot_num=config['robot_num'])
        if transform_cls is None:
            from autolab_core import RigidTransform
            transform_cls = RigidTransform
        self.arm = arm
        self.transform_cls = transform_cls
        self.config = config
        self.clock, self.sleep = clock, sleep
        self.trace = []
        self.last_motion = None

    def joint_diagnostics(self):
        """Optional read-only telemetry; never substitute it for motion checks."""
        state = {}
        for name, method in (('joints_rad','get_joints'),('joint_velocities_rad_s','get_joint_velocities')):
            getter = getattr(self.arm,method,None)
            if getter is None:
                state[name] = None
                continue
            try:
                values = np.asarray(getter(),dtype=float)
                if values.shape != (7,) or not np.isfinite(values).all():
                    raise ValueError('Expected seven finite values')
                state[name] = values.tolist()
            except Exception as error:
                state[name] = None
                state[name+'_read_error'] = str(error)
        return state

    def pose(self):
        pose = self.arm.get_pose()
        xyz, rot = np.array(pose.translation), np.array(pose.rotation)
        if xyz.shape != (3,) or rot.shape != (3,3) or not np.isfinite(xyz).all() or not np.isfinite(rot).all():
            raise RuntimeError('Invalid or nonfinite tool-pose feedback')
        return xyz, rot

    def move(self, xyz, rotation, duration, label):
        pose = self.transform_cls(rotation=np.asarray(rotation),translation=np.asarray(xyz),
                                  from_frame='franka_tool',to_frame='world')
        entry = {'operation':'goto_pose','label':label,'translation':np.asarray(xyz).tolist(),
                 'rotation':np.asarray(rotation).tolist(),'duration_s':float(duration),
                 'use_impedance':self.config['use_impedance'],'status':'STARTED',
                 'start_joint_state':self.joint_diagnostics()}
        self.trace.append(entry)
        self.last_motion = entry
        try:
            self.arm.goto_pose(pose,duration=duration,use_impedance=self.config['use_impedance'],
                              block=False,ignore_errors=False,ignore_virtual_walls=False,
                              skill_desc=label)
            deadline = self.clock() + duration + self.config['motion_timeout_margin_s']
            while not self.arm.is_skill_done(ignore_errors=False):
                if self.clock() > deadline:
                    raise TimeoutError(f'Motion timed out: {label}')
                self.sleep(.02)
            actual_xyz, actual_rotation = self.pose()
            pos_error = float(np.linalg.norm(actual_xyz-xyz))
            angle = float(np.arccos(np.clip((np.trace(actual_rotation.T@rotation)-1)/2,-1.,1.)))
            entry.update(actual_translation=actual_xyz.tolist(),translation_error_m=pos_error,
                         rotation_error_rad=angle)
            if pos_error > self.config['translation_tolerance_m'] or angle > self.config['rotation_tolerance_rad']:
                raise RuntimeError(f'Motion feedback outside tolerance: {label}; position={pos_error:.4f} m, angle={angle:.4f} rad')
            entry['status'] = 'COMPLETED'
        except BaseException as error:
            # Preserve the controller error before any further robot reads.
            # A faulted interface must not hide it behind a telemetry wait.
            entry.update(status='FAILED',error=f'{type(error).__name__}: {error}')
            raise

    def gripper(self, grasp, label):
        width = self.config['gripper_grasp_width_m'] if grasp else self.config['gripper_open_width_m']
        self.arm.goto_gripper(width,grasp=grasp,speed=self.config['gripper_speed_m_s'],
                              force=self.config['grasp_force_N'] if grasp else 0.,
                              epsilon_inner=self.config['gripper_width_tolerance_m'],
                              epsilon_outer=(self.config['block_width_m']-width+self.config['gripper_width_tolerance_m']) if grasp else self.config['gripper_width_tolerance_m'],
                              block=True,ignore_errors=False,skill_desc=label)
        return self.check_width(grasp, label, width)

    def check_width(self, grasp, label, commanded_width=None):
        measured = float(self.arm.get_gripper_width())
        expected = self.config['block_width_m'] if grasp else self.config['gripper_open_width_m']
        self.trace.append({'operation':'gripper_feedback','label':label,'grasp':grasp,
                           'commanded_width_m':commanded_width,'measured_width_m':measured})
        if not np.isfinite(measured) or abs(measured-expected) > self.config['gripper_width_tolerance_m']:
            raise RuntimeError(f'Unexpected gripper width during {label}: {measured:.4f} m, expected near {expected:.4f} m')
        return measured

    def pause(self, seconds):
        self.sleep(seconds)

    def stop(self):
        # Preserve any held block. Do not automatically open the hand or reset joints.
        self.arm.stop_skill()
