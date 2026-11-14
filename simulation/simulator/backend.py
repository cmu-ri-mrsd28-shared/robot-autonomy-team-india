"""Contact-driven Panda simulation. Block qpos is never written during execution."""

from pathlib import Path
import numpy as np
import mujoco as mj

from .scene import build_scene, grid_xy, staging
from .motion import IK, rotation, min_jerk
from .recording import Recorder
from .viewer_timing import ViewerTiming


class Simulation:
    def __init__(self, config, directory, record=True, gui=False, speed=1., viewer_fps=30.):
        self.config = config
        self.directory = Path(directory)
        self.model = mj.MjModel.from_xml_path(str(build_scene(config, self.directory / "scene.xml")))
        self.data = mj.MjData(self.model)
        self.arm_qpos = np.array([int(self.model.joint(f"joint{i}").qposadr[0]) for i in range(1,8)])
        self.arm_dofs = np.array([int(self.model.joint(f"joint{i}").dofadr[0]) for i in range(1,8)])
        self.actuators = np.array([self.model.actuator(f"actuator{i}").id for i in range(1,8)])
        self.grip_actuator = self.model.actuator("actuator8").id
        self.finger_qpos = [int(self.model.joint(f"finger_joint{i}").qposadr[0]) for i in (1,2)]
        self.site = self.model.site("grasp_center").id
        self.ik = IK(self.model, self.site, self.arm_qpos, self.arm_dofs)
        self.home_slots = staging(config)
        self.blocks = {name:self.model.body(name).id for name in self.home_slots}
        self.block_geoms = {self.model.geom(f"{name}_geom").id:name for name in self.blocks}
        self.finger_bodies = {self.model.body(name).id for name in ("left_finger","right_finger")}
        self.robot_geoms = set()
        for geom in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[geom])
            while body:
                if body == self.model.body("link0").id:
                    self.robot_geoms.add(geom)
                    break
                body = int(self.model.body_parentid[body])
        self.kp = np.array([1100,1100,1000,900,600,450,300.])
        self.kd = np.array([65,65,60,55,40,32,25.])
        self.q_des = np.array([0.,-0.5,0.,-2.,0.,1.5,0.7854])
        self.grip = config["open_finger_position"]
        self.data.qpos[self.arm_qpos] = self.q_des
        self.data.qpos[self.finger_qpos] = self.grip
        mj.mj_forward(self.model,self.data)
        self.recorder = Recorder(self.model, self.directory, record)
        self.viewer = None
        self.viewer_timing = ViewerTiming(speed, viewer_fps) if gui else None
        if gui:
            from mujoco import viewer
            self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.cam.lookat[:] = [.47,0,.10]
            self.viewer.cam.distance = 1.45
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -45
        self.active = None
        self.phase = "initializing"
        self.steps = 0
        self.transfers = []
        self.worst_tracking = 0.
        self.allowed_contact_penetration = 0.
        self.hold(0.6)
        self.initial_positions = self.positions()
        self.recorder.frame(self.data, force=True, snapshot="initial")

    def positions(self):
        return {name:self.data.xpos[body].copy() for name,body in self.blocks.items()}

    def validate_grasp_model(self):
        """Probe real collision meshes on scratch data, never the running world.

        Sample approach height and aperture for each cardinal/diagonal neighbor.
        This validates the configured candidate grasps, not every possible yaw.
        """
        scratch=mj.MjData(self.model)
        scratch.qpos[:]=self.data.qpos
        target_address=int(self.model.joint("R0_joint").qposadr[0])
        neighbor_address=int(self.model.joint("B0_joint").qposadr[0])
        center=np.array([*self.config["grid_center"],self.config["block_half_size"]])
        scratch.qpos[target_address:target_address+3]=center
        neighbor_geom=self.model.geom("B0_geom").id
        spacing=self.config["grid_spacing"]
        probes=[]
        for name,offset,expected in (
            ("north",(1,0),["EW"]),("south",(-1,0),["EW"]),
            ("east",(0,-1),["NS"]),("west",(0,1),["NS"]),
            ("north_east",(1,-1),["NS","EW"]),
            ("north_west",(1,1),["NS","EW"]),
            ("south_east",(-1,-1),["NS","EW"]),
            ("south_west",(-1,1),["NS","EW"]),
        ):
            scratch.qpos[neighbor_address:neighbor_address+3]=center+np.r_[np.array(offset)*spacing,0.]
            clear=[]
            penetrations={}
            for axis in ("NS","EW"):
                seed=self.data.qpos[self.arm_qpos].copy()
                worst=0.
                for delta in (.12,.08,.06,.04,.02,.005,0.):
                    height=self.config["block_half_size"]+delta
                    seed=self.ik.solve([*center[:2],height],rotation(axis),seed)
                    scratch.qpos[self.arm_qpos]=seed
                    for aperture in (self.config["closed_finger_position"],self.config["block_half_size"],self.config["open_finger_position"]):
                        scratch.qpos[self.finger_qpos]=aperture
                        mj.mj_forward(self.model,scratch)
                        for contact in scratch.contact:
                            a,b=int(contact.geom1),int(contact.geom2)
                            if (a==neighbor_geom and b in self.robot_geoms) or (b==neighbor_geom and a in self.robot_geoms):
                                worst=max(worst,float(-contact.dist))
                penetrations[axis]=worst
                if worst<0.0001:
                    clear.append(axis)
            if set(clear)!=set(expected):
                raise RuntimeError(f"Grasp abstraction disagrees with collision geometry for {name}: clear={clear}, expected={expected}, penetration={penetrations}")
            probes.append({"neighbor":name,"clear_axes":clear,"max_penetration_m":penetrations})
        print("PASS grasp geometry: all 8 cardinal/diagonal neighbor probes agree",flush=True)
        return probes

    def _contacts(self):
        for contact in self.data.contact:
            self.allowed_contact_penetration=max(self.allowed_contact_penetration,float(-contact.dist))
            if contact.dist > -0.0008:
                continue
            a,b = int(contact.geom1), int(contact.geom2)
            for robot,other in ((a,b),(b,a)):
                if robot not in self.robot_geoms:
                    continue
                body = int(self.model.geom_bodyid[robot])
                if self.block_geoms.get(other) == self.active and body in self.finger_bodies:
                    continue
                label = lambda g: self.model.geom(g).name or self.model.body(int(self.model.geom_bodyid[g])).name
                raise RuntimeError(f"Unexpected collision during {self.phase}: {label(robot)} / {label(other)}, penetration {-contact.dist:.4f} m")
            if a in self.block_geoms and b in self.block_geoms:
                raise RuntimeError(f"Block collision: {self.block_geoms[a]} / {self.block_geoms[b]}")

    def step(self, q_des=None, qd_des=None):
        if q_des is not None:
            self.q_des = np.array(q_des)
        qd = np.zeros(7) if qd_des is None else qd_des
        error = self.q_des-self.data.qpos[self.arm_qpos]
        self.worst_tracking = max(self.worst_tracking,float(np.max(np.abs(error))))
        tau = self.kp*error + self.kd*(qd-self.data.qvel[self.arm_dofs]) + self.data.qfrc_bias[self.arm_dofs]
        limits = self.model.actuator_ctrlrange[self.actuators]
        self.data.ctrl[self.actuators] = np.clip(tau,limits[:,0],limits[:,1])
        self.data.ctrl[self.grip_actuator] = self.grip
        mj.mj_step(self.model,self.data)
        self.steps += 1
        self._contacts()
        self.recorder.frame(self.data)
        if self.viewer:
            if not self.viewer.is_running():
                raise RuntimeError("Viewer closed")
            if self.viewer_timing.update_due(self.data.time):
                self.viewer.sync()

    def hold(self, duration):
        for _ in range(max(1,round(duration/self.model.opt.timestep))):
            self.step()

    def joint_move(self, goal, duration):
        start = self.data.qpos[self.arm_qpos].copy()
        count = max(1,round(duration/self.model.opt.timestep))
        for index in range(1,count+1):
            t = index/count
            s = min_jerk(t)
            ds = (30*t*t-60*t**3+30*t**4)/duration
            self.step(start+(goal-start)*s,(goal-start)*ds)
        self.hold(0.12)
        error = np.max(np.abs(self.data.qpos[self.arm_qpos]-goal))
        if error > 0.035:
            raise RuntimeError(f"Joint target not reached: {error:.4f} rad")

    def move(self, xyz, axis, linear=True):
        xyz = np.asarray(xyz,dtype=float)
        orient = rotation(axis)
        start = self.data.site_xpos[self.site].copy()
        seed = self.data.qpos[self.arm_qpos].copy()
        if not linear:
            goal = self.ik.solve(xyz,orient,seed)
            duration = max(1.2,float(np.max(np.abs(goal-seed)))/0.45)
            self.joint_move(goal,duration)
        else:
            distance = np.linalg.norm(xyz-start)
            count = max(1,int(np.ceil(distance/0.018)))
            for i in range(1,count+1):
                waypoint = start+(xyz-start)*(i/count)
                goal = self.ik.solve(waypoint,orient,seed)
                self.joint_move(goal,max(0.22,float(np.max(np.abs(goal-seed)))/0.5))
                seed=goal
        pos_error = np.linalg.norm(self.data.site_xpos[self.site]-xyz)
        if pos_error > 0.003:
            raise RuntimeError(f"Tool target not reached: {pos_error:.4f} m; actual={self.data.site_xpos[self.site]}, desired={xyz}; joint error={self.data.qpos[self.arm_qpos]-self.q_des}")

    def transfer(self, name, destination, axis):
        self.active = name
        before = self.positions()
        source = before[name]
        height = self.config["transfer_height"]
        half = self.config["block_half_size"]
        print(f"  {name}: {np.round(source[:2],3)} -> {np.round(destination,3)}, {axis}",flush=True)
        self.phase = "source approach"
        self.grip = self.config["open_finger_position"]
        self.move([*source[:2],height],axis,linear=False)
        self.move([*source[:2],half],axis)
        self.phase = "grasp"
        self.grip = self.config["closed_finger_position"]
        self.hold(0.6)
        contact_bodies = set()
        for contact in self.data.contact:
            a,b=int(contact.geom1),int(contact.geom2)
            for finger,block in ((a,b),(b,a)):
                if self.block_geoms.get(block)==name:
                    body = int(self.model.geom_bodyid[finger])
                    if body in self.finger_bodies:
                        contact_bodies.add(body)
        if len(contact_bodies)!=2:
            raise RuntimeError(f"{name}: bilateral finger contacts absent")
        self.phase = "lift"
        self.move([*source[:2],height],axis)
        lifted = self.positions()[name]
        if np.linalg.norm(lifted-self.data.site_xpos[self.site]) > 0.012:
            raise RuntimeError(f"{name}: grasp slipped; block={lifted}, tool={self.data.site_xpos[self.site]}")
        self.phase = "transfer"
        self.move([*destination,height],axis)
        self.phase = "placement"
        self.move([*destination,half+0.0005],axis)
        self.phase = "release"
        self.grip = self.config["open_finger_position"]
        self.hold(0.5)
        self.phase = "retreat"
        self.move([*destination,height],axis)
        self.hold(0.25)
        after = self.positions()
        error = float(np.linalg.norm(after[name][:2]-destination))
        if error > 0.004 or abs(after[name][2]-half)>0.003:
            raise RuntimeError(f"{name}: placement error={error:.4f} m, z={after[name][2]:.4f}")
        moved_others = {other:float(np.linalg.norm(after[other]-before[other]))
                        for other in before if other!=name and np.linalg.norm(after[other]-before[other])>0.003}
        if moved_others:
            raise RuntimeError(f"Neighbors disturbed: {moved_others}")
        self.transfers.append({"block":name,"destination":list(destination),"orientation":axis,
                               "xy_error_m":error,"time":float(self.data.time)})
        self.active = None

    def check_arrangement(self, rows, locations):
        positions = self.positions()
        expected = "".join(rows)
        observed = ["."]*9
        for name,position in positions.items():
            distances = [np.linalg.norm(position[:2]-grid_xy(self.config,i)) for i in range(9)]
            cell = int(np.argmin(distances))
            if distances[cell]<0.015:
                if observed[cell]!=".":
                    raise RuntimeError("Two blocks in one cell")
                observed[cell]=name[0]
            target = grid_xy(self.config,locations[name]) if locations[name] is not None else self.home_slots[name]
            if np.linalg.norm(position[:2]-target)>0.004 or abs(position[2]-self.config["block_half_size"])>0.003:
                raise RuntimeError(f"Unexpected final location of {name}: {position}")
        if "".join(observed)!=expected:
            raise RuntimeError(f"Pattern mismatch: {observed}, expected {expected}")
        return {name:position.tolist() for name,position in positions.items()}

    def snapshot(self, name, label):
        self.phase="park for observation"
        self.move([.34,-.23,.35],"EW",linear=False)
        self.recorder.label=label
        self.hold(0.8)
        self.recorder.frame(self.data,force=True,snapshot=name)
        self.recorder.grid_snapshot(self.data,name)

    def close(self):
        self.recorder.close()
        if self.viewer:
            self.viewer.close()
