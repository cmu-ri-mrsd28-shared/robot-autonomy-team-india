"""MuJoCo scene and collision queries; block poses are never teleported in execution."""

import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

from .lab2_bridge import LAB2, lab2
from planner import COLORS, INITIAL


def make_model(layout):
    tree = ET.parse(LAB2 / lab2.ROOT_MODEL_XML)
    root = tree.getroot()
    root.find("compiler").set("meshdir", str(LAB2 / "franka_emika_panda" / "assets"))
    root.find("option").set("timestep", "0.002")
    world = root.find("worldbody")
    # Reuse lab2 shelf/table dimensions, keeping the shared lab environment.
    for name, pos, size in lab2.BLOCKS:
        pos = [pos[0], pos[1], pos[2] + layout.table_z - 0.0001]
        body = ET.SubElement(world, "body", name=name, pos=" ".join(map(str, pos)))
        ET.SubElement(
            body,
            "geom",
            name=name + "_geom",
            type="box",
            size=" ".join(map(str, size)),
            rgba="0.25 0.3 0.65 1",
            friction="1.5 0.01 0.001",
        )
    world.find("geom[@name='ground']").set(
        "pos", f"0 0 {min(0.0, layout.table_z - 0.0001)}"
    )
    for i in range(9):
        x, y, _ = layout.xyz(i)
        ET.SubElement(
            world,
            "site",
            name=f"cell_{i}",
            type="box",
            pos=f"{x} {y} {layout.table_z + 0.0002}",
            size=f"{layout.half + 0.003} {layout.half + 0.003} 0.0001",
            rgba="0.6 0.6 0.6 0.5",
        )
    for i, slot in enumerate(INITIAL):
        body = ET.SubElement(
            world, "body", name=f"block_{i}", pos=" ".join(map(str, layout.xyz(slot)))
        )
        ET.SubElement(body, "freejoint", name=f"block_joint_{i}")
        ET.SubElement(
            body,
            "geom",
            name=f"block_geom_{i}",
            type="box",
            size=" ".join([str(layout.half)] * 3),
            mass="0.06",
            friction="2 0.01 0.001",
            condim="4",
            rgba="0.9 0.1 0.05 1" if COLORS[i] == "R" else "0.05 0.4 0.9 1",
        )
    # Stable tracking servos use the same arm/joint model and min-jerk trajectories
    # as lab2; physical execution still sends those waypoints through FrankaPy.
    actuators = root.find("actuator")
    for old in list(actuators)[:7]:
        actuators.remove(old)
    for i in range(7):
        ET.SubElement(
            actuators,
            "position",
            name=f"arm_servo_{i}",
            joint=f"joint{i + 1}",
            kp="1200",
            kv="70",
            forcelimited="true",
            forcerange="-87 87",
        )
    # Existing finger actuator remains first. Look up all actuators by name.
    finger = actuators.find("position")
    finger.set("kp", "2000")
    model = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return model


class Scene:
    def __init__(self, layout):
        self.layout = layout
        self.model = make_model(layout)
        self.data = mj.MjData(self.model)
        self.arm_act = np.array(
            [self.model.actuator(f"arm_servo_{i}").id for i in range(7)]
        )
        self.grip_act = self.model.actuator("actuator8").id
        self.block_bodies = [self.model.body(f"block_{i}").id for i in range(8)]
        self.block_q = [
            self.model.jnt_qposadr[self.model.joint(f"block_joint_{i}").id]
            for i in range(8)
        ]
        self.robot_bodies = {
            i
            for i in range(1, self.model.nbody)
            if self.model.body(i).name not in {f"block_{k}" for k in range(8)}
            and self.model.body(i).name not in {b[0] for b in lab2.BLOCKS}
        }
        self.hand_bodies = {
            self.model.body(n).id for n in ("hand", "left_finger", "right_finger")
        }
        self.table = self.model.body("TablePlane").id
        self.set_state(INITIAL, lab2.Q_HOME)
        # Adjacent link meshes overlap by design. Record home self-contacts as
        # permanent model exclusions; other nonadjacent self contacts are rejected.
        self.home_pairs = self.self_contacts()

    def set_state(self, state, q, grip=0.04, carried=None):
        d, m = self.data, self.model
        d.qpos[:7] = q
        d.qpos[7:9] = grip
        d.qvel[:] = 0
        for i, slot in enumerate(state):
            adr = self.block_q[i]
            d.qpos[adr : adr + 3] = self.layout.xyz(slot)
            d.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
        mj.mj_forward(m, d)
        if carried is not None:
            block, relative = carried
            hand = m.body("hand").id
            T = np.eye(4)
            T[:3, :3] = d.xmat[hand].reshape(3, 3)
            T[:3, 3] = d.xpos[hand]
            pose = T @ relative
            adr = self.block_q[block]
            d.qpos[adr : adr + 3] = pose[:3, 3]
            quat = np.empty(4)
            mj.mju_mat2Quat(quat, pose[:3, :3].ravel())
            d.qpos[adr + 3 : adr + 7] = quat
            mj.mj_forward(m, d)

    def self_contacts(self):
        pairs = set()
        for c in self.data.contact:
            a, b = (int(self.model.geom_bodyid[g]) for g in (c.geom1, c.geom2))
            if a in self.robot_bodies and b in self.robot_bodies:
                pairs.add(tuple(sorted((a, b))))
        return pairs

    def valid(self, q, state, grip=0.04, touch=None, carried=None):
        if not lab2.in_limits(q):
            return False
        self.set_state(state, q, grip, carried)
        m = self.model
        target = self.block_bodies[touch] if touch is not None else None
        held = self.block_bodies[carried[0]] if carried else None
        for c in self.data.contact:
            if c.dist >= -0.0005:
                continue
            a, b = (int(m.geom_bodyid[g]) for g in (c.geom1, c.geom2))
            pair = tuple(sorted((a, b)))
            if a in self.robot_bodies and b in self.robot_bodies:
                # Consecutive bodies and the two fingers are coupled rigidly;
                # their mesh overlap is part of the provided robot model.
                if (
                    m.body_parentid[a] == b
                    or m.body_parentid[b] == a
                    or (a in self.hand_bodies and b in self.hand_bodies)
                ):
                    continue
                if pair not in self.home_pairs:
                    return False
            elif a in self.robot_bodies or b in self.robot_bodies:
                robot, other = (a, b) if a in self.robot_bodies else (b, a)
                if other == 0 and robot == m.body("link0").id:
                    continue
                if other in (target, held) and robot in self.hand_bodies:
                    continue
                return False
            elif held in (a, b):
                # Contact at the departure/arrival support is permitted; overlap
                # with every other block and shelf is checked throughout transit.
                other = b if a == held else a
                if other not in (0, self.table) or c.dist < -0.0015:
                    return False
        return True

    def relative_block(self, block):
        d, m = self.data, self.model
        hand = m.body("hand").id
        H = np.eye(4)
        H[:3, :3] = d.xmat[hand].reshape(3, 3)
        H[:3, 3] = d.xpos[hand]
        B = np.eye(4)
        B[:3, :3] = d.xmat[self.block_bodies[block]].reshape(3, 3)
        B[:3, 3] = d.xpos[self.block_bodies[block]]
        return np.linalg.inv(H) @ B
