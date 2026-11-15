"""MuJoCo model only: robot assets, table, grid and physical cubes."""

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco as mj


def find_model():
    for parent in Path(__file__).resolve().parents:
        for folder in (parent / "lab2", parent / "labs" / "lab2"):
            model = folder / "franka_emika_panda" / "panda_torque_table.xml"
            if model.is_file():
                return model
    raise FileNotFoundError("Simulation needs the existing lab2 Panda model and assets")


def make_model(layout):
    model_path = find_model()
    tree = ET.parse(model_path)
    root = tree.getroot()
    root.find("compiler").set("meshdir", str(model_path.parent / "assets"))
    root.find("option").set("timestep", "0.002")
    world = root.find("worldbody")
    # Reuse lab2 shelf/table dimensions, keeping the shared lab environment.
    for name, pos, size in [
        ("TablePlane", [0.46, 0.0, -0.005], [0.275, 0.504, 0.0051])
    ]:
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
    for i, slot in enumerate(layout.initial_slots):
        position = layout.xyz(slot).copy()
        position[2] = layout.table_z + layout.half
        body = ET.SubElement(
            world, "body", name=f"block_{i}", pos=" ".join(map(str, position))
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
            rgba="0.9 0.1 0.05 1" if layout.blocks[i][0] == "R" else "0.05 0.4 0.9 1",
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
    # Make reported simulated tool Z match the taught hardware grasp Z.
    # Physical finger-pad center is 0.103 m from the hand origin.
    offset = 0.103 - (layout.config["grasp_height"] - layout.table_z - layout.half)
    ET.SubElement(
        root.find(".//body[@name='hand']"),
        "site",
        name="tool",
        pos=f"0 0 {offset}",
        size="0.004",
        rgba="1 0 0 1",
    )
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
        self.site = self.model.site("tool").id
        self.data.qpos[:7] = [0, -0.5, 0, -2, 0, 1.5, 0.8]
        self.data.qpos[7:9] = layout.config["gripper"]["opening"] / 2
        self.data.ctrl[self.arm_act] = self.data.qpos[:7]
        self.data.ctrl[self.grip_act] = layout.config["gripper"]["opening"] / 2
        mj.mj_forward(self.model, self.data)
