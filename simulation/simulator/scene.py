"""Scene construction from the Lab 2 Panda asset, with physical free blocks."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    return json.loads(Path(path or ROOT / "config/scene.json").read_text())


def grid_xy(config, cell):
    row, col = divmod(cell, 3)
    x, y = config["grid_center"]
    spacing = config["grid_spacing"]
    return [x + (1-row)*spacing, y + (1-col)*spacing]


def staging(config):
    return {f"{color}{i}": [x, sign*config["staging_y"]]
            for color, sign in (("R", 1), ("B", -1))
            for i, x in enumerate(config["staging_x"])}


def build_scene(config, output):
    tree = ET.parse(ROOT / "assets/panda.xml")
    root = tree.getroot()
    root.find("compiler").set("meshdir", str(ROOT / "assets/meshes"))
    option = root.find("option")
    option.set("timestep", str(config["timestep"]))
    option.set("integrator", "implicitfast")
    root.find("default/default/joint").set("damping", "2")
    world = root.find("worldbody")
    ground = world.find("geom[@name='ground']")
    ground.set("pos", "0 0 -0.051")
    ET.SubElement(world, "geom", name="table", type="box", size="0.275 0.504 0.025",
                  pos="0.46 0 -0.025", rgba="0.65 0.58 0.47 1", friction="1 0.01 0.001")
    for cell in range(9):
        x, y = grid_xy(config, cell)
        ET.SubElement(world, "geom", name=f"grid{cell}", type="box", size="0.023 0.023 0.0001",
                      pos=f"{x} {y} 0.00015", rgba="0.3 0.34 0.38 1", contype="0", conaffinity="0")
    half = config["block_half_size"]
    for name, (x, y) in staging(config).items():
        body = ET.SubElement(world, "body", name=name, pos=f"{x} {y} {half+0.0001}")
        ET.SubElement(body, "freejoint", name=f"{name}_joint")
        color = "0.9 0.08 0.04 1" if name.startswith("R") else "0.02 0.4 0.85 1"
        ET.SubElement(body, "geom", name=f"{name}_geom", type="box", size=f"{half} {half} {half}",
                      mass=str(config["block_mass"]), rgba=color, friction="1.5 0.02 0.002",
                      condim="4", solref="0.008 1")
    hand = world.find(".//body[@name='hand']")
    ET.SubElement(hand, "site", name="grasp_center", pos="0 0 0.1034", size="0.002", rgba="0 1 0 0")
    # More closing force than the demonstration controller, still contact-driven.
    grip = root.find("actuator/position")
    grip.set("kp", "1200")
    grip.set("forcerange", "-20 20")
    for finger in ("left_finger", "right_finger"):
        for geom in world.findall(f".//body[@name='{finger}']/geom"):
            geom.set("friction", "1.5 0.02 0.002")
            geom.set("condim", "4")
    ET.SubElement(world, "camera", name="overview", pos="1.4 -1.15 1.25",
                  xyaxes="0.78 0.63 0 -0.42 0.52 0.74")
    cx, cy = config["grid_center"]
    ET.SubElement(world, "camera", name="grid_view", pos=f"{cx} {cy} 0.65",
                  xyaxes="0 -1 0 1 0 0", fovy="32")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output
