"""Independent geometry cases for the NumPy collision backend."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from lab3_cspace.collision import overlaps,convex_collision,box_vertices,CollisionScene
from lab3_cspace.kinematics import load_config,task_config,rz
from lab3_cspace.task import Planner


def test_gjk_matches_box_sat_for_random_rotated_boxes():
    rng=np.random.default_rng(7)
    for _ in range(150):
        a,b=rng.uniform(-1,1,(2,3)); ha,hb=rng.uniform(.05,.6,(2,3))
        ra,rb=[np.linalg.qr(rng.normal(size=(3,3)))[0] for _ in range(2)]
        assert convex_collision(box_vertices(a,ra,ha),box_vertices(b,rb,hb))==bool(overlaps(a,ra,ha,b,rb,hb))


def test_contact_clearance_containment_and_flat_convex_shapes():
    a=box_vertices(np.zeros(3),np.eye(3),np.ones(3))
    inside=box_vertices(np.zeros(3),rz(.4),np.full(3,.2))
    assert convex_collision(a,inside)
    assert convex_collision(a,a+[2,0,0])  # face contact
    assert not convex_collision(a,a+[2.002,0,0],.001)
    assert convex_collision(a,a+[2.0005,0,0],.001)
    # Overlapping bounding boxes do not imply intersecting convex shapes.
    triangle=np.array([[0.,0,0],[1,0,0],[0,1,0]])
    assert not convex_collision(triangle,triangle+[.75,.75,0])
    assert convex_collision(triangle,triangle+[.1,.1,0])


@pytest.fixture
def cfg(): return load_config(Path(__file__).resolve().parents[1]/'simulation.json')


def test_home_is_free_but_added_wrist_obstacle_is_rejected(cfg):
    layout=Planner(task_config(cfg)); scene=CollisionScene(cfg,layout)
    assert scene.valid(cfg['home_q'],layout.initial_locations,.08)
    c,_,_=scene.robot_boxes(cfg['home_q'],.08)
    scene.environment.append(('test_obstacle',c[scene.names.index('link7:8')],np.full(3,.03)))
    assert not scene.valid(cfg['home_q'],layout.initial_locations,.08)
    assert scene.last_collision[1]=='test_obstacle'


def test_only_selected_cube_may_touch_fingers(cfg):
    layout=Planner(task_config(cfg)); scene=CollisionScene(cfg,layout)
    q=scene.robot.home; tcp=scene.robot.fk(q)
    # Place a cube between closed fingers at the home tool pose. Remove the
    # environment here to isolate the contact exception from table placement.
    cfg['table_z']=tcp[2,3]-cfg['cube_size']/2
    layout.config['boxes']['B0'][:2]=tcp[:2,3].tolist()
    scene.environment=[]
    assert not scene.valid(q,{'B0':'waiting:B0'},.041)
    assert scene.valid(q,{'B0':'waiting:B0'},.041,touch='B0')
    assert scene.valid(q,{'B0':'held'},.041,held='B0',held_rotation=tcp[:3,:3].T)


def test_held_cube_checks_support_penetration_and_other_blocks(cfg):
    layout=Planner(task_config(cfg)); scene=CollisionScene(cfg,layout)
    q=scene.robot.home; tcp=scene.robot.fk(q)
    # A tiny obstacle at the cube centre fits between the jaws and only hits
    # the held cube, so this checks carried geometry rather than arm geometry.
    scene.environment=[('held_only_obstacle',tcp[:3,3],np.full(3,.002))]
    assert scene.valid(q,{},.041)
    assert not scene.valid(q,{'B0':'held'},.041,held='B0',held_rotation=tcp[:3,:3].T)
    assert scene.last_collision==('B0','held_only_obstacle')


def test_baked_geometry_matches_source_vertices():
    from lab3_cspace.build_collision_boxes import boxes
    root=Path(__file__).resolve().parents[2]
    expected=boxes(root/'simulation/assets/panda.xml')
    actual=json.loads((root/'lab3_cspace/panda_boxes.json').read_text())
    assert actual==expected


def test_invalid_obstacle_cannot_be_treated_as_free_space(cfg):
    cfg['obstacles']=[{'name':'bad','center':[float('nan'),0,0],'half_size':[.1,.1,.1]}]
    with pytest.raises(ValueError,match='Invalid obstacle geometry'):
        CollisionScene(cfg,Planner(task_config(cfg)))


def test_folded_arm_self_collision_is_rejected(cfg):
    layout=Planner(task_config(cfg)); scene=CollisionScene(cfg,layout)
    q=[.4455173168,.2965396469,-.3083498266,-3.0518,-.1831600957,.8262801306,.8361592293]
    assert not scene.valid(q,layout.initial_locations,.08)
    assert scene.last_collision==('link1:1','link6:17')
