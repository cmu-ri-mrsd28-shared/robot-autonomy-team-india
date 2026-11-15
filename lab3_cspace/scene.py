"""MuJoCo model and collision queries; imported only for planning/simulation."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco as mj
from lab3_cspace.kinematics import Panda


def matrix_quaternion(r):
    result=np.empty(4)
    mj.mju_mat2Quat(result,np.asarray(r).reshape(9))
    return result


class Scene:
    def __init__(self,cfg,layout):
        self.cfg,self.layout=cfg,layout
        # Reuse robot meshes only, never the legacy planner or controller.
        source=Path(cfg.get('model_xml',Path(__file__).resolve().parents[1]/'simulation/assets/panda.xml'))
        root=ET.parse(source).getroot()
        root.find('compiler').set('meshdir',str(source.parent/'meshes'))
        option=root.find('option'); option.set('timestep','.002'); option.set('integrator','implicitfast')
        option.set('iterations','80')
        # Suppress regularization-induced static-friction creep during long
        # rest-to-rest joint segments; blocks remain free contact bodies.
        option.set('noslip_iterations','3')
        world=root.find('worldbody')
        ET.SubElement(world,'geom',name='table',type='box',pos=f'.46 0 {cfg["table_z"]-.025}',size='.275 .504 .025',rgba='.25 .3 .55 1')
        for obstacle in cfg.get('obstacles',[]):
            ET.SubElement(world,'geom',name=obstacle['name'],type='box',
                          pos=' '.join(map(str,obstacle['center'])),
                          size=' '.join(map(str,obstacle['half_size'])),rgba='.4 .4 .45 1')
        world.find("geom[@name='ground']").set('pos',f'0 0 {min(-.03,cfg["table_z"]-.05)}')
        hand=root.find(".//body[@name='hand']")
        tool=cfg['flange_to_tcp']
        # MJCF hand already includes a -45 degree rotation from the flange.
        if not np.isclose(tool['yaw_deg'],-45):
            raise ValueError('This simulation hand model expects flange_to_tcp.yaw_deg=-45')
        ET.SubElement(hand,'site',name='tcp',pos=' '.join(map(str,tool['translation'])),size='.004',rgba='1 0 0 1')
        for i in range(9):
            xyz=layout.xyz(i); xyz[2]=cfg['table_z']+.0002
            ET.SubElement(world,'site',name=f'cell_{i}',type='box',pos=' '.join(map(str,xyz)),
                          size=f'{cfg["cube_size"]/2} {cfg["cube_size"]/2} .0001',rgba='.65 .65 .65 .5')
        for block in layout.blocks:
            xyz=layout.xyz(block); xyz[2]=cfg['table_z']+cfg['cube_size']/2
            body=ET.SubElement(world,'body',name='cube_'+block,pos=' '.join(map(str,xyz)))
            ET.SubElement(body,'freejoint',name='free_'+block)
            ET.SubElement(body,'geom',name='geom_'+block,type='box',size=' '.join([str(cfg['cube_size']/2)]*3),
                          mass=str(cfg['cube_mass']),friction='2 .01 .001',condim='4',
                          rgba='0.9 .08 .05 1' if block[0]=='R' else '.05 .25 .9 1')
        acts=root.find('actuator')
        for element in list(acts)[:7]: acts.remove(element)
        for i in range(7):
            ET.SubElement(acts,'position',name=f'arm_{i}',joint=f'joint{i+1}',kp='1200',kv='70',
                          forcelimited='true',forcerange='-87 87')
        finger=acts.find("position[@name='actuator8']")
        finger.set('kp','1600'); finger.set('ctrlrange','0 .04')
        # Keep the actual finger/contact geometry; no welds or block snapping.
        for default in root.findall(".//default[@class='collision']/geom"):
            default.set('friction','2 .01 .001')
        self.model=mj.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
        self.data=mj.MjData(self.model); self.query=mj.MjData(self.model)
        self.arm_act=np.array([self.model.actuator(f'arm_{i}').id for i in range(7)])
        self.finger_act=self.model.actuator('actuator8').id
        self.tcp=self.model.site('tcp').id
        self.block_bodies={b:self.model.body('cube_'+b).id for b in layout.blocks}
        self.block_qpos={b:self.model.jnt_qposadr[self.model.joint('free_'+b).id] for b in layout.blocks}
        self.fingers={self.model.body(n).id for n in ('left_finger','right_finger')}
        self.moving=set()
        root_id=self.model.body('link1').id
        for body in range(self.model.nbody):
            ancestor=body
            while ancestor:
                if ancestor==root_id: self.moving.add(body); break
                ancestor=int(self.model.body_parentid[ancestor])
        self.data.qpos[:7]=cfg['home_q']; self.data.qpos[7:9]=cfg['gripper']['open_width']/2
        self.data.ctrl[self.arm_act]=self.data.qpos[:7]
        self.data.ctrl[self.finger_act]=cfg['gripper']['open_width']/2
        mj.mj_forward(self.model,self.data)
        expected=Panda(cfg).fk(self.data.qpos[:7])
        if (np.linalg.norm(expected[:3,3]-self.data.site_xpos[self.tcp])>1e-6
                or np.linalg.norm(expected[:3,:3]-self.data.site_xmat[self.tcp].reshape(3,3))>1e-6):
            raise RuntimeError('FK and MuJoCo tool frames disagree')
        self.last_collision=None

    def valid(self,q,locations,width,held=None,held_rotation=None,touch=None):
        from lab3_cspace.kinematics import QMIN,QMAX
        q=np.asarray(q)
        if q.shape!=(7,) or not np.isfinite(q).all() or np.any(q<QMIN) or np.any(q>QMAX): return False
        d=self.query
        d.qpos[:7]=q; d.qpos[7:9]=width/2
        mj.mj_forward(self.model,d)
        for block,location in locations.items():
            a=self.block_qpos[block]
            if block==held:
                d.qpos[a:a+3]=d.site_xpos[self.tcp]
                r=d.site_xmat[self.tcp].reshape(3,3)@held_rotation
                d.qpos[a+3:a+7]=matrix_quaternion(r)
            else:
                xyz=self.layout.xyz(location).copy(); xyz[2]=self.cfg['table_z']+self.cfg['cube_size']/2
                d.qpos[a:a+3]=xyz; d.qpos[a+3:a+7]=[1,0,0,0]
        mj.mj_forward(self.model,d)
        held_body=self.block_bodies.get(held,-1)
        touch_body=self.block_bodies.get(touch,-1)
        for contact in d.contact:
            a,b=map(int,self.model.geom_bodyid[[contact.geom1,contact.geom2]])
            if not (a in self.moving or b in self.moving or a==held_body or b==held_body): continue
            if ((a in self.fingers and b in (held_body,touch_body)) or
                    (b in self.fingers and a in (held_body,touch_body))): continue
            # Object/support contact is expected at pick/place endpoints.
            if (a==held_body or b==held_body) and contact.dist>=-.001:
                other=int(contact.geom2 if a==held_body else contact.geom1)
                if mj.mj_id2name(self.model,mj.mjtObj.mjOBJ_GEOM,other)=='table': continue
            if contact.dist < self.cfg['motion']['clearance']:
                self.last_collision=(mj.mj_id2name(self.model,mj.mjtObj.mjOBJ_GEOM,int(contact.geom1)),
                                     mj.mj_id2name(self.model,mj.mjtObj.mjOBJ_GEOM,int(contact.geom2)),float(contact.dist))
                return False
        return True
