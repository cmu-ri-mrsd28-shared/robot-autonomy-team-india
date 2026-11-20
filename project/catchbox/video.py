"""CPU-only mesh rendering of recorded MuJoCo states; no graphics context required.

Uses actual compiled model mesh vertices and geom transforms. The shaded view is
illustrative; collision physics always comes from MuJoCo, not this renderer.
"""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
from .core import Config,sample_throws
from .sim import Simulation
from .experiments import setup,throws_for,rollout

def primitive(kind,size):
    if kind==mujoco.mjtGeom.mjGEOM_BOX:
        v=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])*size
        f=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],
                    [2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]])
        return v,f
    if kind==mujoco.mjtGeom.mjGEOM_CYLINDER:
        n=24;a=np.arange(n)*2*np.pi/n
        v=np.vstack([np.c_[size[0]*np.cos(a),size[0]*np.sin(a),np.full(n,z)] for z in [-size[1],size[1]]]+[np.array([[0,0,-size[1]],[0,0,size[1]]])])
        f=[]
        for i in range(n):
            j=(i+1)%n;f.extend([[i,j,j+n],[i,j+n,i+n],[2*n,j,i],[2*n+1,i+n,j+n]])
        return v,np.array(f)
    # Sphere tessellation; only sphere primitives in these models are visualized.
    v=[];n=16;r=10
    for j in range(r+1):
        for i in range(n):
            a=2*np.pi*i/n;b=np.pi*j/r
            v.append([size[0]*np.sin(b)*np.cos(a),size[0]*np.sin(b)*np.sin(a),size[0]*np.cos(b)])
    f=[]
    for j in range(r):
        for i in range(n):
            a=j*n+i;b=j*n+(i+1)%n;f.extend([[a,b,b+n],[a,b+n,a+n]])
    return np.array(v),np.array(f)

class Renderer:
    def __init__(self,sim):
        self.sim=sim;self.geometry=[]
        m=sim.model
        for g in range(m.ngeom):
            if m.geom_group[g]!=1 or m.geom_rgba[g,3]<.1:continue
            kind=m.geom_type[g]
            if kind==mujoco.mjtGeom.mjGEOM_PLANE:continue
            if kind==mujoco.mjtGeom.mjGEOM_MESH:
                mid=m.geom_dataid[g];va=m.mesh_vertadr[mid];fa=m.mesh_faceadr[mid]
                v=m.mesh_vert[va:va+m.mesh_vertnum[mid]].copy()
                f=m.mesh_face[fa:fa+m.mesh_facenum[mid]].copy()
                # Vertex clustering only for the movie's robot visual mesh;
                # MuJoCo collision geometry and trajectories are untouched.
                _,mapping=np.unique(np.round(v/.003).astype(int),axis=0,return_inverse=True)
                count=np.bincount(mapping)
                v=np.column_stack([np.bincount(mapping,weights=v[:,j])/count for j in range(3)])
                f=mapping[f]
                f=f[(f[:,0]!=f[:,1])&(f[:,0]!=f[:,2])&(f[:,1]!=f[:,2])]
            elif kind in [mujoco.mjtGeom.mjGEOM_BOX,mujoco.mjtGeom.mjGEOM_CYLINDER,mujoco.mjtGeom.mjGEOM_SPHERE]:
                v,f=primitive(kind,m.geom_size[g])
            else:continue
            self.geometry.append((g,v,f,np.array(m.geom_rgba[g,:3])))
        try:self.font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',19)
        except OSError:self.font=ImageFont.load_default()
    def scene(self,size,look,eye,scale):
        im=Image.new('RGB',size,(23,30,42));draw=ImageDraw.Draw(im)
        look=np.asarray(look);eye=np.asarray(eye)
        forward=(look-eye);forward/=np.linalg.norm(forward)
        right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
        up=np.cross(right,forward);basis=np.array([right,up,forward])
        def project(v):
            c=(v-look)@basis.T
            return np.c_[size[0]/2+c[:,0]*scale,size[1]/2-c[:,1]*scale],c[:,2]
        for x in np.arange(-.4,1.61,.2):
            xy,_=project(np.array([[x,-.8,0],[x,.8,0]]));draw.line([tuple(p) for p in xy],fill=(42,51,65))
        for y in np.arange(-.8,.81,.2):
            xy,_=project(np.array([[-.4,y,0],[1.6,y,0]]));draw.line([tuple(p) for p in xy],fill=(42,51,65))
        triangles=[];depth=[];colors=[]
        d=self.sim.data
        for g,v,f,col in self.geometry:
            world=v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
            xy,z=project(world)
            tris=xy[f];n=np.cross(world[f[:,1]]-world[f[:,0]],world[f[:,2]]-world[f[:,0]])
            n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-12)
            shade=.38+.62*np.abs(n@np.array([.3,-.4,.866]))
            triangles.append(tris);depth.append(z[f].mean(axis=1));colors.append(np.clip(shade[:,None]*col*255,0,255).astype('uint8'))
        triangles=np.concatenate(triangles);depth=np.concatenate(depth);colors=np.concatenate(colors)
        visible=(triangles[:,:,0].max(axis=1)>=0)&(triangles[:,:,0].min(axis=1)<=size[0])&(triangles[:,:,1].max(axis=1)>=0)&(triangles[:,:,1].min(axis=1)<=size[1])
        triangles=triangles[visible];depth=depth[visible];colors=colors[visible]
        for i in np.argsort(depth)[::-1]:
            tri=triangles[i]
            draw.polygon([tuple(p) for p in tri],fill=tuple(colors[i]))
        return im
    def frame(self,row,title,outcome):
        s=self.sim;s.data.qpos[:]=row['qpos'];mujoco.mj_forward(s.model,s.data)
        im=Image.new('RGB',(960,600),(15,21,31))
        im.paste(self.scene((640,490),getattr(self,'main_lookat',[.5,0,.4]),[1.5,-2,1.3],getattr(self,'main_scale',430)),(0,60))
        center=s.data.site_xpos[s.site].copy()
        im.paste(self.scene((320,320),center,center+[.4,-.7,.45],1250),(640,120))
        draw=ImageDraw.Draw(im);font=self.font
        draw.text((20,16),title,font=font,fill='white')
        draw.text((655,90),'CONTACT CLOSE-UP',font=font,fill=(100,210,220))
        draw.text((655,465),f"Simulation t = {row['t']:.2f} s",font=font,fill='white')
        draw.text((20,555),f'Outcome: {outcome} | 0.5x playback | MuJoCo state / CPU mesh render',font=font,fill=(235,205,130))
        return im

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('results/videos'));a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    cfg=Config(duration=3.0);clips=[];manifest=[]
    # Fixed seeds, not selected for success. Two throws per tool; paddle shows both controllers.
    for tool in ['box','paddle','gripper']:
        sim=Simulation(cfg) if tool=='box' else setup(tool,cfg)
        throws=sample_throws(42,2,'lob',sim.nominal_opening) if tool=='box' else throws_for(sim,42,2,'lob',tool=='gripper')
        renderer=Renderer(sim)
        for i,throw in enumerate(throws):
            mode='scripted' if tool=='box' else ['stationary','soft'][i] if tool=='paddle' else 'parked'
            if tool=='box':result,rows=sim.rollout(throw,trace=True);rows=rows[::2]
            else:result,rows=rollout(sim,throw,mode=mode,opening=.065,closure_offset=-.1,record=True)
            result=json.loads(json.dumps(result,default=lambda x:x.item()))
            title=f'{tool.upper()}  {i+1}/2  |  {mode}  |  seed 42, throw {i+1}'
            path=a.out/f'{tool}-{i+1}.mp4'
            cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','960x600','-r','25','-i','-','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
            proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
            try:
                for j,row in enumerate(rows):
                    frame=renderer.frame(row,title,result['outcome'])
                    if j==15:frame.save(a.out/f'{tool}-{i+1}.png')
                    proc.stdin.write(frame.tobytes())
            finally:proc.stdin.close()
            if proc.wait()!=0:raise RuntimeError('ffmpeg failed')
            clips.append(path);manifest.append(dict(tool=tool,mode=mode,result=result,video=str(path),seed=42,throw_index=i))
            (a.out/f'{tool}-{i+1}-trace.json').write_text(json.dumps(rows))
            print(f'Wrote {path}: {result["outcome"]}',flush=True)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    listing=a.out/'concat.txt';listing.write_text(''.join(f"file '{p.name}'\n" for p in clips))
    subprocess.run(['ffmpeg','-y','-loglevel','error','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(a.out/'all-six-rollouts.mp4')],check=True)
if __name__=='__main__':main()
