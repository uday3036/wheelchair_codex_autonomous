"""Render the reference and simpler visual design using actual MuJoCo geometry."""
import argparse
from pathlib import Path
import sys
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.grasp_scene import bottle_scene
from src.model import initialize
from src.stowage import MAIN_FOLD,AUX_FOLD
from src.run_experiments import plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/design_validation/design_comparison.png')
    p.add_argument('--azimuth',type=float,default=130.,help='Inspection camera angle in degrees')
    args=p.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    images=[]
    for style,cutaway in (('reference',False),('simple',False),('simple',True)):
        m=bottle_scene(mobile=True,auxiliary_clamp=True,opening_fixture=True,design_style=style)
        s,r=initialize(m)
        s.qpos[r['arm_p']],s.qpos[r['sec_p']]=MAIN_FOLD,AUX_FOLD
        for prefix in ('grasp','aux'):
            for side in ('left','right'):s.qpos[m.jnt_qposadr[m.joint(f'{prefix}_{side}_slide').id]]=.03
        mujoco.mj_forward(m,s)
        camera=mujoco.MjvCamera();camera.lookat[:]=[0,0,.65]
        camera.distance,camera.azimuth,camera.elevation=2.4,args.azimuth,-18.
        option=mujoco.MjvOption();option.geomgroup[4]=0;option.geomgroup[5]=0 if cutaway else 1
        option.geomgroup[2]=0  # Passenger envelopes are diagnostics, not hardware.
        # Hide the task environment for this robot-only design comparison.
        m.geom_rgba[[m.geom(n).id for n in ('bottle_collision','bottle_neck','cap_collision','recipient_pad')],3]=0
        for g in range(m.ngeom):
            if (m.geom(g).name or '').startswith('cap_rib_'):m.geom_rgba[g,3]=0
        with mujoco.Renderer(m,height=480,width=640) as renderer:
            renderer.update_scene(s,camera=camera,scene_option=option);images.append(renderer.render().copy())
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,im,title in zip(axes,images,('Reference design','Simpler shells and palette','Simpler design: case cutaway')):
        ax.imshow(im);ax.set_title(title);ax.axis('off')
    fig.tight_layout();fig.savefig(args.output,dpi=150);plt.close(fig)
    print(args.output)


if __name__=='__main__':main()
