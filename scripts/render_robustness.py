"""Render actual reverse/payload recordings and their quantitative comparison."""
from pathlib import Path
import sys
import mujoco
import numpy as np
import imageio.v2 as imageio

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.grasp_scene import bottle_scene
from src.run_experiments import plt


def render(model,poses,filename,mocaps=None,closeup=False):
    state=mujoco.MjData(model)
    option=mujoco.MjvOption();option.geomgroup[2]=0;option.geomgroup[3:6]=1
    camera=mujoco.MjvCamera();camera.lookat[:]=[-.25,0,.65]
    camera.distance,camera.azimuth,camera.elevation=3.2,125.,-20.
    with mujoco.Renderer(model,height=480,width=640) as renderer:
        with imageio.get_writer(filename,mode='I',duration=100,loop=0) as writer:
            for i,q in enumerate(poses):
                state.qpos[:]=q
                if mocaps is not None:state.mocap_pos[:]=mocaps[i]
                mujoco.mj_forward(model,state)
                renderer.update_scene(state,camera='grasp_view' if closeup else camera,scene_option=option)
                writer.append_data(renderer.render().copy())


def main():
    output=ROOT/'results/robustness_validation'
    for label in ('prealigned','sideways'):
        model=bottle_scene(mobile=True,auxiliary_clamp=True,opening_fixture=True,bottle_present=False)
        with np.load(output/f'reverse_{label}_poses.npz') as z:
            render(model,z['qpos'],output/f'reverse_{label}.gif',z['mocap_pos'])
    for label,friction in (('heavy_original',1.),('heavy_grip_material',1.5)):
        model=bottle_scene(mass=.6,friction=friction)
        with np.load(output/f'payload_{label}_poses.npz') as z:
            render(model,z['qpos'],output/f'payload_{label}.gif',closeup=True)
    fig,axes=plt.subplots(2,1,figsize=(9,7))
    for label,title in (('nominal','0.3 kg, friction 1.0'),('heavy_original','0.6 kg, friction 1.0'),('heavy_grip_material','0.6 kg, friction 1.5')):
        data=np.genfromtxt(output/f'payload_{label}.csv',delimiter=',',names=True)
        axes[0].plot(data['time'],1000*data['relative_slip_m'],label=title)
        axes[1].plot(data['time'],1000*(data['bottle_z']-.88),label=title)
    axes[0].axhline(10,ls='--',color='black',label='10 mm hold drift gate')
    axes[1].axhline(60,ls='--',color='black',label='60 mm lift gate')
    axes[0].set_ylabel('Relative tool/bottle drift [mm]');axes[1].set_ylabel('Lift [mm]');axes[1].set_xlabel('Time [s]')
    for ax in axes:ax.legend();ax.grid(True,alpha=.3)
    fig.tight_layout();fig.savefig(output/'payload_material_comparison.png',dpi=140);plt.close(fig)
    print('Saved four actual-physics animations and payload comparison graph:',output)


if __name__=='__main__':main()
