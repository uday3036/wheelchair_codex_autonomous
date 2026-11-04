"""Render measured obstacle-navigation states and task stages for review."""
import argparse
from pathlib import Path
import sys
import numpy as np
import mujoco
import imageio.v2 as imageio

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.replay_autonomous import Recording
from src.run_experiments import plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT/'results/obstacle_validation')
    args=parser.parse_args()
    recording=Recording(args.directory)
    nav_end=recording.metrics['navigation']['duration_s']
    ids=[0,int(np.argmin(abs(recording.qpos[:,0]+3.7))),
         int(np.argmin(abs(recording.times-nav_end))),len(recording.times)-1]
    captions=['Bottle detected; arms stored','Measured detour around obstacles',
              'Docking gates passed','Final recorded state']
    option=mujoco.MjvOption();option.geomgroup[2]=0;option.geomgroup[4:6]=1
    snapshots=[]
    with mujoco.Renderer(recording.model,height=480,width=640) as renderer:
        for index in ids:
            recording.apply(recording.relative_times[index])
            renderer.update_scene(recording.state,camera='autonomous_view',scene_option=option)
            snapshots.append(renderer.render().copy())
        # Preserve original measured navigation frames and their timing.
        with imageio.get_writer(args.directory/'obstacle_navigation.gif',mode='I',duration=100,loop=0) as writer:
            for i,t in enumerate(recording.times):
                if t>nav_end:break
                recording.apply(recording.relative_times[i])
                renderer.update_scene(recording.state,camera='autonomous_view',scene_option=option)
                writer.append_data(renderer.render().copy())
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for ax,pixels,title in zip(axes.flat,snapshots,captions):
        ax.imshow(pixels);ax.set_title(title);ax.axis('off')
    fig.tight_layout();fig.savefig(args.directory/'obstacle_stages.png',dpi=140);plt.close(fig)
    print(args.directory/'obstacle_stages.png')


if __name__=='__main__':main()
