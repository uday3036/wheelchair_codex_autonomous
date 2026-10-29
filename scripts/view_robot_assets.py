"""Inspect generated design assets without running uncontrolled robot physics.

macOS: .venv/bin/mjpython scripts/view_robot_assets.py
Static image: .venv/bin/python -B scripts/view_robot_assets.py --render /tmp/robot.png
"""
import argparse
import json
from pathlib import Path
import time

import mujoco

ROOT = Path(__file__).resolve().parents[1]


def load_design(urdf=False, scene=False):
    path = ROOT/("assets/urdf/wheelchair.urdf" if urdf else
                 "assets/mjcf/wheelchair_scene.xml" if scene else "assets/mjcf/wheelchair_robot.xml")
    model = mujoco.MjModel.from_xml_path(str(path))
    state = mujoco.MjData(model)
    if urdf:
        values = json.loads((ROOT/"assets/urdf/folded_joint_positions.json").read_text())
        for name, value in values.items():
            state.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    else:
        mujoco.mj_resetDataKeyframe(model,state,model.key("folded").id)
    mujoco.mj_forward(model,state)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = state.xpos[model.body("base_link").id]+[0.,0.,.2]
    camera.distance, camera.azimuth, camera.elevation = 2.4, 130., -20.
    option = mujoco.MjvOption()
    option.geomgroup[4:6] = 1
    option.geomgroup[2] = 0
    if urdf:
        option.geomgroup[0] = 0  # URDF collision skins; show its visual group.
    return model, state, camera, option


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--urdf",action="store_true")
    group.add_argument("--scene",action="store_true")
    parser.add_argument("--render",type=Path)
    args = parser.parse_args()
    model,state,camera,option = load_design(args.urdf,args.scene)
    if args.render:
        import imageio.v2 as imageio
        args.render.parent.mkdir(parents=True,exist_ok=True)
        with mujoco.Renderer(model,height=480,width=640) as renderer:
            renderer.update_scene(state,camera=camera,scene_option=option)
            imageio.imwrite(args.render,renderer.render())
        print(f"Saved {args.render}")
    else:
        from mujoco import viewer as mujoco_viewer
        with mujoco_viewer.launch_passive(model,state) as viewer:
            viewer.cam.lookat[:]=camera.lookat
            viewer.cam.distance,viewer.cam.azimuth,viewer.cam.elevation = camera.distance,camera.azimuth,camera.elevation
            viewer.opt.geomgroup[:]=option.geomgroup
            print("Design inspection: physics is paused. Drag to orbit, scroll to zoom; close the window to exit.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(.02)


if __name__ == "__main__":
    main()
