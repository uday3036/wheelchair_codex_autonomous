"""Interactive forward kinematics / inverse kinematics / Jacobian explorer (macOS: use mjpython).

Run:  .venv/bin/mjpython scripts/kinematics_explorer.py

Two ways to move the arm; the matrices in the terminal update live.
  FK  Open the right-hand panel (Shift+Tab, or the "Joint" section) and drag a joint slider.
      The terminal shows T(q) (forward kinematics) and J(q) (the Jacobian).
  IK  Double-click the red target frame, then Ctrl + right-drag to move it and Ctrl + left-drag to
      rotate it. The project's own NumPy damped-least-squares IK drives the arm to the target.
The white arrows are the tool frame; the green-yellow ellipsoid is the velocity ellipsoid of the
linear Jacobian (its axes are the singular values): it flattens at a singularity. Coloured rods
are the joint axes (the columns of J). Close the window to stop.

No physics runs: this is pure kinematics (gravity and torques are ignored on purpose).
"""
import argparse
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import realistic_arm as ra
from src.manipulation import SerialArm, rotation_vector

HOME_Q = np.array([.3, .5, 1.2, .4, .8, -.3])  # a comfortable, non-singular start
ROW_NAMES = ("vx", "vy", "vz", "wx", "wy", "wz")
np.set_printoptions(precision=3, suppress=True, linewidth=140)


def build_model():
    """The buildable 6-DOF arm plus a mocap target frame you can drag in the viewer."""
    root = ET.fromstring(ra.build_xml(0.))
    world = root.find("worldbody")
    target = ET.SubElement(world, "body", name="ik_target", mocap="true", pos="0.4 0 0.4")
    for name, pos, size, rgba in (("x", ".03 0 0", ".03 .004 .004", "1 0 0 1"), ("y", "0 .03 0", ".004 .03 .004", "0 1 0 1"),
                                  ("z", "0 0 .03", ".004 .004 .03", "0 0 1 1")):
        ET.SubElement(target, "geom", name=f"target_{name}", type="box", pos=pos, size=size, rgba=rgba,
                      contype="0", conaffinity="0")
    ET.SubElement(target, "geom", name="target_ball", type="sphere", size=".012", rgba="1 .2 .2 .7",
                  contype="0", conaffinity="0")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def pose_matrix(data, model):
    T = np.eye(4)
    T[:3, 3] = data.mocap_pos[0]
    mujoco.mju_quat2Mat(R := np.zeros(9), data.mocap_quat[0])
    T[:3, :3] = R.reshape(3, 3)
    return T


def set_mocap(data, T):
    data.mocap_pos[0] = T[:3, 3]
    mujoco.mju_mat2Quat(q := np.zeros(4), T[:3, :3].reshape(-1))
    data.mocap_quat[0] = q


class Explorer:
    """Mode logic and numbers, independent of any window (so tests can drive it)."""

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.arm = SerialArm(model, ra.JOINTS, "grasp_center")
        self.site = model.site("grasp_center").id
        self.mode, self.ik = "FK", None
        data.qpos[:6] = HOME_Q
        mujoco.mj_forward(model, data)
        self.last_q = data.qpos[:6].copy()
        self._sync_target()

    def _sync_target(self):
        set_mocap(self.data, self.arm.forward(self.data.qpos[:6]))
        self.last_target = np.r_[self.data.mocap_pos[0], self.data.mocap_quat[0]]

    def update(self):
        """Detect which input the user moved (a joint slider or the target) and respond."""
        d = self.data
        if np.max(np.abs(d.qpos[:6]-self.last_q)) > 1e-9:       # a joint slider moved: forward kinematics
            self.mode, self.ik = "FK", None
            self._sync_target()
        else:
            signature = np.r_[d.mocap_pos[0], d.mocap_quat[0]]
            if np.max(np.abs(signature-self.last_target)) > 1e-9:   # the target moved: inverse kinematics
                self.mode = "IK"
                target = pose_matrix(d, self.model)
                q, info = self.arm.inverse(target, d.qpos[:6].copy(), iterations=60)
                d.qpos[:6] = q
                self.ik = dict(info, target=target)
                self.last_target = signature
        mujoco.mj_forward(self.model, d)
        self.last_q = d.qpos[:6].copy()

    def numbers(self):
        """Everything shown to the user, from the NumPy FK/Jacobian, checked against the engine."""
        q = self.data.qpos[:6]
        T, J = self.arm.forward(q, True)
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jp, jr, self.site)
        U, S, Vt = np.linalg.svd(J)
        Uv, Sv, _ = np.linalg.svd(J[:3])  # linear part: the velocity ellipsoid
        return dict(q=q.copy(), T=T, J=J, singular_values=S, sigma_min=S[-1], condition=S[0]/max(S[-1], 1e-12),
                    manipulability=float(np.prod(S)), rank=int((S > 1e-6).sum()),
                    fk_error=float(np.abs(T[:3, 3]-self.data.site_xpos[self.site]).max()),
                    jacobian_error=float(np.abs(J-np.vstack((jp, jr))[:, :6]).max()),
                    ellipsoid=(Uv, Sv), limits=self.arm.limits)

    def text(self):
        n = self.numbers()
        lines = [f"MODE: {self.mode}   (drag a joint slider -> FK, drag the red target -> IK)", ""]
        lines.append("joint angles q [deg]: " + "  ".join(f"{name}={np.degrees(v):7.1f}" for name, v in zip(ra.JOINTS, n["q"])))
        margin = np.minimum(n["q"]-n["limits"][:, 0], n["limits"][:, 1]-n["q"])
        lines.append("distance to nearest limit [deg]: " + "  ".join(f"{np.degrees(m):6.1f}" for m in margin))
        lines += ["", "FORWARD KINEMATICS   T(q) = base -> tool frame (NumPy, product of the joint transforms)",
                  *["  "+row for row in np.array2string(n["T"]).splitlines()],
                  f"  position [m] = {np.round(n['T'][:3, 3], 4)}   (engine disagreement {n['fk_error']:.1e} m)"]
        lines += ["", "JACOBIAN   J(q): rows = tool velocity [vx vy vz wx wy wz], columns = joints  (v = J dq/dt)"]
        header = "         " + "".join(f"{name:>9s}" for name in ra.JOINTS)
        lines.append(header)
        for row, name in zip(n["J"], ROW_NAMES):
            lines.append(f"   {name:>3s}  " + "".join(f"{v:9.3f}" for v in row))
        lines.append(f"  (engine disagreement {n['jacobian_error']:.1e})")
        flag = "   <<< SINGULAR: a tool direction is lost" if n["sigma_min"] < .02 else (
            "   << near a singularity" if n["sigma_min"] < .06 else "")
        lines += ["", f"SINGULARITY   singular values {n['singular_values']}", f"  sigma_min {n['sigma_min']:.4f}   condition {n['condition']:.0f}   "
                  f"manipulability {n['manipulability']:.5f}   rank {n['rank']}/6{flag}"]
        if self.ik is not None:
            target, info = self.ik["target"], self.ik
            error = np.r_[target[:3, 3]-n["T"][:3, 3], rotation_vector(target[:3, :3]@n["T"][:3, :3].T)]
            lines += ["", "INVERSE KINEMATICS   solve T(q) = target by damped least squares (NumPy)",
                      f"  target position [m] {np.round(target[:3, 3], 4)}",
                      f"  solved: {'yes' if info['success'] else 'NO (target out of reach or singular)'}   iterations {info['iterations']}",
                      f"  position error {1000*np.linalg.norm(error[:3]):.2f} mm   orientation error {np.degrees(np.linalg.norm(error[3:])):.2f} deg"]
        return "\n".join(lines)


def draw_aids(viewer, ex):
    """Tool frame arrows, joint-axis rods and the linear-Jacobian velocity ellipsoid."""
    scn, n = viewer.user_scn, ex.numbers()
    T, geoms = n["T"], []

    def connector(a, b, rgba, width=.004):
        geom = scn.geoms[len(geoms)]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(rgba, np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
        geoms.append(geom)
    for axis, rgba in zip(T[:3, :3].T, ((1, 1, 1, 1), (.7, .7, .7, 1), (.45, .45, .45, 1))):  # tool x (approach), y, z
        connector(T[:3, 3], T[:3, 3]+.08*axis, rgba, .005)
    colours = ((1, .3, .3, .8), (1, .6, .2, .8), (1, 1, .2, .8), (.4, 1, .4, .8), (.3, .8, 1, .8), (.7, .5, 1, .8))
    for j, name in enumerate(ra.JOINTS):
        jid = ex.model.joint(name).id
        centre, axis = ex.data.xanchor[jid], ex.data.xaxis[jid]
        connector(centre-.07*axis, centre+.07*axis, colours[j], .003)
    U, S = n["ellipsoid"]
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    geom = scn.geoms[len(geoms)]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ELLIPSOID, np.maximum(.08*S, 1e-4), T[:3, 3], U.reshape(-1),
                        np.array([.5, 1., .2, .35], np.float32))
    geoms.append(geom)
    scn.ngeom = len(geoms)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exit-after", type=float, help="Close the window after this many seconds (smoke test)")
    p.add_argument("--rate", type=float, default=5., help="Terminal refresh rate [Hz]")
    args = p.parse_args(argv)
    import mujoco.viewer
    model = build_model()
    data = mujoco.MjData(model)
    explorer = Explorer(model, data)
    start, last_print = time.time(), 0.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:], viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = [.3, 0, .25], 1.6, 140, -20
        while viewer.is_running():
            if args.exit_after and time.time()-start > args.exit_after:
                break
            with viewer.lock():
                explorer.update()
                draw_aids(viewer, explorer)
            viewer.sync()
            if time.time()-last_print > 1./args.rate:
                last_print = time.time()
                print("\033[H\033[J"+explorer.text(), flush=True)
            time.sleep(.01)


if __name__ == "__main__":
    main()
