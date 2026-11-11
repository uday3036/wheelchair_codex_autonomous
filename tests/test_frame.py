"""Rigid side-rail frame (src/frame_design.py) versus the rocker-bogie reference."""
import unittest

import mujoco
import numpy as np

from src.model import initialize, load_model
from src.simulation import simulate

WHEELS = ('wheel_front_left', 'wheel_front_right', 'wheel_mid_left', 'wheel_mid_right',
          'wheel_rear_left', 'wheel_rear_right')
CURRENT = dict(steering_mode='ackermann', feedback_mode='steering_drive_pi',
               rocker_stiffness=120., clearance_geometry=True)


def models():
    reference = load_model(clearance_geometry=True, rocker_stiffness=120.)
    frame = load_model(clearance_geometry=True, rocker_stiffness=120., simple_frame=True)
    return reference, frame


def whole_robot_inertia(model):
    """Total mass, centre of mass and inertia tensor about it (neutral pose)."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mass = model.body_mass.sum()
    centre = (model.body_mass[:, None]*data.xipos).sum(0)/mass
    inertia = np.zeros((3, 3))
    for i in range(model.nbody):
        rotation = data.ximat[i].reshape(3, 3)
        offset = data.xipos[i]-centre
        inertia += (rotation@np.diag(model.body_inertia[i])@rotation.T +
                    model.body_mass[i]*(offset@offset*np.eye(3)-np.outer(offset, offset)))
    return mass, centre, inertia, data


def static_wheel_loads(model, seconds=1.5):
    state, _ = initialize(model)
    for _ in range(int(seconds/model.opt.timestep)):
        state.ctrl[:] = 0.
        mujoco.mj_step(model, state)
    ground = model.geom('ground').id
    loads = np.zeros(len(WHEELS))
    for c in range(state.ncon):
        contact = state.contact[c]
        for i, name in enumerate(WHEELS):
            pair = (contact.geom1, contact.geom2)
            if model.geom(name+'_collision_0').id in pair and ground in pair:
                force = np.zeros(6)
                mujoco.mj_contactForce(model, state, c, force)
                loads[i] += force[0]
    return loads


class FrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference, cls.frame = models()

    def test_mass_centre_of_mass_and_inertia_are_preserved(self):
        m0, c0, i0, d0 = whole_robot_inertia(self.reference)
        m1, c1, i1, d1 = whole_robot_inertia(self.frame)
        self.assertAlmostEqual(m0, m1, places=9)
        np.testing.assert_allclose(c0, c1, atol=1e-9)
        np.testing.assert_allclose(i0, i1, atol=1e-9)

    def test_wheel_steering_and_caster_axes_do_not_move(self):
        d0, d1 = (mujoco.MjData(m) for m in (self.reference, self.frame))
        for m, d in ((self.reference, d0), (self.frame, d1)):
            mujoco.mj_forward(m, d)
        for name in ('wheel_rear_left', 'wheel_rear_right', 'front_left_knuckle', 'front_right_knuckle',
                     'mid_left_caster_swivel', 'mid_right_caster_swivel', 'wheel_mid_left', 'wheel_front_right'):
            np.testing.assert_allclose(d0.xpos[self.reference.body(name).id],
                                       d1.xpos[self.frame.body(name).id], atol=1e-12, err_msg=name)

    def test_only_the_intended_degrees_of_freedom_changed(self):
        names = lambda m: {m.joint(j).name for j in range(m.njnt)}
        removed = names(self.reference)-names(self.frame)
        added = names(self.frame)-names(self.reference)
        self.assertEqual(removed, {'rocker_pivot_left', 'rocker_pivot_right',
                                   'bogie_pivot_left', 'bogie_pivot_right'})
        self.assertEqual(added, {f'{kind}_{side}_suspension' for kind in ('front', 'mid') for side in ('left', 'right')})
        self.assertEqual(self.reference.nu, self.frame.nu)  # same actuators
        for joint in ('front_left_steer_joint', 'front_right_steer_joint', 'wheel_rear_left_joint',
                      'wheel_rear_right_joint', 'mid_left_caster_yaw', 'mid_right_caster_yaw'):
            self.assertEqual(self.reference.jnt_range[self.reference.joint(joint).id].tolist(),
                             self.frame.jnt_range[self.frame.joint(joint).id].tolist())

    def test_static_load_is_shared_by_all_six_wheels(self):
        """A rigid frame alone left the middle wheels at 11% of the weight each, and
        with only the casters sprung the rear wheels still swapped load diagonally;
        springing the front and middle units brings every wheel into range."""
        loads = static_wheel_loads(self.frame)
        reference = static_wheel_loads(self.reference)
        weight = 9.81*self.frame.body_mass.sum()
        self.assertAlmostEqual(loads.sum(), weight, delta=.01*weight)
        share = loads/loads.sum()
        self.assertGreater(share.min(), .12)
        self.assertLess(share.max(), .21)
        middle = (loads[2]+loads[3])/loads.sum()
        self.assertAlmostEqual(middle, (reference[2]+reference[3])/reference.sum(), delta=.04)

    def test_standard_mobility_gates_match_the_rocker_reference(self):
        for scenario in ('hold', 'straight', 'left', 'right', 'reverse', 'brake', 'reverse_brake'):
            *_, metrics = simulate(scenario, simple_frame=True, **CURRENT)
            self.assertTrue(metrics['passed'], scenario)
            self.assertTrue(metrics['mobility_passed'], (scenario, metrics['mobility']))
            self.assertLess(metrics['peak_tilt_deg'], 1.)

    def test_reverse_turns_remain_a_documented_failure(self):
        """Passive casters still do not align in reverse turns (speed ratio below 0.8);
        the frame changes the number (0.42/0.47 -> about 0.64), not the verdict."""
        for scenario in ('reverse_left', 'reverse_right'):
            *_, metrics = simulate(scenario, simple_frame=True, **CURRENT)
            self.assertFalse(metrics['mobility_passed'], scenario)
            self.assertLess(metrics['mobility']['steady_speed_ratio'], .8)


if __name__ == '__main__':
    unittest.main()
