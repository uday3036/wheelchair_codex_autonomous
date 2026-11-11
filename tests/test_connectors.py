"""Connector registration and unchanged dynamics (presentation geometry only)."""
import unittest
from unittest.mock import patch
import mujoco
import numpy as np
from src.grasp_scene import bottle_scene
from src.model import initialize
from src.stowage import MAIN_FOLD, AUX_FOLD
from src.geometry_audit import assembly_components

# MuJoCo 3.3.7 mj_geomDistance can return a false 0.0 for separated rotated boxes
# when distmax is large (about 3% of random pairs at distmax>=0.1; none at 1 mm).
DISTMAX = 1e-3
OPTIONS = dict(mobile=True, auxiliary_clamp=True, opening_fixture=True)  # rocker-bogie (default)
FRAME_OPTIONS = dict(OPTIONS, simple_frame=True)  # optional rigid side-rail frame
DESIGNS = (OPTIONS, FRAME_OPTIONS)


def random_pose(m, state, rng, mode):
    """Set every actuated/passive joint except the free bottle/cap in its range."""
    for j in range(m.njnt):
        if m.jnt_type[j] not in (2, 3) or m.joint(j).name in ('bottle_free', 'cap_free'):
            continue
        limited = m.jnt_limited[j] and m.jnt_range[j][1] > m.jnt_range[j][0]
        lo, hi = m.jnt_range[j] if limited else (-np.pi, np.pi)
        state.qpos[m.jnt_qposadr[j]] = (rng.uniform(lo, hi) if mode == 'uniform'
                                        else rng.choice([lo, (lo + hi) / 2, hi]))
    mujoco.mj_forward(m, state)


class ConnectorTests(unittest.TestCase):
    def test_whole_assembly_is_one_connected_piece_through_motion(self):
        for options in DESIGNS:
            m = bottle_scene(**options)
            s, r = initialize(m)
            rng = np.random.default_rng(7)
            for mode in ('uniform', 'extremes'):  # extremes: each joint at min/middle/max
                for k in range(150):
                    random_pose(m, s, rng, mode)
                    groups = assembly_components(m, s)
                    self.assertEqual(len(groups), 1, (options, mode, k, [g[:6] for g in groups[1:]]))
            for name, fold in (('folded', 1.), ('deployed', 0.)):
                s.qpos[r['arm_p']] = fold*MAIN_FOLD
                s.qpos[r['sec_p']] = fold*AUX_FOLD
                mujoco.mj_forward(m, s)
                self.assertEqual(len(assembly_components(m, s)), 1, (options, name))

    def test_connectivity_audit_detects_missing_connectors(self):
        """Negative control: without the connector layer the chair falls apart."""
        for options, minimum in ((OPTIONS, 10), (FRAME_OPTIONS, 5)):
            with patch('src.structural_geometry.add_structural_geometry'):
                m = bottle_scene(**options)
            s, _ = initialize(m)
            mujoco.mj_forward(m, s)
            self.assertGreater(len(assembly_components(m, s)), minimum, options)

    def test_every_wheel_unit_is_attached_to_its_bogie_beam(self):
        m = bottle_scene(**OPTIONS)
        s, _ = initialize(m)
        mujoco.mj_forward(m, s)
        for side in ('left', 'right'):
            for kind in ('front', 'middle'):
                riser = m.geom(f'structure_{side}_{kind}_riser').id
                beams = [m.geom(f'bogie_{side}_collision_{i}').id for i in (0, 1)]
                gap = min(mujoco.mj_geomDistance(m, s, riser, b, DISTMAX, None) for b in beams)
                self.assertLessEqual(gap, 1e-8, (side, kind, gap))
        for name in ('core_standoff_0', 'left_footrest_hanger', 'right_footrest_hanger',
                     'left_side_plate_bracket_0', 'left_side_plate_bracket_1'):
            self.assertGreaterEqual(m.geom('structure_'+name).group, 3)
            self.assertEqual(m.geom('structure_'+name).contype, 0)  # visual only

    def test_every_frame_member_is_joined_and_the_chassis_is_rigid(self):
        m = bottle_scene(**FRAME_OPTIONS)
        s, _ = initialize(m)
        mujoco.mj_forward(m, s)
        for side in ('left', 'right'):
            joined = [(f'frame_front_arm_{side}', f'frame_rail_{side}'),
                      (f'frame_middle_arm_{side}', f'frame_rail_{side}'),
                      (f'frame_rear_post_{side}', f'frame_rail_{side}'),
                      (f'frame_rear_motor_{side}', f'frame_rear_post_{side}'),
                      (f'frame_crossmember_0_{side}', f'frame_rail_{side}'),
                      (f'frame_crossmember_1_{side}', f'frame_rail_{side}'),
                      (f'frame_crossmember_0_{side}', 'base_visual_0'),
                      (f'frame_crossmember_1_{side}', 'base_visual_0')]
            for a, b in joined:
                gap = mujoco.mj_geomDistance(m, s, m.geom(a).id, m.geom(b).id, DISTMAX, None)
                self.assertLessEqual(gap, 1e-8, (a, b, gap))
        # No spring-loaded rocker/bogie pivots; the only added joints are the wheel-unit springs.
        names = {m.joint(j).name for j in range(m.njnt)}
        self.assertFalse({n for n in names if n.startswith(('rocker_', 'bogie_'))})
        self.assertEqual({n for n in names if n.endswith('_suspension')},
                         {f'{kind}_{side}_suspension' for kind in ('front', 'mid') for side in ('left', 'right')})
        for name in ('core_standoff_0', 'left_footrest_hanger', 'right_footrest_hanger',
                     'left_side_plate_bracket_0', 'left_side_plate_bracket_1'):
            self.assertGreaterEqual(m.geom('structure_'+name).group, 3)
            self.assertEqual(m.geom('structure_'+name).contype, 0)  # visual only
        for g in range(m.ngeom):
            if m.geom(g).name.startswith('frame_'):  # frame geoms are massless visuals
                self.assertEqual((m.geom_contype[g], m.geom_conaffinity[g]), (0, 0))

    def test_wheel_axles_and_kingpins_through_rotation(self):
        for options in DESIGNS:
            self._wheel_axles_and_kingpins(options)

    def _wheel_axles_and_kingpins(self, options):
        frame = bool(options.get('simple_frame'))
        m = bottle_scene(**options)
        s, r = initialize(m)
        for angle in np.linspace(-np.pi, np.pi, 33):
            for side in ('left', 'right'):
                for name in (f'mid_{side}_caster_yaw', f'wheel_mid_{side}_joint',
                             f'wheel_front_{side}_joint', f'wheel_rear_{side}_joint'):
                    j=m.joint(name).id
                    s.qpos[m.jnt_qposadr[j]]=angle
                j=m.joint(f'front_{side}_steer_joint').id
                s.qpos[m.jnt_qposadr[j]]=np.clip(angle,-.7,.7)
            mujoco.mj_forward(m,s)
            for side in ('left','right'):
                for kind,wheel in [('middle','mid'),('front','front'),('rear','rear')]:
                    a=m.geom('frame_rear_motor_'+side if (kind=='rear' and frame) else f'structure_{side}_{kind}_axle').id
                    b=m.geom(f'wheel_{wheel}_{side}_visual_0').id
                    self.assertLessEqual(mujoco.mj_geomDistance(m,s,a,b,DISTMAX,None),1e-8)
                for kind in ('middle','front'):
                    a=m.geom(f'structure_{side}_{kind}_kingpin').id
                    b=m.geom(f'frame_{kind}_arm_{side}' if frame else f'structure_{side}_{kind}_bridge').id
                    self.assertLessEqual(mujoco.mj_geomDistance(m,s,a,b,DISTMAX,None),1e-8)

    def test_connected_jaws_across_motion_and_fold(self):
        m = bottle_scene(mobile=True, auxiliary_clamp=True, opening_fixture=True)
        s, r = initialize(m)
        for u in np.linspace(0, 1, 41):
            s.qpos[r['arm_p']] = u*MAIN_FOLD
            s.qpos[r['sec_p']] = u*AUX_FOLD
            for prefix in ('grasp', 'aux'):
                for side in ('left', 'right'):
                    j = m.joint(f'{prefix}_{side}_slide').id
                    s.qpos[m.jnt_qposadr[j]] = .035*u
            mujoco.mj_forward(m, s)
            for prefix in ('grasp', 'aux'):
                for side in ('left', 'right'):
                    names = [prefix+'_guide_rail', prefix+'_'+side+'_carriage',
                             prefix+'_'+side+'_finger_support', prefix+'_'+side+'_pad']
                    for a, b in zip(names, names[1:]):
                        gap = mujoco.mj_geomDistance(m, s, m.geom(a).id, m.geom(b).id, DISTMAX, None)
                        self.assertLessEqual(gap, 1e-8, (u, a, b, gap))

    def test_presentation_additions_preserve_physics(self):
        options = dict(mobile=True, auxiliary_clamp=True, opening_fixture=True)
        with patch('src.grasp_scene.connector'), \
                patch('src.structural_geometry.add_structural_geometry'):
            old = bottle_scene(**options)  # no jaw connectors, no structural layer
        new = bottle_scene(**options)
        for attr in ('body_mass', 'body_inertia', 'body_ipos', 'dof_damping', 'actuator_gear'):
            np.testing.assert_array_equal(getattr(old, attr), getattr(new, attr))
        a, r = initialize(old)
        b, _ = initialize(new)
        a.qpos[r['arm_p']] = b.qpos[r['arm_p']] = MAIN_FOLD
        a.qpos[r['sec_p']] = b.qpos[r['sec_p']] = AUX_FOLD
        for _ in range(100):
            mujoco.mj_step(old, a)
            mujoco.mj_step(new, b)
        np.testing.assert_allclose(a.qpos, b.qpos, atol=1e-12, rtol=0)
        np.testing.assert_allclose(a.qvel, b.qvel, atol=1e-12, rtol=0)
