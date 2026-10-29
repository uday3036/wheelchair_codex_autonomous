"""Containment, case/suspension geometry and safe ownership of travel controls."""
import itertools
import json
import unittest
import mujoco
import numpy as np
from src.grasp_scene import bottle_scene
from src.model import initialize, ROOT
from src.navigation import MobileController
from src.stowage import (MAIN_FOLD,AUX_FOLD,START,AUX_WAIT,LID_OPEN,StorageMotion,
                         storage_status,case_indices,auxiliary_route,storage_legal)
from src.task_scene import CapThread


def folded_scene():
    m=bottle_scene(auxiliary_clamp=True,opening_fixture=True,mobile=True)
    s,r=initialize(m);s.qpos[r['arm_p']]=MAIN_FOLD;s.qpos[r['sec_p']]=AUX_FOLD
    for p in ('grasp','aux'):
        for side in ('left','right'):s.qpos[m.jnt_qposadr[m.joint(f'{p}_{side}_slide').id]]=.03
    mujoco.mj_forward(m,s)
    return m,s,r


class StorageTests(unittest.TestCase):
    def test_both_folded_arms_fit_closed_cases_with_clearance(self):
        m,s,r=folded_scene();status=storage_status(m,s,r)
        self.assertTrue(status['ready'])
        self.assertGreaterEqual(min(status['case_margins_m'].values()),.005)
        self.assertEqual(len(r['arm_p']),6);self.assertEqual(len(r['sec_p']),3)
        self.assertEqual(m.nu,19);self.assertEqual(m.neq,0)
        self.assertFalse(any(c.dist<-.0001 for c in s.contact))
        controller=MobileController(m,s,r);controller.park(m,s)
        self.assertFalse(controller.drive(m,s,.1,0.)['travel_inhibited'])
        self.assertIsNone(controller.brake_position)
        s.qpos[controller.brake_qpos]+=.2
        controller.park(m,s)
        np.testing.assert_allclose(s.ctrl[r['rear_a']],0.)

    def test_side_mount_reaches_pickup_and_service_clearance_waypoints(self):
        from src.run_grasp import plan
        from src.manipulation import SerialArm,transform,axis_rotation
        from src.control import ARM_NAMES
        m,s,r=folded_scene()
        s.qpos[r['arm_p']]=START;s.qpos[r['sec_p']]=AUX_WAIT
        s.qpos[case_indices(m)[0]]=LID_OPEN
        for prefix in ('grasp','aux'):
            for side in ('left','right'):
                s.qpos[m.jnt_qposadr[m.joint(f'{prefix}_{side}_slide').id]]=0.
        mujoco.mj_forward(m,s)
        arm,path,ik,clearance=plan(m,s,r)
        self.assertTrue(all(v['success'] for v in ik))
        seed=path[-1]
        for position,angle in (([.66,.4,1.06],None),([.59,.4,1.1],0.),
                               ([.56,.26,1.09],-np.pi/4),([.48,.26,1.14],-np.pi/3),
                               ([.32,.18,1.23],-np.pi/3)):
            rotation=axis_rotation([0,-1,0],.3) if angle is None else axis_rotation([0,0,1],angle)
            seed,info=arm.inverse(transform(position,rotation),seed)
            self.assertTrue(info['success'],(position,info))

    def test_jaw_centering_opposes_slide_imbalance_without_raising_caps(self):
        from src.run_grasp import centered_hold
        force=centered_hold(7.5,[.025,.015],[0.,0.])
        self.assertLess(force[0],force[1])
        self.assertAlmostEqual(float(np.mean(force)),7.5)
        np.testing.assert_allclose(centered_hold(7.5,[.02,.02],[0.,0.]),[7.5,7.5])
        self.assertLessEqual(np.max(abs(centered_hold(7.5,[.04,0.],[-2.,2.]))),8.)

    def test_main_mount_is_on_frame_inside_side_case_below_armrest(self):
        m,s,r=folded_scene()
        np.testing.assert_allclose(m.body_pos[m.body('arm_base').id],[.02,.39,.18])
        self.assertLess(m.body_pos[m.body('arm_base').id,2],.31)
        self.assertGreater(m.body_pos[m.body('arm_base').id,1],.30)
        self.assertEqual(m.body_parentid[m.body('arm_base').id],m.body('base_link').id)
        self.assertGreater(m.body_mass[m.body('main_frame_bracket').id],0.)
        self.assertLess(m.geom_pos[m.geom('controller_box_collision').id,1],0.)
        from src.stowage import CASE_BOUNDS,extents,arm_geoms
        low,high=extents(m,s,arm_geoms(m,'main'),r['body'])
        self.assertTrue(np.all(low>=CASE_BOUNDS['main'][0]+.002))
        self.assertTrue(np.all(high<=CASE_BOUNDS['main'][1]-.002))
        for x,y in itertools.product(*[(bounds[0]-.006,bounds[1]+.006)
                                      for bounds in CASE_BOUNDS['main'].T[:2]]):
            self.assertLessEqual(np.hypot(x,y),.57)

    def test_unfolded_or_open_cover_blocks_drive_and_retains_tool_control(self):
        for unfolded in (False,True):
            m,s,r=folded_scene();controller=MobileController(m,s,r)
            s.qpos[case_indices(m)[0]]=LID_OPEN
            if unfolded:s.qpos[r['arm_p']]=START;s.qpos[r['sec_p']]=AUX_WAIT
            mujoco.mj_forward(m,s)
            s.ctrl[:]=.25;before=s.ctrl.copy()
            output=controller.drive(m,s,.12,.2)
            self.assertTrue(output['travel_inhibited'])
            np.testing.assert_allclose(s.ctrl[r['rear_a']],0.)
            keep=np.ones(m.nu,dtype=bool);keep[np.r_[r['rear_a'],r['front_a']]]=False
            np.testing.assert_array_equal(s.ctrl[keep],before[keep])

    def test_case_cannot_close_over_deployed_arms(self):
        m,s,r=folded_scene();s.qpos[r['arm_p']]=START;s.qpos[r['sec_p']]=AUX_WAIT
        s.qpos[case_indices(m)[0]]=LID_OPEN;mujoco.mj_forward(m,s)
        motion=StorageMotion(m,s,MobileController(m,s,r),CapThread());saved=s.qpos.copy()
        with self.assertRaisesRegex(RuntimeError,'not contained/folded'):motion.covers(False)
        np.testing.assert_array_equal(s.qpos,saved)

    def test_folding_requires_stationary_base(self):
        m,s,r=folded_scene();s.qvel[0]=.1
        motion=StorageMotion(m,s,MobileController(m,s,r),CapThread())
        with self.assertRaisesRegex(RuntimeError,'chair moves'):motion.fold()
        with self.assertRaisesRegex(RuntimeError,'chair moves'):motion.covers(True)

    def test_cases_clear_suspension_over_flat_floor_articulation_sweep(self):
        m,s,r=folded_scene();names=('rocker_pivot_left','rocker_pivot_right','bogie_pivot_left','bogie_pivot_right')
        positions=[m.jnt_qposadr[m.joint(n).id] for n in names]
        for values in itertools.product((-.07,0.,.07),repeat=4):
            s.qpos[positions]=values;mujoco.mj_forward(m,s)
            for c in s.contact:
                pair=(m.geom(c.geom1).name or '',m.geom(c.geom2).name or '')
                if any('case_' in n for n in pair) and any(n.startswith(('rocker_','bogie_','wheel_')) for n in pair):
                    self.assertGreaterEqual(c.dist,-.0001,(values,pair,c.dist))

    def test_checked_auxiliary_route_preserves_physical_state(self):
        m,s,r=folded_scene();s.qpos[r['arm_p']]=START;s.qpos[case_indices(m)[0]]=LID_OPEN
        mujoco.mj_forward(m,s);saved=s.qpos.copy();vel=s.qvel.copy()
        route=auxiliary_route(m,s,r,AUX_WAIT)
        np.testing.assert_array_equal(s.qpos,saved);np.testing.assert_array_equal(s.qvel,vel)
        for a,b in zip(route[:-1],route[1:]):
            for u in np.linspace(0,1,160):
                s.qpos[r['sec_p']]=a+u*(b-a);mujoco.mj_forward(m,s)
                for c in s.contact:
                    names=(m.geom(c.geom1).name or '',m.geom(c.geom2).name or '')
                    self.assertTrue(c.dist>=.001 or storage_legal(names),(names,c.dist))


class StorageEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # ContinuousEvidenceTests regenerates stale physical evidence first.
        from tests.test_autonomous import ContinuousEvidenceTests
        ContinuousEvidenceTests.setUpClass()
        cls.report=ContinuousEvidenceTests.report;cls.out=ROOT/'results/autonomous_validation'

    def test_motion_is_folded_and_covered_then_deploys_after_braking(self):
        nominal=self.report['trials']['nominal'];nav=nominal['navigation']
        data=np.genfromtxt(self.out/'nominal_navigation.csv',delimiter=',',names=True)
        self.assertTrue(np.all(data['travel_ready']==1))
        self.assertGreaterEqual(data['main_case_margin_m'].min(),.002)
        self.assertGreaterEqual(data['aux_case_margin_m'].min(),.002)
        self.assertLess(np.max(abs(data['main_cover_m'])),.0015)
        self.assertLess(np.max(abs(data['aux_cover_m'])),.0015)
        deployment=nominal['storage']['deployment']
        events={e['event']:e['time_s'] for e in deployment['events']}
        self.assertGreater(events['covers_opened'],nav['duration_s'])
        self.assertGreater(events['arms_deployed_after_stop'],events['covers_opened'])

    def test_empty_arms_return_to_cases_after_delivery_with_recipient_support(self):
        nominal=self.report['trials']['nominal'];idle=nominal['storage']['idle']
        self.assertTrue(idle['final_storage']['ready'])
        last=np.load(self.out/'nominal_poses.npz')['qpos'][-1]
        m,_,r=folded_scene()
        np.testing.assert_allclose(last[r['arm_p']],MAIN_FOLD,atol=np.deg2rad(.5))
        np.testing.assert_allclose(last[r['sec_p']],AUX_FOLD,atol=np.deg2rad(.5))
        np.testing.assert_allclose(last[case_indices(m)[0]],0.,atol=.0015)
        recording=np.load(self.out/'nominal_poses.npz')['qpos']
        for name in ('waist','shoulder','elbow','wrist1','wrist2','wrist3',
                     'secondary_shoulder','secondary_elbow','secondary_wrist'):
            joint=m.joint(name).id;position=m.jnt_qposadr[joint]
            self.assertGreaterEqual(recording[:,position].min(),m.jnt_range[joint,0]-np.deg2rad(.5))
            self.assertLessEqual(recording[:,position].max(),m.jnt_range[joint,1]+np.deg2rad(.5))
        for position,limit in zip(case_indices(m)[0],LID_OPEN):
            self.assertGreaterEqual(recording[:,position].min(),-.0015)
            self.assertLessEqual(recording[:,position].max(),limit+.0015)
        for prefix in ('grasp','aux'):
            for side in ('left','right'):
                position=m.jnt_qposadr[m.joint(f'{prefix}_{side}_slide').id]
                self.assertGreaterEqual(recording[:,position].min(),-.0005)
                self.assertLessEqual(recording[:,position].max(),.0355)
        data=np.genfromtxt(self.out/'nominal_idle_stow.csv',delimiter=',',names=True)
        self.assertGreaterEqual(data['recipient_N'].min(),.8*.3*9.81)
        self.assertLessEqual(data['recipient_N'].max(),6.)
        self.assertGreaterEqual(data['passenger_gap_m'].min(),.01)
        events={e['event']:e['time_s'] for e in nominal['delivery']['events']}
        self.assertGreater(idle['events'][-1]['time_s'],events['passenger_release'])
        for label in ('no_acknowledgement','no_recipient_support'):
            self.assertNotIn('idle',self.report['trials'][label]['storage'])
