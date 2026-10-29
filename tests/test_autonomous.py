"""Sensor/math checks and independent audits of the continuous mobile task."""
import json
import unittest
import mujoco
import numpy as np

from src.control import ARM_NAMES, SEC_NAMES, CASTER_NAMES
from src.grasp_scene import bottle_scene
from src.manipulation import SerialArm
from src.model import ROOT, initialize
from src.navigation import (START, AUX_WAIT, camera_observation, detect_bottle,
                            lidar_observation, OccupancyGrid, MobileController)
from src.run_autonomous import source_fingerprint, main
from src.stowage import case_indices, LID_OPEN, MAIN_FOLD, AUX_FOLD, storage_status


def scene(**options):
    m = bottle_scene(auxiliary_clamp=True, opening_fixture=True, mobile=True, **options)
    s,r = initialize(m)
    s.qpos[0] = -1.2
    s.qpos[r['arm_p']],s.qpos[r['sec_p']] = START,AUX_WAIT
    s.qpos[case_indices(m)[0]]=LID_OPEN
    mujoco.mj_forward(m,s)
    return m,s,r


class SensorAndPlanningTests(unittest.TestCase):
    def test_rgb_depth_estimate_matches_three_independent_bottle_poses(self):
        for xy in ((.605,.215),(.62,.225),(.635,.205)):
            m,s,r = scene(bottle_xy=xy)
            detected = detect_bottle(camera_observation(m,s,r))
            self.assertIsNotNone(detected)
            np.testing.assert_allclose(detected['position'],[*xy,.88],atol=.003)

    def test_missing_bottle_and_nonblue_objects_do_not_trigger(self):
        m,s,r = scene(bottle_present=False)
        self.assertIsNone(detect_bottle(camera_observation(m,s,r)))
        m,s,r = scene()
        m.geom_rgba[m.geom('bottle_collision').id,:3] = [.8,.1,.1]
        self.assertIsNone(detect_bottle(camera_observation(m,s,r)))

    def test_low_barrier_is_seen_by_lidar_without_hiding_bottle(self):
        m,s,r = scene(blocked_route=True)
        self.assertIsNotNone(detect_bottle(camera_observation(m,s,r)))
        origin,dirs,ranges,_ = lidar_observation(m,s,r)
        grid = OccupancyGrid();grid.update(origin,dirs,ranges)
        self.assertIsNone(grid.plan([-1.2,0.],[0.,0.]))

    def test_inflation_detour_unknown_space_and_corner_gates(self):
        g = OccupancyGrid(bounds=(0,4,0,4),resolution=.1)
        self.assertIsNone(g.plan([.2,.2],[3.8,3.8],radius=.15))
        g.cells[:]=0;g.cells[18:22,8:32]=1
        path = g.plan([.3,2.],[3.7,2.],radius=.2)
        self.assertIsNotNone(path)
        self.assertTrue(np.max(abs(path[:,1]-2.))>1.2)
        forbidden = g.inflated(.2)
        for a,b in zip(path[:-1],path[1:]):
            for u in np.linspace(0,1,200):
                self.assertFalse(forbidden[g.index(a+u*(b-a))])
        g.cells[:]=0;g.cells[1,2]=1;g.cells[2,1]=1;g.cells[0,:]=1;g.cells[:,0]=1
        self.assertIsNone(g.plan([.15,.15],[.25,.25],radius=0.))

    def test_moving_root_numpy_fk_and_jacobian_against_engine(self):
        m,s,r = scene()
        s.qpos[:3] = [-.7,.12,.41]
        s.qpos[3:7] = [np.cos(.2),0,0,np.sin(.2)]
        mujoco.mj_forward(m,s)
        for names,site in ((ARM_NAMES,'grasp_center'),(SEC_NAMES,'aux_grasp_center')):
            arm=SerialArm(m,names,site,s)
            q=np.array([s.qpos[m.jnt_qposadr[m.joint(n).id]] for n in names])
            T,J=arm.forward(q,True)
            sid=m.site(site).id
            np.testing.assert_allclose(T[:3,3],s.site_xpos[sid],atol=1e-10)
            np.testing.assert_allclose(T[:3,:3],s.site_xmat[sid].reshape(3,3),atol=1e-10)
            jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv));mujoco.mj_jacSite(m,s,jp,jr,sid)
            dofs=[m.jnt_dofadr[m.joint(n).id] for n in names]
            np.testing.assert_allclose(J,np.vstack([jp[:,dofs],jr[:,dofs]]),atol=1e-10)

    def test_chassis_is_free_and_middle_wheels_have_no_actuation(self):
        m,s,r=scene()
        self.assertEqual(m.neq,0)
        for n in CASTER_NAMES+('wheel_mid_left_joint','wheel_mid_right_joint','wheel_front_left_joint','wheel_front_right_joint'):
            j=m.joint(n).id
            self.assertFalse(m.jnt_limited[j])
            self.assertNotIn(j,m.actuator_trnid[:,0])
        controller=MobileController(m,s,r)
        controller.park(m,s)
        s.qpos[controller.brake_qpos] += 1.
        controller.park(m,s)
        self.assertTrue(np.all(abs(s.ctrl[r['rear_a']])<=35.))


class ContinuousEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output=ROOT/'results/autonomous_validation'
        path=cls.output/'metrics.json'
        report=json.loads(path.read_text()) if path.exists() else {}
        if report.get('source_sha256')!=source_fingerprint() or len(report.get('trials',{}))!=7:
            code=main(['--acknowledge-passenger','--compare'])
            if code: raise AssertionError('Autonomous comparison did not satisfy expected outcomes')
            report=json.loads(path.read_text())
        cls.report=report

    def test_nominal_physics_passes_all_gates_with_free_base(self):
        t=self.report['trials']['nominal']
        self.assertTrue(t['passed'])
        self.assertTrue(all(t['navigation']['checks'].values()))
        self.assertTrue(all(t['delivery']['checks'].values()))
        self.assertEqual(t['delivery']['model_equalities'],0)
        self.assertLess(t['parking']['maximum_base_drift_m'],.02)
        self.assertLess(t['parking']['maximum_chassis_tilt_deg'],3.)

    def test_recording_is_continuous_and_bottle_not_teleported(self):
        p=np.load(self.output/'nominal_poses.npz')
        q,t=p['qpos'],p['frame_times_s']
        self.assertEqual(len(q),len(t));self.assertEqual(len(q),len(p['mocap_pos']))
        self.assertTrue(np.all(np.diff(t)>0))
        self.assertLess(np.max(np.diff(t)),.103)
        self.assertLess(np.max(np.linalg.norm(np.diff(q[:,:3],axis=0),axis=1)),.03)
        m,_,_=scene();bp=m.jnt_qposadr[m.joint('bottle_free').id]
        self.assertLess(np.max(np.linalg.norm(np.diff(q[:,bp:bp+3],axis=0),axis=1)),.03)
        self.assertGreater(q[-1,0]-q[0,0],1.1)
        end=self.report['trials']['nominal']['navigation']['duration_s']
        at=np.argmin(abs(t-end))
        self.assertLess(np.linalg.norm(q[at+1,:3]-q[at,:3]),.002)
        np.testing.assert_allclose(q[0,bp:bp+3],[.62,.215,.88],atol=.001)
        np.testing.assert_allclose(q[-1,bp:bp+3],[.32,.18,1.15],atol=.015)

    def test_release_requires_real_recipient_force_after_acknowledgement(self):
        d=self.report['trials']['nominal']['delivery']
        events={e['event']:e for e in d['events']}
        self.assertLess(events['simulated_passenger_acknowledgement']['time_s'],events['recipient_support_verified']['time_s'])
        self.assertLess(events['recipient_support_verified']['time_s'],events['passenger_release']['time_s'])
        log=np.genfromtxt(self.output/'nominal_delivery.csv',delimiter=',',names=True)
        final=log['phase']==d['phases']['delivered_hold']
        self.assertGreaterEqual(final.sum(),140)
        self.assertGreater(log['recipient_N'][final].min(),.8*.3*9.81)
        self.assertLess(log['recipient_N'].max(),6.)
        for name in ('main_left_N','main_right_N','aux_left_N','aux_right_N'):
            self.assertLess(log[name][final].max(),.2)
        self.assertGreaterEqual(log['passenger_gap_m'].min(),.01)

    def test_missing_bottle_and_blocked_route_prevent_driving(self):
        for name in ('bottle_absent','blocked_route'):
            t=self.report['trials'][name]
            self.assertTrue(t['control_passed'])
            self.assertNotIn('delivery',t)
            self.assertLess(t['navigation']['travelled_m'],.01)
        self.assertFalse(self.report['trials']['bottle_absent']['navigation']['checks']['bottle_detected'])
        self.assertFalse(self.report['trials']['blocked_route']['navigation']['checks']['observed_route_found'])

    def test_mobile_ack_and_support_negative_controls_and_approaches(self):
        for name in ('no_acknowledgement','no_recipient_support'):
            t=self.report['trials'][name]
            self.assertTrue(t['navigation']['passed'])
            self.assertTrue(t['control_passed'])
            self.assertFalse(any(e['event']=='passenger_release' for e in t['delivery']['events']))
        for name in ('caster_sideways','offset_approach'):
            self.assertTrue(self.report['trials'][name]['navigation']['passed'])


if __name__=='__main__':
    unittest.main()
