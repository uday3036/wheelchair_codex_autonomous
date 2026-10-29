"""Independent checks for the live step, simpler skins, sensing and docking."""
import unittest
import numpy as np
import mujoco

from src.grasp_scene import bottle_scene
from src.model import initialize
from src.runtime import step,observe
from src.navigation import SensorNoise,observe_bottle,DockingCorridor,OccupancyGrid
from src.task_scene import conservative_gap


class RuntimeAndDesignTests(unittest.TestCase):
    def test_observer_preserves_exact_physics_and_nested_context(self):
        m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint type="hinge"/><geom type="capsule" size=".02 .15" pos=".15 0 0"/></body></worldbody></mujoco>')
        a,b=mujoco.MjData(m),mujoco.MjData(m)
        class Observer:
            def __init__(self):self.before=self.after=0
            def before_step(self,m,s):self.before+=1
            def after_step(self,m,s):self.after+=1
        outer,inner=Observer(),Observer()
        with observe(outer):
            for _ in range(100):step(m,a)
            with observe(inner):step(m,a)
            step(m,a)
        for _ in range(102):mujoco.mj_step(m,b);mujoco.mj_forward(m,b)
        np.testing.assert_array_equal(a.qpos,b.qpos);np.testing.assert_array_equal(a.qvel,b.qvel)
        self.assertEqual((outer.before,outer.after,inner.before,inner.after),(101,101,1,1))

    def test_simple_style_changes_visuals_without_changing_dynamics_or_collisions(self):
        old,new=(bottle_scene(mobile=True,auxiliary_clamp=True,opening_fixture=True,design_style=s) for s in ('reference','simple'))
        for field in ('body_mass','body_inertia','body_pos','body_quat','jnt_pos','jnt_axis','jnt_range','dof_armature','dof_damping','actuator_trnid','actuator_ctrlrange','exclude_signature'):
            np.testing.assert_array_equal(getattr(old,field),getattr(new,field),err_msg=field)
        for g in range(old.ngeom):
            if not (old.geom_contype[g] or old.geom_conaffinity[g]):continue
            h=new.geom(old.geom(g).name).id
            for field in ('geom_type','geom_size','geom_pos','geom_quat','geom_friction','geom_contype','geom_conaffinity','geom_solref','geom_solimp'):
                np.testing.assert_array_equal(getattr(old,field)[g],getattr(new,field)[h])
        visible=lambda m:sum(m.geom_rgba[:,3]>0)
        self.assertLess(visible(new),visible(old))
        self.assertEqual(new.geom_type[new.geom('reference_link2_outer').id],mujoco.mjtGeom.mjGEOM_CAPSULE)
        a,b=mujoco.MjData(old),mujoco.MjData(new)
        for _ in range(100):step(old,a);step(new,b)
        np.testing.assert_array_equal(a.qpos,b.qpos)


class SensorAndDockingTests(unittest.TestCase):
    def test_noisy_detections_and_missing_bottle(self):
        for present in (True,False):
            m=bottle_scene(mobile=True,auxiliary_clamp=True,opening_fixture=True,bottle_present=present)
            s,r=initialize(m);s.qpos[0]=-1.2;mujoco.mj_forward(m,s)
            for seed in range(4):
                _,detected=observe_bottle(m,s,r,noise=SensorNoise(range_std=.001,rgb_std=.015,dropout=.05,seed=seed))
                if present:
                    self.assertIsNotNone(detected)
                    self.assertLess(np.linalg.norm(detected['position']-[.62,.215,.88]),.003)
                else:self.assertIsNone(detected)
        with self.assertRaises(ValueError):SensorNoise(dropout=1.)

    def test_corridor_derivatives_endpoints_and_obstacle_rejection(self):
        grid=OccupancyGrid();grid.cells[:]=0
        path=DockingCorridor([-1.2,.035],[0,0],.025,grid)
        first=path.reference(-1.2);last=path.reference(0)
        np.testing.assert_allclose(first[:2],[.035,.025],atol=1e-12)
        np.testing.assert_allclose(last,[0,0,0],atol=1e-12)
        x=-.7;eps=1e-5
        derivative=(path.reference(x+eps)[0]-path.reference(x-eps)[0])/(2*eps)
        self.assertAlmostEqual(derivative,np.tan(path.reference(x)[1]),places=8)
        grid.cells[grid.index([-.6,0])]=1
        with self.assertRaisesRegex(ValueError,'collision-free'):DockingCorridor([-1.2,0],[0,0],0,grid)
        with self.assertRaises(ValueError):DockingCorridor([-.2,0],[0,0],0,grid)

    def test_rotated_support_certificate_preserves_gap_and_rejects_overlap(self):
        # A 12 mm gap between boxes must not vanish when the whole pair rotates.
        m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="a" mocap="true"><geom name="a" type="box" size=".2 .1 .1"/></body><body name="b" mocap="true"><geom name="b" type="box" size=".2 .1 .1"/></body></worldbody></mujoco>')
        s=mujoco.MjData(m)
        for angle in np.linspace(-np.pi,np.pi,25):
            q=[np.cos(angle/2),0,0,np.sin(angle/2)]
            s.mocap_quat[:]=q
            s.mocap_pos[0]=[0,0,0];s.mocap_pos[1]=.412*np.array([np.cos(angle),np.sin(angle),0])
            mujoco.mj_forward(m,s)
            self.assertGreaterEqual(conservative_gap(m,s,0,1),.012-1e-12)
            self.assertLessEqual(conservative_gap(m,s,0,1),.012+1e-12)
            s.mocap_pos[1]=.39*np.array([np.cos(angle),np.sin(angle),0]);mujoco.mj_forward(m,s)
            self.assertLess(conservative_gap(m,s,0,1),.01)


if __name__=='__main__':unittest.main()
