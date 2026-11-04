"""Sensor-driven detour geometry and fail-closed planning checks."""
import unittest
import mujoco
import numpy as np
from src.grasp_scene import bottle_scene
from src.model import initialize
from src.navigation import OccupancyGrid, observe_bottle, lidar_observation
from src.obstacle_navigation import ForwardDetour
from src.stowage import MAIN_FOLD, AUX_FOLD


def sensed_scene(layout='detour', present=True):
    m=bottle_scene(mobile=True,opening_fixture=True,auxiliary_clamp=True,
                   obstacles=layout,bottle_present=present)
    s,r=initialize(m);s.qpos[0]=-5.4
    s.qpos[r['arm_p']],s.qpos[r['sec_p']]=MAIN_FOLD,AUX_FOLD
    mujoco.mj_forward(m,s)
    _,bottle=observe_bottle(m,s,r,resolution=(320,200),max_range=6.5)
    grid=OccupancyGrid(bounds=(-6.5,1.5,-3,3))
    origin,dirs,ranges,_=lidar_observation(m,s,r,max_range=6.5,rays=1440)
    grid.update(origin,dirs,ranges,max_range=6.5)
    return m,s,r,bottle,grid


class ObstaclePlanningTests(unittest.TestCase):
    def test_sensor_detection_and_opposite_side_detours(self):
        for layout,sign in [('detour',-1),('mirrored',1)]:
            m,s,r,bottle,grid=sensed_scene(layout)
            self.assertIsNotNone(bottle)
            # Distant docking uses XY only; the partly observed tabletop height
            # is refreshed at close range before any arm target is generated.
            np.testing.assert_allclose(bottle['position'][:2],[.62,.215],atol=.003)
            self.assertLess(abs(bottle['position'][2]-.88),.01)
            goal=bottle['position'][:2]-[.62,.215]
            self.assertIsNotNone(grid.plan(s.xpos[r['body'],:2],goal))
            route=ForwardDetour([-5.4,0],goal,0,grid)
            self.assertGreater(sign*route.reference(-3.7)[0],.15)
            self.assertLessEqual(route.max_curvature,.58)
            self.assertTrue(route.remaining_clear(grid,-5.4))
            np.testing.assert_allclose(route.reference(goal[0]),[goal[1],0,0],atol=1e-10)
            self.assertTrue(grid.inflated(.57)[grid.index([-3.7,0])])

    def test_blocked_and_unknown_space_rejected(self):
        _,_,_,bottle,grid=sensed_scene('blocked')
        self.assertIsNotNone(bottle)
        with self.assertRaises(ValueError):ForwardDetour([-5.4,0],[0,0],0,grid)
        grid.cells[:]=-1
        with self.assertRaises(ValueError):ForwardDetour([-5.4,0],[0,0],0,grid)

    def test_new_obstacle_invalidates_remaining_reference(self):
        _,_,_,_,grid=sensed_scene()
        route=ForwardDetour([-5.4,0],[0,0],0,grid)
        point=route.path[len(route.path)//2]
        grid.cells[grid.index(point)]=1
        self.assertFalse(route.remaining_clear(grid,-5.4))

    def test_absent_bottle_not_inferred_from_scene(self):
        self.assertIsNone(sensed_scene(present=False)[3])

    def test_derivatives_and_segment_continuity(self):
        _,_,_,_,grid=sensed_scene()
        route=ForwardDetour([-5.4,0],[0,0],0,grid)
        for x in (-3.4,-3.7,-1.7,-1.):
            eps=1e-5
            derivative=(route.reference(x+eps)[0]-route.reference(x-eps)[0])/(2*eps)
            self.assertAlmostEqual(derivative,np.tan(route.reference(x)[1]),places=7)
        for x in (route.segments[0][1],route.end):
            np.testing.assert_allclose(route.reference(x-1e-7),route.reference(x+1e-7),atol=1e-5)


if __name__=='__main__':unittest.main()
