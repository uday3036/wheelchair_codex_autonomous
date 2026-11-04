"""Sensor-only sampled forward detours for the bounded table-approach scene.

Search two C2 quintic lateral segments and a straight terminal segment. This
is a finite trajectory library, not a complete arbitrary-room planner. Unknown
cells are forbidden. No obstacle or bottle simulator poses enter this module.
"""
from time import perf_counter
import numpy as np
from .navigation import wrap


class ForwardDetour:
    def __init__(self, start, goal, heading, grid):
        began = perf_counter()
        self.start, self.goal = np.asarray(start, float), np.asarray(goal, float)
        self.end = float(goal[0]-2.)
        distance = self.end-start[0]
        if distance < 2. or abs(heading) > .15:
            raise ValueError('Outside forward detour approach bounds')
        transit, docking = grid.inflated(.70), grid.inflated(.57)
        self.candidates = 0
        best = None
        # Offsets and split locations are searched, not supplied by the scene.
        for fraction in (.4, .5, .6):
            mid = start[0]+fraction*distance
            for offset in np.arange(-1.2, 1.201, .05):
                self.candidates += 1
                segments = [self.segment(start[0], mid, start[1], goal[1]+offset, np.tan(heading)),
                            self.segment(mid, self.end, goal[1]+offset, goal[1], 0.)]
                xs = np.linspace(start[0], goal[0], int(np.ceil((goal[0]-start[0])/.015))+1)
                values = self.evaluate(xs, segments, goal[1])
                ys, slopes, curvatures = values.T
                if np.max(abs(curvatures)) > .58:
                    continue
                indices = np.floor((np.c_[xs, ys]-grid.low)/grid.resolution).astype(int)
                if np.any(indices < 0) or np.any(indices >= grid.shape):
                    continue
                occupied = np.where(xs < goal[0]-1., transit[indices[:,0],indices[:,1]],
                                    docking[indices[:,0],indices[:,1]])
                if occupied.any():
                    continue
                integrand = np.sqrt(1+slopes**2)+.15*curvatures**2
                cost = np.sum(.5*(integrand[1:]+integrand[:-1])*np.diff(xs))
                if best is None or cost < best[0]:
                    best = (cost, segments, np.c_[xs, ys], float(np.max(abs(curvatures))))
        self.planning_s = perf_counter()-began
        if best is None:
            raise ValueError('No observed collision-free curvature-bounded detour')
        self.cost, self.segments, self.path, self.max_curvature = best

    @staticmethod
    def segment(x0, x1, y0, y1, slope):
        length = x1-x0
        c = np.zeros(6); c[0], c[1] = y0, slope*length
        c[3:] = np.linalg.solve([[1,1,1],[3,4,5],[6,12,20]], [y1-y0-c[1],-c[1],0])
        return x0, x1, c

    @staticmethod
    def evaluate(xs, segments, terminal_y):
        xs = np.atleast_1d(xs)
        result = np.zeros((len(xs),3)); result[:,0] = terminal_y
        for x0, x1, c in reversed(segments):
            mask = (xs >= x0-1e-9) & (xs <= x1)
            u = np.clip((xs[mask]-x0)/(x1-x0),0,1)
            y = np.polynomial.polynomial.polyval(u,c)
            dy = np.polynomial.polynomial.polyval(u,np.arange(1,6)*c[1:])/(x1-x0)
            ddy = np.polynomial.polynomial.polyval(u,np.arange(2,6)*np.arange(1,5)*c[2:])/(x1-x0)**2
            result[mask] = np.c_[y,dy,ddy/(1+dy*dy)**1.5]
        return result

    def reference(self, x):
        y, slope, curvature = self.evaluate([np.clip(x,self.start[0],self.goal[0])], self.segments,self.goal[1])[0]
        return y, np.arctan(slope), curvature

    def steering(self, xy, heading, wheelbase):
        y, desired_heading, curvature = self.reference(xy[0])
        # More heading damping than the short straight-corridor controller:
        # caster lag after a detour otherwise leaves residual docking yaw.
        curvature += 5.*wrap(desired_heading-heading)+8.*(y-xy[1])
        return float(np.clip(np.arctan(wheelbase*curvature),-.48,.48))

    def remaining_clear(self, grid, x):
        points = self.path[self.path[:,0] >= x-.03]
        indices = np.floor((points-grid.low)/grid.resolution).astype(int)
        if np.any(indices < 0) or np.any(indices >= grid.shape):
            return False
        transit, docking = grid.inflated(.70), grid.inflated(.57)
        blocked = np.where(points[:,0] < self.goal[0]-1.,transit[indices[:,0],indices[:,1]],docking[indices[:,0],indices[:,1]])
        return not blocked.any()
