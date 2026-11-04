"""NumPy perception, occupancy search and feedback navigation in the MuJoCo scene.

The ideal calibrated RGB-D camera ray-traces surface colors and depths; it is
not a neural detector. LiDAR senses environmental collision surfaces in group 4.
Object names/poses never enter the detector or the planner. Pose localization is
an explicit simulated calibrated localization input, not claimed SLAM.
"""
import heapq
import mujoco
import numpy as np

from .control import (IntegralFeedback, hold_and_drive, yaw, ARM_KP, ARM_KD,
                      ARM_MAX, SEC_KP, SEC_KD, SEC_MAX)
from .run_grasp import START
from .transfer import MOBILE_AUX_WAIT as AUX_WAIT
from .stowage import MAIN_FOLD, AUX_FOLD, storage_status, arm_geoms, lid_control, case_indices


def wrap(angle):
    return (angle+np.pi)%(2*np.pi)-np.pi


def ray_sensor(model, state, origin, directions, groups, max_range=5.):
    """Sensor boundary: expose only measured ranges and surface RGB, no IDs."""
    vectors = np.ascontiguousarray(directions, dtype=float)
    ids = np.full(len(vectors), -1, np.int32)
    distances = np.full(len(vectors), -1., float)
    mujoco.mj_multiRay(model, state, np.asarray(origin, float), vectors.ravel(),
                      np.asarray(groups, np.uint8), 1, -1, ids, distances, len(ids), max_range)
    rgb = np.zeros((len(ids), 3))
    seen = ids >= 0
    rgb[seen] = model.geom_rgba[ids[seen], :3]
    distances[(distances < 0) | (distances > max_range)] = np.nan
    return distances, rgb


def camera_observation(model, state, refs, width=160, height=100, pitch=.21, max_range=5.):
    """Ideal pinhole color/depth, calibrated to the chassis; physical occlusion."""
    R = state.xmat[refs['body']].reshape(3, 3)
    origin = state.xpos[refs['body']]+R@np.array([.15, 0., .90])
    # Optical forward pitched down 12 degrees, horizontal FOV 64 degrees.
    forward = R@np.array([np.cos(pitch), 0., -np.sin(pitch)])
    right = R@np.array([0., -1., 0.])
    up = np.cross(right, forward)
    focal = width/(2*np.tan(np.deg2rad(32.)))
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    directions = (forward+(u.ravel()[:, None]-(width-1)/2)*right/focal-
                  (v.ravel()[:, None]-(height-1)/2)*up/focal)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    depth, rgb = ray_sensor(model, state, origin, directions, [1, 1, 0, 0, 1, 1], max_range=max_range)
    points = origin+depth[:, None]*directions
    return {'rgb': rgb.reshape(height, width, 3), 'range': depth.reshape(height, width),
            'points': points.reshape(height, width, 3), 'origin': origin}


class SensorNoise:
    """Seeded range/RGB noise and dropped pixels at the sensor boundary.

    This is a stress-test distribution, not calibrated camera hardware.
    """
    def __init__(self,range_std=.001,rgb_std=.01,dropout=.02,seed=42):
        if not np.isfinite([range_std,rgb_std,dropout]).all() or range_std<0 or rgb_std<0 or not 0<=dropout<1:
            raise ValueError('Nonnegative finite noise and dropout in [0,1) required')
        self.range_std,self.rgb_std,self.dropout=range_std,rgb_std,dropout
        self.rng=np.random.default_rng(seed)

    def apply(self,observation):
        result={k:np.array(v,copy=True) for k,v in observation.items()}
        depth=result['range']
        vectors=result['points']-result['origin']
        valid=np.isfinite(depth)&(depth>0)
        directions=np.zeros_like(vectors)
        directions[valid]=vectors[valid]/depth[valid,None]
        noisy=depth+self.rng.normal(0,self.range_std,depth.shape)
        noisy[(self.rng.random(depth.shape)<self.dropout)|(noisy<=0)]=np.nan
        result['range']=noisy
        result['points']=result['origin']+directions*noisy[...,None]
        result['rgb']=np.clip(result['rgb']+self.rng.normal(0,self.rgb_std,result['rgb'].shape),0,1)
        return result


def observe_bottle(model,state,refs,pitch=.21,noise=None,resolution=(160,100),max_range=5.):
    """Use only sensor samples; repeated fits reduce noisy partial-arc bias."""
    raw=camera_observation(model,state,refs,pitch=pitch,width=resolution[0],height=resolution[1],max_range=max_range)
    if noise is None:
        return raw,detect_bottle(raw)
    detections=[]
    for _ in range(5):
        raw=noise.apply(camera_observation(model,state,refs,pitch=pitch,width=resolution[0],height=resolution[1],max_range=max_range))
        detected=detect_bottle(raw,refine=True)
        if detected is not None: detections.append(detected)
    if len(detections)<3: return raw,None
    position=np.median([d['position'] for d in detections],axis=0)
    if np.max(np.linalg.norm(np.array([d['position'] for d in detections])-position,axis=1))>.006:
        return raw,None
    result=dict(detections[-1]);result['position']=position;result['consistent_observations']=len(detections)
    return raw,result


def detect_bottle(observation,refine=False):
    """Fit known 28 mm radius vertical blue cylinder from RGB-D surface points.

    Brown horizontal tabletop evidence is required beneath the candidate.
    This recognizes the simulated bottle class; arbitrary real bottles excluded.
    """
    rgb = observation['rgb'].reshape(-1, 3)
    xyz = observation['points'].reshape(-1, 3)
    finite = np.isfinite(xyz).all(axis=1)
    blue = finite & (rgb[:, 2] > .75) & (rgb[:, 1] > .4) & (rgb[:, 0] < .2)
    blue &= (xyz[:, 2] > .805) & (xyz[:, 2] < .955)
    p = xyz[blue]
    if len(p) < 8:
        return None
    # Linear least squares circle fit; reject ill-conditioned/poorly fitted arcs.
    A = np.c_[2*p[:, :2], np.ones(len(p))]
    solution, _, _, _ = np.linalg.lstsq(A, np.sum(p[:, :2]**2, axis=1), rcond=None)
    centre = solution[:2]
    radius = np.sqrt(max(0., solution[2]+centre@centre))
    if refine:
        # Fixed-radius Gauss-Newton on measured surface points. Joint use of
        # algebraic radius and residual still rejects a different-sized object.
        for _ in range(12):
            delta=centre-p[:,:2];distance=np.maximum(np.linalg.norm(delta,axis=1),1e-9)
            J=delta/distance[:,None]
            residuals=distance-.028
            correction=np.linalg.solve(J.T@J+1e-6*np.eye(2),J.T@residuals)
            centre-=np.clip(correction,-.005,.005)
            if np.linalg.norm(correction)<1e-7:break
    residual = np.sqrt(np.mean((np.linalg.norm(p[:, :2]-centre, axis=1)-.028)**2))
    brown = finite & (rgb[:, 0] > .4) & (rgb[:, 0] < .6) & (rgb[:, 1] > .25) & (rgb[:, 1] < .45)
    table = xyz[brown]
    table = table[(np.abs(table[:, 2]-.78) < .015) & (np.linalg.norm(table[:, :2]-centre, axis=1) < .3)]
    if len(table) < 8 or abs(radius-.028) > .004 or residual > .003:
        return None
    tabletop = float(np.median(table[:, 2]))
    return {'position': np.r_[centre, tabletop+.1], 'pixels': int(len(p)),
            'radius_m': float(radius), 'fit_rms_m': float(residual), 'tabletop_z': tabletop}


class DockingCorridor:
    """NumPy quintic lateral alignment with heading/curvature feedforward.

    Bounded forward corridor, not a general nonholonomic global planner.
    Every sampled reference must remain in observed/inflated free space.
    """
    def __init__(self,start,goal,heading,grid):
        self.x0=float(start[0]);distance=float(goal[0]-start[0])
        if distance<.8 or abs(goal[1]-start[1])>.08 or abs(heading)>.08:
            raise ValueError('Outside the validated forward docking corridor')
        # Finish lateral alignment before the last 0.30 m straight settling
        # section: passive caster/tire dynamics lag the geometric reference.
        self.length=distance-.30
        self.coeff=np.zeros(6);self.coeff[0]=start[1];self.coeff[1]=self.length*np.tan(heading)
        self.coeff[3:]=np.linalg.solve(np.array([[1,1,1],[3,4,5],[6,12,20]],float),
                                      [goal[1]-sum(self.coeff[:3]),-self.coeff[1],0.])
        blocked=grid.inflated()
        for x in np.linspace(start[0],goal[0],160):
            y,_,_=self.reference(x);index=grid.index([x,y])
            if not grid.inside(index) or blocked[index]:
                raise ValueError('Docking corridor is not observed collision-free')

    def reference(self,x):
        u=float(np.clip((x-self.x0)/self.length,0,1))
        c=self.coeff
        y=float(np.polynomial.polynomial.polyval(u,c))
        dy=float(np.polynomial.polynomial.polyval(u,np.arange(1,6)*c[1:])/self.length)
        ddy=float(np.polynomial.polynomial.polyval(u,np.arange(2,6)*np.arange(1,5)*c[2:])/self.length**2)
        return y,float(np.arctan(dy)),float(ddy/(1+dy*dy)**1.5)

    def steering(self,xy,heading,wheelbase):
        y,desired_heading,curvature=self.reference(xy[0])
        curvature+=3.5*wrap(desired_heading-heading)+8.*(y-xy[1])
        return float(np.clip(np.arctan(wheelbase*curvature),-.48,.48))


def lidar_observation(model, state, refs, max_range=5., rays=720):
    R = state.xmat[refs['body']].reshape(3, 3)
    origin = state.xpos[refs['body']]+R@np.array([.15, 0., .1])
    angles = np.linspace(-np.pi, np.pi, rays, endpoint=False)
    directions = np.einsum('ij,kj->ik', np.c_[np.cos(angles), np.sin(angles), np.zeros(len(angles))], R)
    ranges, _ = ray_sensor(model, state, origin, directions, [0, 0, 0, 0, 1, 0],max_range=max_range)
    endpoints = origin+ranges[:, None]*directions
    return origin[:2], directions[:, :2], ranges, endpoints


class OccupancyGrid:
    """Log-free ternary occupancy; unknown cells remain forbidden to the planner."""
    def __init__(self, bounds=(-2.5, 1.5, -2., 2.), resolution=.04):
        self.resolution = resolution
        self.low = np.array([bounds[0], bounds[2]])
        self.high = np.array([bounds[1], bounds[3]])
        self.shape = tuple(np.ceil((self.high-self.low)/resolution).astype(int))
        self.cells = np.full(self.shape, -1, np.int8)

    def index(self, point):
        return tuple(np.floor((np.asarray(point)-self.low)/self.resolution).astype(int))

    def inside(self, ij):
        return all(0 <= ij[k] < self.shape[k] for k in range(2))

    def point(self, ij):
        return self.low+(np.asarray(ij)+.5)*self.resolution

    def update(self, origin, directions, ranges, max_range=5.):
        for direction, distance in zip(directions, ranges):
            hit = np.isfinite(distance)
            length = min(distance, max_range) if hit else max_range
            for t in np.arange(0., max(0., length-.02), self.resolution/2):
                cell = self.index(origin+direction*t)
                if self.inside(cell) and self.cells[cell] != 1:
                    self.cells[cell] = 0
            if hit:
                cell = self.index(origin+direction*distance)
                if self.inside(cell):
                    self.cells[cell] = 1

    def inflated(self, radius=.57):
        occupied = self.cells == 1
        forbidden = self.cells != 0
        count = int(np.ceil(radius/self.resolution))
        for x in range(-count, count+1):
            for y in range(-count, count+1):
                if np.hypot(x, y)*self.resolution <= radius:
                    a = (slice(max(0, x), min(self.shape[0], self.shape[0]+x)),
                         slice(max(0, y), min(self.shape[1], self.shape[1]+y)))
                    b = (slice(max(0, -x), min(self.shape[0], self.shape[0]-x)),
                         slice(max(0, -y), min(self.shape[1], self.shape[1]-y)))
                    forbidden[a] |= occupied[b]
        return forbidden

    def simplify(self, points, radius=.57):
        blocked = self.inflated(radius)
        def visible(a, b):
            for u in np.linspace(0., 1., max(2, int(np.linalg.norm(b-a)/self.resolution*3))):
                ij = self.index(a+u*(b-a))
                if not self.inside(ij) or blocked[ij]:
                    return False
            return True
        output, i = [points[0]], 0
        while i < len(points)-1:
            j = len(points)-1
            while j > i+1 and not visible(points[i], points[j]):
                j -= 1
            output.append(points[j]); i = j
        return np.asarray(output)

    def plan(self, start, goal, radius=.57):
        blocked = self.inflated(radius)
        a, b = self.index(start), self.index(goal)
        if not self.inside(a) or not self.inside(b) or blocked[a] or blocked[b]:
            return None
        queue, cost, parent = [(0., a)], {a: 0.}, {}
        expanded = set()
        directions = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if i or j]
        while queue:
            _, current = heapq.heappop(queue)
            if current in expanded:
                continue
            expanded.add(current)
            if current == b:
                path = [current]
                while current != a:
                    current = parent[current]
                    path.append(current)
                points = np.array([self.point(i) for i in path[::-1]])
                points[0], points[-1] = start, goal
                return self.simplify(points, radius)
            for di, dj in directions:
                nxt = (current[0]+di, current[1]+dj)
                if not self.inside(nxt) or blocked[nxt]:
                    continue
                if di and dj and (blocked[current[0]+di, current[1]] or blocked[current[0], current[1]+dj]):
                    continue
                value = cost[current]+np.hypot(di, dj)
                if value < cost.get(nxt, np.inf):
                    cost[nxt], parent[nxt] = value, current
                    heapq.heappush(queue, (value+np.linalg.norm(np.subtract(nxt, b)), nxt))
        return None


class MobileController:
    """Canonical bounded wheel feedback plus parked arm references and brakes."""
    def __init__(self, model, state, refs):
        self.refs = refs
        self.integral = IntegralFeedback('steering_drive_pi')
        self.brake_position = None
        self.parking_rows = []
        self.parking_calls = 0
        self.max_park_drift = 0.
        self.max_park_tilt = 0.
        self.lid_target=state.qpos[case_indices(model)[0]].copy();self.lid_velocity=np.zeros(2)
        self.storage_groups={kind:arm_geoms(model,kind) for kind in ('main','aux')}
        self.interlock_stops=0

    def drive(self, model, state, v, steer):
        r = self.refs
        status=storage_status(model,state,r,self.storage_groups)
        inhibited=abs(v)>1e-10 and not status['ready']
        if inhibited:
            v,steer=0.,0.
            self.interlock_stops+=1
        if v!=0.: self.brake_position=None  # Recapture wheel position on the next stop.
        previous_ctrl=state.ctrl.copy()
        targets = hold_and_drive(model, state, r, v, steer, steering_mode='ackermann', integral=self.integral)
        targets['travel_inhibited']=inhibited
        if inhibited:
            # The manipulation owner retains its arm/tool/cover controls; a
            # rejected travel request must not release a held object or fold it.
            preserve=np.ones(model.nu,dtype=bool)
            preserve[np.r_[r['rear_a'],r['front_a']]]=False
            state.ctrl[preserve]=previous_ctrl[preserve]
            return targets
        lid_control(model,state,self.lid_target,self.lid_velocity)
        for p, d, a, goal, kp, kd, limit in (
                (r['arm_p'], r['arm_d'], r['arm_a'], MAIN_FOLD, ARM_KP, ARM_KD, ARM_MAX),
                (r['sec_p'], r['sec_d'], r['sec_a'], AUX_FOLD, SEC_KP, SEC_KD, SEC_MAX)):
            if status['fold_error_rad']>.035 or min(status['case_margins_m'].values())<0: goal=state.qpos[p].copy()
            state.ctrl[a] = np.clip(kp*(goal-state.qpos[p])-kd*state.qvel[d]+state.qfrc_bias[d], -limit, limit)
        for prefix in ('grasp', 'aux'):
            for side in ('left', 'right'):
                j = model.joint(f'{prefix}_{side}_slide').id
                state.ctrl[model.actuator(f'{prefix}_{side}_motor').id] = np.clip(
                    400*(.03-state.qpos[model.jnt_qposadr[j]])-3*state.qvel[model.jnt_dofadr[j]], -8., 8.)
        return targets

    def park(self, model, state):
        r = self.refs
        lid_control(model,state,self.lid_target,self.lid_velocity)
        if self.brake_position is None:
            joints = [model.joint(n).id for n in ('wheel_rear_left_joint', 'wheel_rear_right_joint')]
            self.brake_qpos = model.jnt_qposadr[joints]
            self.brake_position = state.qpos[self.brake_qpos].copy()
            self.park_origin = state.xpos[r['body']].copy()
        state.ctrl[r['rear_a']] = np.clip(120*(self.brake_position-state.qpos[self.brake_qpos])-
                                        12*state.qvel[r['rear_d']], -35., 35.)
        state.ctrl[r['front_a']] = np.clip(-80*state.qpos[r['front_p']]-8*state.qvel[r['front_d']], -25., 25.)

        drift = float(np.linalg.norm(state.xpos[r['body']]-self.park_origin))
        tilt = float(np.rad2deg(np.arccos(np.clip(state.xmat[r['body']].reshape(3,3)[2,2],-1.,1.))))
        self.max_park_drift = max(self.max_park_drift, drift)
        self.max_park_tilt = max(self.max_park_tilt, tilt)
        if self.parking_calls % 50 == 0:
            self.parking_rows.append([state.time, *state.xpos[r['body']], yaw(state,r), drift, tilt,
                                      *state.ctrl[r['rear_a']], *state.ctrl[r['front_a']]])
        self.parking_calls += 1
        if drift > .02 or tilt > 3.:
            raise RuntimeError('Parking stability gate exceeded: retain object and block next transition')
