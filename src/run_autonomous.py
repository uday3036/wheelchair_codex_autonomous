"""Continuous sensor-to-navigation-to-delivery MuJoCo experiment.

Run: .venv/bin/python -m src.run_autonomous --acknowledge-passenger --render
"""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
from .runtime import step as physics_tick
import numpy as np
from .model import ROOT, initialize
from .grasp_scene import bottle_scene
from .navigation import (MobileController, camera_observation, detect_bottle,
                         lidar_observation, OccupancyGrid, wrap, yaw, START, AUX_WAIT,
                         SensorNoise,observe_bottle,DockingCorridor)
from .task_scene import CapThread, conservative_gap
from .assistive_task import simulate_delivery, AssistiveTask
from .stowage import MAIN_FOLD, AUX_FOLD, storage_status, StorageMotion, case_indices, save_motion, motion_metrics, plot_storage
from .run_experiments import plt
from .simulation import contacts, vehicle_measurement


def navigate(bottle_present=True, blocked_route=False, start=(-1.2, 0., 0.), bottle_xy=(.62, .215),
             caster_angles=(0., 0.),sensor_noise=None,dynamic_obstacle_at=None,precision_docking=True,mass=.3,grip_friction=1.):
    model = bottle_scene(auxiliary_clamp=True, opening_fixture=True, mobile=True,
                         bottle_present=bottle_present, blocked_route=blocked_route, bottle_xy=bottle_xy,
                         dynamic_obstacle=dynamic_obstacle_at is not None,mass=mass,friction=grip_friction)
    state, refs = initialize(model)
    state.qpos[:3] = [start[0], start[1], .4]
    state.qpos[3:7] = [np.cos(start[2]/2), 0., 0., np.sin(start[2]/2)]
    state.qpos[refs['arm_p']], state.qpos[refs['sec_p']] = MAIN_FOLD, AUX_FOLD
    state.qpos[refs['caster_p']] = caster_angles
    for prefix in ('grasp','aux'):
        for side in ('left','right'):
            state.qpos[model.jnt_qposadr[model.joint(f'{prefix}_{side}_slide').id]]=.03
    mujoco.mj_forward(model, state)
    controller = MobileController(model, state, refs)
    thread = CapThread()
    controller.thread = thread
    frames, times, mocaps, rows, events = [], [], [], [], []
    failure, detected, path, goal = None, None, None, None
    grid = OccupancyGrid()
    # All motion requires fresh observed bottle AND a free-space path.
    noise=SensorNoise(**sensor_noise) if sensor_noise is not None else None
    controller.sensor_noise=noise
    observation,detected=observe_bottle(model,state,refs,noise=noise)
    corridor=None
    if detected is None:
        failure = 'No supported bottle detected; driving and grasping inhibited'
    else:
        goal = detected['position'][:2]-[.62, .215]
        origin, dirs, ranges, _ = lidar_observation(model, state, refs)
        grid.update(origin, dirs, ranges)
        path = grid.plan(state.xpos[refs['body'], :2], goal)
        if path is None:
            failure = 'No collision-free observed approach; driving and grasping inhibited'
        else:
            events.append({'event': 'bottle_detected_and_route_planned', 'time_s': float(state.time),
                           'observed_xyz': detected['position'].tolist()})
            if precision_docking:
                try: corridor=DockingCorridor(state.xpos[refs['body'],:2],goal,start[2],grid)
                except ValueError as exc:
                    failure=f'Precision docking rejected: {exc}'
    speed, steering, stable, travelled = 0., 0., 0, 0.
    previous = state.xpos[refs['body'], :2].copy()
    max_wheel_torque = np.zeros(4)
    max_tilt, min_environment_gap = 0., np.inf
    max_arm_error = 0.
    minimum_storage_margin=np.inf
    stowage_interlock_violations=0
    storage_pause=False
    last_unready=-1.
    pause_start=None
    initial_bottle = state.xpos[model.body("bottle").id].copy() if bottle_present else None
    unexpected = {}
    brake_start, docked = None, False
    last_scan = -1.
    failure_started=None
    # Footprint conservative disk only for planning; contacts and actual
    # environment distances are independently audited during physics execution.
    robot_geoms = [i for i in range(model.ngeom) if model.geom_contype[i] and model.geom_group[i] != 4
                   and model.geom_bodyid[i] != 0 and (model.body(model.geom_bodyid[i]).name or '') not in ('bottle', 'cap')]
    environment = [i for i in range(model.ngeom) if model.geom_group[i] == 4]
    for step in range(int(55/model.opt.timestep)):
        t = float(state.time)
        if dynamic_obstacle_at is not None:
            bid=model.body('dynamic_barrier').id
            state.mocap_pos[model.body_mocapid[bid]]=[0.,max(0.,3.-max(0.,t-dynamic_obstacle_at)),.35]
        xy = state.xpos[refs['body'], :2].copy()
        heading = yaw(state, refs)
        if failure is None and not docked:
            remaining = goal[0]-xy[0]
            if remaining < .003 and brake_start is None:
                brake_start = t
                events.append({'event': 'braking_started', 'time_s': t})
            if brake_start is None:
                # Follow the observed route with a lookahead point, never a
                # geometric object pose. Forward-only avoids reverse caster flips.
                nearest = int(np.argmin(np.linalg.norm(path-xy, axis=1)))
                ahead = nearest
                while ahead+1 < len(path) and np.linalg.norm(path[ahead]-xy) < .45:
                    ahead += 1
                target = path[ahead]
                offset = target-xy
                lateral = -np.sin(heading)*offset[0]+np.cos(heading)*offset[1]
                curvature = 2*lateral/max(.2, np.dot(offset, offset))
                desired_steering = float(np.clip(np.arctan(refs['wheelbase']*curvature), -.48, .48))
                if corridor is not None:
                    desired_steering=corridor.steering(xy,heading,refs['wheelbase'])
                desired_speed = float(np.clip(.8*remaining, .025, .14))
            else:
                desired_speed, desired_steering = 0., 0.
            readiness=storage_status(model,state,refs,controller.storage_groups)
            if not readiness['ready']:
                last_unready=t
                if not storage_pause:
                    storage_pause=True;pause_start=t
                    events.append({'event':'travel_paused_for_storage_readiness','time_s':t})
                # A traction transient must not bypass the storage interlock.
                # Brake, let bounded folded control settle, then ramp again.
                speed,steering=0.,0.
            if storage_pause:
                if readiness['ready'] and t-last_unready>=.25:
                    storage_pause=False;pause_start=None
                    events.append({'event':'storage_readiness_reconfirmed','time_s':t})
                else:
                    desired_speed,desired_steering=0.,0.
                    if t-pause_start>3.:
                        failure='Storage readiness did not recover under braking'
            if t<1.5: desired_speed,desired_steering=0.,0.
            speed += np.clip(desired_speed-speed, -.15*model.opt.timestep, .08*model.opt.timestep)
            steering += np.clip(desired_steering-steering, -.15*model.opt.timestep, .15*model.opt.timestep)
            targets=controller.drive(model, state, speed, steering)
            if targets['travel_inhibited']:
                failure='Travel interlock: arms/covers not confirmed stowed'
                stowage_interlock_violations+=1
        else:
            speed,steering=0.,0.
            controller.drive(model, state, 0., 0.)
        if bottle_present:
            thread(model, state)
        # The prescribed recipient surrogate travels with the passenger.
        R = state.xmat[refs['body']].reshape(3, 3)
        state.mocap_pos[0] = state.xpos[refs['body']]+R@np.array([.45, .5, .64])
        physics_tick(model, state)
        if not np.isfinite(np.r_[state.qpos, state.qvel]).all():
            failure = 'Nonfinite navigation physics'
            break
        delta = np.linalg.norm(state.xpos[refs['body'], :2]-previous)
        travelled += delta
        previous = state.xpos[refs['body'], :2].copy()
        max_wheel_torque = np.maximum(max_wheel_torque, np.abs(state.ctrl[np.r_[refs['rear_a'], refs['front_a']]]))
        tilt = np.rad2deg(np.arccos(np.clip(state.xmat[refs['body']].reshape(3, 3)[2, 2], -1., 1.)))
        max_tilt = max(max_tilt, float(tilt))
        for i, c in enumerate(state.contact):
            names = (model.geom(c.geom1).name or '', model.geom(c.geom2).name or '')
            permitted = ('ground' in names and any(n.startswith('wheel_') for n in names)) or (
                'bottle_collision' in names and 'grasp_table' in names) or (set(names) == {'bottle_neck', 'cap_collision'})
            f = np.zeros(6)
            mujoco.mj_contactForce(model, state, i, f)
            if not permitted and (f[0] > 1. or c.dist < -.001):
                unexpected[' / '.join(sorted(names))] = float(f[0])
                failure = 'Unexpected physical contact during navigation'
        if step % 50 == 0:
            ground, slip, bad, traction = contacts(model, state, refs)
            vehicle = vehicle_measurement(model, state, refs)
            max_arm_error = max(max_arm_error, float(np.max(np.abs(np.r_[state.qpos[refs["arm_p"]]-MAIN_FOLD, state.qpos[refs["sec_p"]]-AUX_FOLD]))))
            stowed=storage_status(model,state,refs,controller.storage_groups)
            minimum_storage_margin=min(minimum_storage_margin,min(stowed['case_margins_m'].values()))
            rows.append([state.time, *state.xpos[refs['body']], yaw(state, refs), speed, steering,
                         *state.qvel[refs['rear_d']], *state.ctrl[refs['rear_a']],
                         *state.qpos[refs['front_p']], *state.qpos[refs['caster_p']], *ground,
                         vehicle[2], vehicle[3], *slip.ravel(), *traction, *stowed['case_margins_m'].values(), float(stowed['ready']), *state.qpos[case_indices(model)[0]]])
            frames.append(state.qpos.copy()); times.append(float(state.time)); mocaps.append(state.mocap_pos.copy())
            for a in robot_geoms:
                for b in environment:
                    gap = conservative_gap(model, state, a, b)
                    min_environment_gap = min(min_environment_gap, float(gap))
                    if gap < .01 and failure is None:
                        failure = 'Observed physical clearance below 10 mm; emergency stop' 
            if t-last_scan >= .5:
                last_scan = t
                origin, dirs, ranges, _ = lidar_observation(model, state, refs)
                grid.update(origin, dirs, ranges)
                if failure is None and brake_start is None:
                    if grid.plan(state.xpos[refs['body'], :2], goal) is None:
                        failure = 'Observed route became blocked; emergency stop'
                        events.append({'event':'observed_route_blocked_braking','time_s':float(state.time)})
        if brake_start is not None:
            settled = np.max(np.abs(state.qvel[:6])) < .008 and abs(speed) < .001
            stable = stable+1 if settled else 0
            if stable*model.opt.timestep >= .5 and t-brake_start > 1.:
                docked = True
                break
        if failure is not None:
            if failure_started is None:failure_started=float(state.time)
            if t>2. and ((t-failure_started>=1. and np.linalg.norm(state.qvel[:6])<.008) or t-failure_started>3.):
                break
    if failure is None and not docked:
        failure = 'Navigation/docking timeout'
    # A new close-range sensor estimate is passed into the grasp planner.
    if failure is None:
        observation,detected=observe_bottle(model,state,refs,pitch=.7,noise=noise)
        if detected is None:
            failure = 'Bottle could not be reobserved at arrival'
        elif np.linalg.norm(state.xpos[refs['body'], :2]-goal) > .02 or abs(yaw(state, refs)) > np.deg2rad(3.):
            failure = 'Arrival outside 20 mm / 3 degree manipulation approach gate'
        else:
            events.append({'event': 'arrival_verified_and_brakes_engaged', 'time_s': float(state.time)})
            controller.park(model, state)
    names = ('time,x,y,z,yaw_rad,command_speed,command_steer,left_wheel_rad_s,right_wheel_rad_s,'
             'left_drive_Nm,right_drive_Nm,left_steer_rad,right_steer_rad,left_caster_rad,right_caster_rad,'
             'rear_l_N,rear_r_N,front_l_N,front_r_N,middle_l_N,middle_r_N,forward_m_s,lateral_m_s').split(',')
    names += [f'{wheel}_{axis}_slip_m_s' for wheel in ('rear_l','rear_r','front_l','front_r','middle_l','middle_r') for axis in ('long','lat')]
    names += [f'{wheel}_traction' for wheel in ('rear_l','rear_r','front_l','front_r','middle_l','middle_r')]
    names += ['main_case_margin_m','aux_case_margin_m','travel_ready','main_cover_m','aux_cover_m']
    log = dict(zip(names, np.asarray(rows).T))
    steady = (log['time'] > 3.) & (log['command_speed'] > .1)
    steady_speed_error = float(np.mean(np.abs(log['forward_m_s'][steady]-log['command_speed'][steady])/log['command_speed'][steady])) if steady.any() else None
    stopped_speed = float(np.linalg.norm(state.qvel[:3]))
    normal_sum = sum(log[f'{wheel}_N'] for wheel in ('rear_l','rear_r','front_l','front_r','middle_l','middle_r'))
    supported_mass = float(model.body_subtreemass[refs['body']])
    normal_ratio = float(np.mean(normal_sum[log['time'] > 1.])/(supported_mass*9.81))
    detection_error = None if detected is None or not bottle_present else float(np.linalg.norm(detected['position']-state.xpos[model.body('bottle').id]))
    bottle_drift = None if not bottle_present else float(np.linalg.norm(state.xpos[model.body('bottle').id]-initial_bottle))
    checks = {'bottle_detected' : detected is not None, 'observed_route_found': path is not None,
              'docked': docked and failure is None, 'no_unexpected_contacts': not unexpected,
              'chassis_tilt_below_3deg': max_tilt < 3., 'motor_limits': bool(np.all(max_wheel_torque <= [35,35,25,25])),
              'no_base_weld': model.neq == 0,
              'arms_contained_in_cases': minimum_storage_margin>=.002,
              'travel_interlock_respected': stowage_interlock_violations==0 and bool(np.all(log['travel_ready'][log['command_speed']>1e-10]>.5)),
              'arrival_speed_below_8mm_s': stopped_speed < .008,
              'docking_lateral_error_below_5mm': goal is not None and abs(state.xpos[refs['body'],1]-goal[1])<.005,
              'docking_heading_error_below_1deg': abs(yaw(state,refs))<np.deg2rad(1),
              'steady_speed_error_below_20pct': steady_speed_error is not None and steady_speed_error < .2,
              'ground_support_within_10pct_weight': .9 < normal_ratio < 1.1,
              'parked_arms_below_1deg': max_arm_error < np.deg2rad(1.),
              'bottle_detection_error_below_3mm': detection_error is not None and detection_error < .003,
              'bottle_remains_on_table': bottle_drift is not None and bottle_drift < .003,
              'environment_clearance_above_10mm': min_environment_gap >= .01,
              'finite': bool(np.isfinite(state.qpos).all())}
    checks = {key: bool(value) for key,value in checks.items()}
    metrics = {'passed': all(checks.values()), 'checks': checks, 'blocked_reason': failure,
               'events': events, 'travelled_m': float(travelled), 'arrival_xyz': state.xpos[refs['body']].tolist(),
               'arrival_yaw_deg': float(np.rad2deg(yaw(state, refs))), 'duration_s': float(state.time),
               'travel_posture': 'folded in protective cases; covers closed',
               'minimum_storage_margin_m': float(minimum_storage_margin),
               'max_chassis_tilt_deg': max_tilt, 'maximum_parked_arm_error_rad': max_arm_error,
               'steady_speed_relative_error': steady_speed_error, 'arrival_speed_m_s': stopped_speed,
               'mean_ground_force_to_weight_ratio': normal_ratio, 'bottle_detection_error_m': detection_error,
               'bottle_drift_m': bottle_drift, 'minimum_environment_gap_m': float(min_environment_gap),
               'max_wheel_torque_Nm': max_wheel_torque.tolist(), 'unexpected_contacts': unexpected,
               'goal_xy': None if goal is None else goal.tolist(),
               'precision_docking':corridor is not None,'sensor_noise':sensor_noise,
               'detection': None if detected is None else {k: v.tolist() if isinstance(v,np.ndarray) else v for k,v in detected.items()},
               'scope': 'ideal surface-color RGB-D and environmental LiDAR; calibrated simulated localization; forward approach; free base and rear motor brakes'}
    return model, state, controller, np.array(frames), times, np.array(mocaps), np.array(rows), names, metrics, observation, grid, path



def source_fingerprint():
    digest = hashlib.sha256()
    for path in sorted((ROOT/'src').glob('*.py'))+[ROOT/'models/wheelchair_m1e_rocker.xml']:
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def plot_trial(output, label, data, names, observation, grid, path, metrics):
    c = dict(zip(names, data.T))
    fig, ax = plt.subplots(2,3,figsize=(15,8))
    ax[0,0].imshow(observation['rgb']); ax[0,0].set_title('Measured ideal RGB camera')
    ax[0,1].imshow(observation['range']); ax[0,1].set_title('Measured surface range [m]')
    ax[0,2].imshow(grid.cells.T,origin='lower',extent=[grid.low[0],grid.high[0],grid.low[1],grid.high[1]],vmin=-1,vmax=1,cmap='gray_r')
    if path is not None: ax[0,2].plot(path[:,0],path[:,1],'b--',label='observed route')
    ax[0,2].plot(c['x'],c['y'],'r',label='physical chassis'); ax[0,2].set_title('LiDAR occupancy / approach [m]'); ax[0,2].legend(fontsize=7)
    ax[1,0].plot(c['time'],c['command_speed'],label='command')
    ax[1,0].plot(c['time'],c['forward_m_s'],label='measured');ax[1,0].set_title('Forward speed [m/s]');ax[1,0].legend()
    for k in ('left_steer_rad','right_steer_rad','left_caster_rad','right_caster_rad'):
        ax[1,1].plot(c['time'],np.rad2deg(c[k]),label=k)
    ax[1,1].set_title('Steered front / passive caster angles [deg]');ax[1,1].legend(fontsize=7)
    for wheel in ('rear_l','rear_r','front_l','front_r','middle_l','middle_r'):
        ax[1,2].plot(c['time'],c[f'{wheel}_N'],label=wheel)
    ax[1,2].set_title('Measured wheel ground forces [N]');ax[1,2].legend(fontsize=7)
    for a in ax[1]: a.set_xlabel('Time [s]');a.grid(True)
    fig.tight_layout(); fig.savefig(output/f'{label}_navigation.png',dpi=120);plt.close(fig)



def plot_delivery(output, label, data, names):
    c = dict(zip(names, data.T))
    fig, ax = plt.subplots(3,2,figsize=(13,10))
    for name in ('bottle_x','bottle_y','bottle_z'):
        ax[0,0].plot(c['time'],c[name],label=name)
    ax[0,0].set_title('Free bottle position [m]')
    for name in ('main_left_N','main_right_N','aux_left_N','aux_right_N','recipient_N','table_N'):
        ax[0,1].plot(c['time'],c[name],label=name)
    ax[0,1].set_title('Measured bottle contact forces [N]')
    ax[1,0].plot(c['time'],c['cap_angle_rad'],label='cap relative angle')
    ax[1,0].plot(c['time'],c['cap_detached'],label='physical detach flag')
    ax[1,0].set_title('Measured thread motion / release')
    ax[1,1].plot(c['time'],c['passenger_gap_m']*1000,label='certified clearance')
    ax[1,1].axhline(10,color='r',ls='--',label='minimum');ax[1,1].set_title('Passenger head / torso clearance [mm]')
    ax[2,0].plot(c['time'],c['recipient_N'],label='recipient support')
    ax[2,0].axhline(.8*.3*9.81,ls='--',label='support gate')
    ax[2,0].axhline(6.,color='r',ls='--',label='force limit');ax[2,0].set_title('Recipient release interlock [N]')
    ax[2,1].plot(c['time'],c['phase'],label='task phase');ax[2,1].set_title('Task progress')
    for a in ax.flat: a.set_xlabel('Continuous simulation time [s]');a.grid(True);a.legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/f'{label}_delivery.png',dpi=120);plt.close(fig)

def render_sequence(model, frames, mocaps, times, filename, metrics):
    import imageio.v2 as imageio
    state = mujoco.MjData(model)
    # Replay collision masks do not alter the saved physical poses.
    stages = [(0.,'Folded arms; protective covers closed'),
              (metrics['navigation']['duration_s'],'Drive and brake at table')]
    if 'delivery' in metrics:
        e = {e['event']:e['time_s'] for e in metrics['delivery']['events']}
        stages += [(e.get('cap_detached',times[-1]),'Contact-driven cap removal'),
                   (e.get('cap_bin_support_confirmed',times[-1]),'Dispose cap in bin'),
                   (e.get('recipient_support_verified',times[-1]),'Verify recipient support'),
                   (times[-1],'Delivered; arms folded in cases')]
    indices = {int(np.argmin(np.abs(np.asarray(times)-t))): title for t,title in stages}
    snapshots = []
    option = mujoco.MjvOption()
    option.geomgroup[2] = 0  # Diagnostic passenger envelopes are not robot hardware.
    option.geomgroup[5] = 1
    option.geomgroup[4] = 1  # Physical table/legs and other LiDAR surfaces.
    with mujoco.Renderer(model,height=480,width=640) as renderer:
        state.qpos[:],state.mocap_pos[:]=frames[0],mocaps[0]
        mujoco.mj_forward(model,state)
        close=mujoco.MjvCamera();close.lookat[:]=state.xpos[model.body('base_link').id]+[0.,.15,.35]
        close.distance=1.7;close.azimuth=130;close.elevation=-18
        inspection=[]
        for visible in (1,0):
            option.geomgroup[5]=visible
            renderer.update_scene(state,camera=close,scene_option=option)
            inspection.append(renderer.render().copy())
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax,im,title in zip(axes,inspection,('Travel: covers closed','Same folded pose: case cutaway')):
            ax.imshow(im);ax.set_title(title);ax.axis('off')
        fig.tight_layout();fig.savefig(filename.parent/'arm_storage.png',dpi=120);plt.close(fig)
        option.geomgroup[5]=1
        with imageio.get_writer(filename,mode='I',duration=100,loop=0) as writer:
            for i,(q,hand) in enumerate(zip(frames,mocaps)):
                state.qpos[:],state.mocap_pos[:] = q,hand
                mujoco.mj_forward(model,state)
                renderer.update_scene(state,camera='autonomous_view',scene_option=option)
                frame = renderer.render().copy();writer.append_data(frame)
                if i in indices:
                    if i > 1:
                        renderer.update_scene(state,camera='grasp_view',scene_option=option);frame=renderer.render().copy()
                    snapshots.append((indices[i],frame))
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for a,(title,frame) in zip(axes.flat,snapshots): a.imshow(frame);a.set_title(title);a.axis('off')
    fig.tight_layout();fig.savefig(filename.parent/'autonomous_stages.png',dpi=120);plt.close(fig)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acknowledge-passenger', action='store_true')
    parser.add_argument('--navigation-only', action='store_true')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--robustness',action='store_true',help='Additional complete-task variations and a moving-obstacle stop')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--live',action='store_true',help='Run controller and physics in a live MuJoCo window (macOS: mjpython)')
    parser.add_argument('--live-speed',type=float,default=1.)
    parser.add_argument('--live-exit-after',type=float,help='Automatic display smoke-test timeout in wall-clock seconds')
    parser.add_argument('--start',type=float,nargs=3,metavar=('X','Y','YAW'),help='Initial pose for the nominal experiment')
    parser.add_argument('--output', type=Path, default=ROOT/'results/autonomous_validation')
    args = parser.parse_args(argv)
    if args.start is not None and not np.isfinite(args.start).all():
        parser.error('--start requires three finite pose values')
    if args.live and (args.compare or args.robustness):
        parser.error('Use --live for a single nominal run; run comparisons separately.')
    from .runtime import observe,SimulationStopped
    from contextlib import nullcontext
    from .live_view import LiveView
    live=LiveView(args.live_speed,args.live_exit_after) if args.live else None
    args.output.mkdir(parents=True, exist_ok=True)
    status_path=args.output/'run_status.json'
    status_path.write_text(json.dumps({'status':'running','live':args.live})+'\n')
    try:
        with observe(live) if live is not None else nullcontext():
            code=_execute(args)
        status_path.write_text(json.dumps({'status':'passed' if code==0 else 'failed','live':args.live})+'\n')
        return code
    except (SimulationStopped,KeyboardInterrupt):
        status_path.write_text(json.dumps({'status':'cancelled','live':args.live,'passed':False})+'\n')
        print('Run cancelled by user; this is not a completed task.',flush=True)
        return 130
    except Exception:
        status_path.write_text(json.dumps({'status':'error','live':args.live,'passed':False})+'\n')
        raise
    finally:
        if live is not None: live.close()


def _execute(args):
    report = {'source_sha256': source_fingerprint(), 'trials': {}}
    cases = [('nominal', {} if args.start is None else {'start':tuple(args.start)})]
    if args.compare:
        cases += [('bottle_absent', {'bottle_present': False}), ('blocked_route', {'blocked_route': True}),
                  ('caster_sideways', {'caster_angles': (np.pi/2,np.pi/2), 'navigation_only': True}),
                  ('offset_approach', {'start': (-1.2,.01,0.), 'navigation_only': True})]
        if not args.navigation_only:
            cases += [('no_acknowledgement', {'acknowledge': False}),
                      ('no_recipient_support', {'recipient_support': False})]
    if args.robustness:
        cases += [('offset_35mm_full',{'start':(-1.2,.035,0.)}),
                  ('heading_full',{'start':(-1.2,0.,.025)}),
                  ('bottle_nearer_full',{'bottle_xy':(.605,.215)}),
                  ('bottle_farther_full',{'bottle_xy':(.635,.215)}),
                  ('noisy_full',{'sensor_noise':{'range_std':.001,'rgb_std':.015,'dropout':.05,'seed':12}}),
                  ('moving_obstacle',{'dynamic_obstacle_at':2.,'navigation_only':True})]
    for label, options in cases:
        case_options = dict(options)
        navigation_only = case_options.pop('navigation_only', args.navigation_only)
        acknowledge = case_options.pop('acknowledge', args.acknowledge_passenger)
        recipient_support = case_options.pop('recipient_support', True)
        m,s,controller,frames,times,mocaps,rows,names,nav,observation,grid,path = navigate(**case_options)
        np.savetxt(args.output/f'{label}_navigation.csv', rows, delimiter=',', header=','.join(names))
        np.savez_compressed(args.output/f'{label}_sensors.npz', **observation, occupancy=grid.cells,
                            path=np.empty((0,2)) if path is None else path)
        metrics = {'navigation': nav, 'passed': False, 'configuration': options}
        print(label, 'navigation', nav['passed'], nav['blocked_reason'], flush=True)
        if nav['passed'] and not navigation_only:
            try:
                deployment=StorageMotion(m,s,controller,controller.thread).deploy()
                save_motion(deployment,args.output,f'{label}_deploy')
                metrics['storage']={'deployment':motion_metrics(deployment)}
                frames=np.concatenate([frames,np.asarray(deployment.frames)])
                times=times+deployment.times
                mocaps=np.concatenate([mocaps,np.asarray(deployment.mocaps)])
                # Close-range bottle estimate is refreshed after deployment.
                _,detected=observe_bottle(m,s,controller.refs,pitch=.7,noise=controller.sensor_noise)
                if detected is None: raise RuntimeError('Bottle not reobserved after deployment')
                service = AssistiveTask(acknowledge=acknowledge, recipient_support=recipient_support, prepared=(m,s),
                    observed_bottle=detected['position'], physics_step=controller.park, thread_state=controller.thread)
                result=service.run()
                m, taskframes, data, tasknames, task = result
                np.savetxt(args.output/f'{label}_delivery.csv', data, delimiter=',', header=','.join(tasknames))
                animation = task.pop('animation')
                metrics['delivery'] = task
                metrics['parking'] = {'maximum_base_drift_m': controller.max_park_drift, 'maximum_chassis_tilt_deg': controller.max_park_tilt, 'base_is_free': m.neq == 0}
                metrics['passed'] = task['passed']
                plot_delivery(args.output, label, data, tasknames)
                frames = np.concatenate([frames, taskframes])
                times = times+animation['frame_times_s']
                mocaps = np.concatenate([mocaps, np.asarray(animation['mocap_pos'])])
                print(label, 'delivery', task['passed'], task['blocked_reason'], flush=True)
                if not task['passed']:
                    print('  failed delivery gates:',[name for name,passed in task['checks'].items() if not passed],flush=True)
                if task['passed']:
                    idle=StorageMotion(m,s,controller,controller.thread).fold()
                    save_motion(idle,args.output,f'{label}_idle_stow')
                    metrics['storage']['idle']=motion_metrics(idle)
                    plot_storage(deployment,idle,args.output/f'{label}_storage.png')
                    metrics['passed']=metrics['passed'] and metrics['storage']['idle']['final_storage']['ready']
                    frames=np.concatenate([frames,np.asarray(idle.frames)])
                    times=times+idle.times;mocaps=np.concatenate([mocaps,np.asarray(idle.mocaps)])
                    print(label,'idle stow confirmed',flush=True)
                metrics['total_duration_s']=float(s.time)
                metrics['parking']={'maximum_base_drift_m':controller.max_park_drift,
                    'maximum_chassis_tilt_deg':controller.max_park_tilt,'base_is_free':m.neq==0}
            except RuntimeError as exc:
                metrics['passed']=False
                metrics['error'] = str(exc)
                if hasattr(exc,'transfer'):metrics['transfer_failure']=exc.transfer
                metrics['abort_time_s']=float(s.time)
                metrics['parking']={'maximum_base_drift_m':controller.max_park_drift,
                                    'maximum_chassis_tilt_deg':controller.max_park_tilt,'base_is_free':m.neq==0}
                contacts=[]
                for i,contact in enumerate(s.contact):
                    force=np.zeros(6);mujoco.mj_contactForce(m,s,i,force)
                    contacts.append({'geoms':[m.geom(contact.geom1).name,m.geom(contact.geom2).name],
                                     'distance_m':float(contact.dist),'normal_N':float(force[0])})
                metrics['abort_contacts']=contacts
                np.savez_compressed(args.output/f'{label}_abort_state.npz',qpos=s.qpos,qvel=s.qvel,time_s=s.time)
                print(label, 'BLOCKED', str(exc), flush=True)
        elif navigation_only:
            metrics['passed'] = nav['passed']
        np.savez_compressed(args.output/f'{label}_poses.npz', qpos=frames, frame_times_s=times,mocap_pos=mocaps)
        np.savetxt(args.output/f'{label}_parking.csv', np.array(controller.parking_rows).reshape(-1,11), delimiter=',',
                   header='time,x,y,z,yaw_rad,drift_m,tilt_deg,left_drive_Nm,right_drive_Nm,left_steer_Nm,right_steer_Nm')
        metrics['expected_failure'] = label in ('bottle_absent','blocked_route','no_acknowledgement','no_recipient_support','moving_obstacle')
        if metrics['expected_failure']:
            if 'delivery' in metrics:
                expected_reason = 'No passenger acknowledgement' if label == 'no_acknowledgement' else 'Recipient support insufficient'
                metrics['control_passed'] = (not metrics['passed'] and not any(e['event']=='passenger_release' for e in metrics['delivery']['events'])
                    and expected_reason in (metrics['delivery']['blocked_reason'] or ''))
            else:
                expected_reason = ('No supported bottle detected' if label == 'bottle_absent' else
                                   'Observed route became blocked' if label == 'moving_obstacle' else 'No collision-free observed approach')
                metrics['control_passed'] = (not nav['passed'] and nav['travelled_m'] < (.5 if label=='moving_obstacle' else .01) and 'error' not in metrics
                    and nav['arrival_speed_m_s']<.008 and not nav['unexpected_contacts']
                    and expected_reason in (nav['blocked_reason'] or ''))
        report['trials'][label] = metrics
        plot_trial(args.output, label, rows, names, observation, grid, path, metrics)
        if args.render and label == 'nominal':
            render_sequence(m, frames, mocaps, times, args.output/f'{label}.gif', metrics)
        report['nominal_passed'] = report['trials']['nominal']['passed']
        report['all_expected_outcomes'] = all(t.get('control_passed', t['passed']) for t in report['trials'].values())
        (args.output/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    return int(not report['all_expected_outcomes'])


if __name__ == '__main__':
    raise SystemExit(main())
