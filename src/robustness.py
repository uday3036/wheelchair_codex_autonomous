"""Measured reverse readiness and payload/material sensitivity experiments.

These do not convert failed physical challenges into successful service tasks.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco

from .grasp_scene import bottle_scene
from .model import initialize,ROOT
from .navigation import MobileController,wrap
from .stowage import MAIN_FOLD,AUX_FOLD,storage_status
from .runtime import step
from .run_grasp import simulate_grasp
from .run_autonomous import source_fingerprint
from .run_experiments import plt


def reverse_trial(caster_angles=(np.pi,np.pi),duration=12.):
    m=bottle_scene(mobile=True,auxiliary_clamp=True,opening_fixture=True,bottle_present=False)
    s,r=initialize(m)
    s.qpos[r['arm_p']],s.qpos[r['sec_p']]=MAIN_FOLD,AUX_FOLD
    s.qpos[r['caster_p']]=caster_angles
    for prefix in ('grasp','aux'):
        for side in ('left','right'):
            s.qpos[m.jnt_qposadr[m.joint(f'{prefix}_{side}_slide').id]]=.03
    mujoco.mj_forward(m,s)
    c=MobileController(m,s,r)
    rows=[];frames=[];times=[];mocaps=[];unexpected={}
    speed=0.;stable=0.;enabled=False;stop_started=None
    initial=s.xpos[r['body'],:2].copy()
    for k in range(round((duration+2)/m.opt.timestep)):
        angle_error=float(np.max(np.abs(wrap(s.qpos[r['caster_p']]-np.pi))))
        posture=storage_status(m,s,r,c.storage_groups)
        if s.time<1.5:
            request=0.
        elif s.time<duration and stop_started is None:
            stable=stable+m.opt.timestep if angle_error<np.deg2rad(20) else 0.
            enabled=stable>=.5
            request=-.08 if enabled else -.025
        else:
            request=0.
            if stop_started is None:stop_started=float(s.time)
        if not posture['ready']:request=0.
        speed+=np.clip(request-speed,-.08*m.opt.timestep,.15*m.opt.timestep)
        c.drive(m,s,speed,0.)
        step(m,s)
        for i,contact in enumerate(s.contact):
            names=(m.geom(contact.geom1).name,m.geom(contact.geom2).name)
            if 'ground' in names and any(n.startswith('wheel_') for n in names):continue
            force=np.zeros(6);mujoco.mj_contactForce(m,s,i,force)
            if force[0]>1 or contact.dist<-.001:unexpected[' / '.join(sorted(names))]=float(force[0])
        if unexpected and stop_started is None:stop_started=float(s.time)
        if k%50==0:
            rows.append([s.time,*s.xpos[r['body'],:2],speed,angle_error,float(enabled),
                         *s.qpos[r['caster_p']],float(posture['ready']),np.linalg.norm(s.qvel[:3])])
            frames.append(s.qpos.copy());times.append(float(s.time));mocaps.append(s.mocap_pos.copy())
    travel=float(np.linalg.norm(s.xpos[r['body'],:2]-initial))
    checks={'no_unexpected_contacts':not unexpected,'finite':bool(np.isfinite(s.qpos).all()),
            'stopped_below_8mm_s':np.linalg.norm(s.qvel[:3])<.008,
            'arms_still_stored':storage_status(m,s,r,c.storage_groups)['ready']}
    return m,np.asarray(rows),np.asarray(frames),np.asarray(times),np.asarray(mocaps),{
        'initial_caster_angles_rad':list(caster_angles),'travelled_m':travel,'full_reverse_ready':enabled,
        'final_alignment_error_deg':float(np.rad2deg(angle_error)),
        'checks':{k:bool(v) for k,v in checks.items()},'safe_stop':all(checks.values()),
        'reverse_completed':bool(all(checks.values()) and enabled and travel>.3),
        'unexpected_contacts':unexpected,'scope':'open-floor straight reverse; bounded creep until passive caster readiness; no arbitrary reverse turning claim'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/robustness_validation')
    args=p.parse_args(argv);args.output.mkdir(parents=True,exist_ok=True)
    report={'source_sha256':source_fingerprint(),'reverse':{},'payload':{}}
    for label,angles in (('prealigned',(np.pi,np.pi)),('sideways',(np.pi/2,np.pi/2)),('opposed',(0.,np.pi))):
        m,rows,frames,times,mocaps,metrics=reverse_trial(angles)
        report['reverse'][label]=metrics
        np.savetxt(args.output/f'reverse_{label}.csv',rows,delimiter=',',header='time,x,y,command_speed,alignment_error_rad,full_speed_ready,left_caster,right_caster,travel_ready,base_speed')
        np.savez_compressed(args.output/f'reverse_{label}_poses.npz',qpos=frames,frame_times_s=times,mocap_pos=mocaps)
        fig,axes=plt.subplots(2,1,figsize=(8,6))
        axes[0].plot(rows[:,0],np.rad2deg(rows[:,4]));axes[0].axhline(20,ls='--');axes[0].set_ylabel('Caster error [deg]')
        axes[1].plot(rows[:,0],rows[:,3],label='command');axes[1].plot(rows[:,0],rows[:,-1],label='base speed magnitude')
        axes[1].set_ylabel('Speed [m/s]');axes[1].set_xlabel('Time [s]');axes[1].legend();fig.tight_layout()
        fig.savefig(args.output/f'reverse_{label}.png',dpi=120);plt.close(fig)
        print('reverse',label,metrics['reverse_completed'],metrics['safe_stop'],flush=True)
    for label,mass,friction,force in (('nominal',.3,1.,7.5),('heavy_original',.6,1.,7.5),('heavy_grip_material',.6,1.5,7.5)):
        m,frames,rows,names,metrics=simulate_grasp(mass=mass,friction=friction,hold_force=force)
        report['payload'][label]=metrics
        np.savetxt(args.output/f'payload_{label}.csv',rows,delimiter=',',header=','.join(names))
        np.savez_compressed(args.output/f'payload_{label}_poses.npz',qpos=frames)
        print('payload',label,metrics['passed'],flush=True)
    report['scope']='Material friction is an explicit uncalibrated simulation hypothesis. Heavy pickup does not validate complete heavy delivery.'
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    return int(not all(t['safe_stop'] for t in report['reverse'].values()) or not report['payload']['nominal']['passed'] or not report['payload']['heavy_grip_material']['passed'])


if __name__=='__main__':raise SystemExit(main())
