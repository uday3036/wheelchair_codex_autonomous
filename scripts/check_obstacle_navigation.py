"""Audit saved current-source obstacle trials and their recorded geometry."""
import json
from pathlib import Path
import sys
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.run_autonomous import source_fingerprint, navigate, plot_trial
from scripts.replay_autonomous import Recording


def main():
    if '--run-controls' in sys.argv:
        for directory,options in [('obstacle_blocked',{'obstacles':'blocked'}),
                                  ('obstacle_absent',{'obstacles':'detour','bottle_present':False})]:
            path=ROOT/'results'/directory;path.mkdir(exist_ok=True)
            m,s,c,q,t,h,rows,names,nav,observation,grid,route=navigate(**options)
            trial={'configuration':options,'passed':nav['passed'],'navigation':nav}
            report={'source_sha256':source_fingerprint(),'trials':{'nominal':trial}}
            (path/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
            np.savez_compressed(path/'nominal_poses.npz',qpos=q,frame_times_s=t,mocap_pos=h)
            np.savetxt(path/'nominal_navigation.csv',rows,delimiter=',',header=','.join(names))
            plot_trial(path,'nominal',rows,names,observation,grid,route,trial)
    checks={};summary={}
    for directory in ('obstacle_validation','obstacle_mirrored','obstacle_blocked','obstacle_absent'):
        path=ROOT/'results'/directory
        report=json.loads((path/'metrics.json').read_text())
        trial=report['trials']['nominal'];nav=trial['navigation']
        checks[directory+'_current_source']=report['source_sha256']==source_fingerprint()
        if directory in ('obstacle_blocked','obstacle_absent'):
            reason='No collision-free observed approach' if directory=='obstacle_blocked' else 'No supported bottle detected'
            checks[directory+'_protective_stop']=(not trial['passed'] and reason in (nav['blocked_reason'] or '')
                and nav['travelled_m']<.01 and nav['arrival_speed_m_s']<.008 and not nav['unexpected_contacts']
                and 'delivery' not in trial)
        else:
            checks[directory+'_navigation']=nav['passed'] and all(nav['checks'].values())
            recording=Recording(path)
            checks[directory+'_recording']=bool(np.all(np.diff(recording.times)>0))
            checks[directory+'_measured_detour']=float(np.ptp(recording.qpos[:,1]))>.15
            checks[directory+'_scene_restored']=mujoco.mj_name2id(recording.model,mujoco.mjtObj.mjOBJ_GEOM,'approach_box')>=0
            if directory=='obstacle_validation':
                checks['delivery_and_idle_storage']=(trial['passed'] and all(trial['delivery']['checks'].values())
                    and trial['storage']['idle']['final_storage']['ready'])
                # Counterfactual geometry audit: straight travel intersects a
                # box. This is a probe only; no simulation recording is edited.
                m,s=recording.model,mujoco.MjData(recording.model)
                s.qpos[:]=recording.qpos[0]
                straight_contact=False
                for x in np.linspace(-4.5,-3.,100):
                    s.qpos[0]=x;s.qpos[1]=0
                    mujoco.mj_forward(m,s)
                    straight_contact |= any('approach_box' in (m.geom(c.geom1).name,m.geom(c.geom2).name) and c.dist<0 for c in s.contact)
                checks['straight_route_really_intersects_obstacle']=straight_contact
        summary[directory]={key:nav[key] for key in ('passed','blocked_reason','travelled_m','duration_s',
            'arrival_xyz','arrival_yaw_deg','minimum_environment_gap_m','arrival_speed_m_s','planner')}
    result={'source_sha256':source_fingerprint(),'checks':{k:bool(v) for k,v in checks.items()},
            'passed':all(checks.values()),'trials':summary}
    target=ROOT/'results/obstacle_validation/acceptance_summary.json'
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return int(not result['passed'])


if __name__=='__main__':raise SystemExit(main())
