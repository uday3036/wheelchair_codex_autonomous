"""Independently audit current experiment evidence without rerunning physics."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.run_autonomous import source_fingerprint


def main():
    fp=source_fingerprint();checks={};reports={}
    for name in ('autonomous_validation','full_task_variations','robustness_validation','live_validation'):
        path=ROOT/'results'/name/'metrics.json'
        reports[name]=json.loads(path.read_text())
        checks[name+'_source_current']=reports[name].get('source_sha256')==fp
    baseline=reports['autonomous_validation']
    checks['seven_baseline_outcomes']=len(baseline['trials'])==7 and baseline['all_expected_outcomes']
    variants=reports['full_task_variations']
    checks['seven_variation_outcomes']=len(variants['trials'])==7 and variants['all_expected_outcomes']
    checks['live_controller_task']=reports['live_validation']['nominal_passed'] and json.loads((ROOT/'results/live_validation/run_status.json').read_text())=={'status':'passed','live':True}
    for label in ('no_acknowledgement','no_recipient_support'):
        trial=baseline['trials'][label]['delivery']
        rows=np.genfromtxt(ROOT/f'results/autonomous_validation/{label}_delivery.csv',delimiter=',',names=True)
        held=rows['phase']==trial['phases']['blocked_retention']
        force=sum(rows[key][held] for key in ('main_left_Fz','main_right_Fz','aux_left_Fz','aux_right_Fz'))
        checks[label+'_retained_under_braking']=bool(held.sum()>=140 and force.min()>=.8*trial['bottle_mass_kg']*9.81
            and not any(e['event']=='passenger_release' for e in trial['events']))
    for label,trial in variants['trials'].items():
        if trial.get('expected_failure'):
            checks[label+'_stopped_without_contact']=bool(trial['control_passed'] and trial['navigation']['arrival_speed_m_s']<.008 and not trial['navigation']['unexpected_contacts'])
        else:
            checks[label+'_full_task_gates']=bool(trial['passed'] and all(trial['delivery']['checks'].values())
                and trial['storage']['idle']['final_storage']['ready'] and trial['delivery']['peak_recipient_normal_N']<=6.)
    sensitivity=reports['robustness_validation']
    checks['prealigned_reverse']=sensitivity['reverse']['prealigned']['reverse_completed']
    checks['misaligned_reverse_stops']=all(sensitivity['reverse'][k]['safe_stop'] and not sensitivity['reverse'][k]['reverse_completed'] for k in ('sideways','opposed'))
    checks['material_comparison']=not sensitivity['payload']['heavy_original']['passed'] and sensitivity['payload']['heavy_grip_material']['passed']
    manifest=json.loads((ROOT/'assets/manifest.json').read_text())
    checks['asset_sources_current']=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected for name,expected in manifest['sources'].items())
    checks['asset_files_unchanged']=all(hashlib.sha256((ROOT/'assets'/name).read_bytes()).hexdigest()==expected for name,expected in manifest['files_sha256'].items())
    output=ROOT/'results/project_validation';output.mkdir(parents=True,exist_ok=True)
    report={'source_sha256':fp,'checks':{k:bool(v) for k,v in checks.items()},'passed':all(checks.values()),
            'scope':'Bounded service and explicitly tested comparisons; misaligned reverse remains unresolved, and hardware/human calibration is not validated'}
    (output/'integration_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return int(not report['passed'])


if __name__=='__main__':raise SystemExit(main())
