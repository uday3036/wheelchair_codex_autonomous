"""Protective arm cases, measured travel interlock and checked fold/deploy motion.

The cases are task-scene accessories; arm axes, lengths and limits are retained.
All case dimensions below are in the nominal chassis frame (world z minus 0.4).
"""
import xml.etree.ElementTree as ET
import mujoco
from .runtime import step as physics_tick
import numpy as np
from .bottle_command import resting_distractor
from .control import ARM_KP, ARM_KD, ARM_MAX, SEC_KP, SEC_KD, SEC_MAX
from .run_grasp import START, track_arm
from .transfer import MOBILE_AUX_WAIT as AUX_WAIT
from .manipulation import SerialArm, quintic_segment, transform, rotation_vector, axis_rotation
from .control import ARM_NAMES, SEC_NAMES
from .task_scene import conservative_gap

MAIN_FOLD = np.array([0., -.6, 2.4, 0., 1.3, 0.])
AUX_FOLD = np.array([-.8, 2.1, 1.6])
MAIN_EXIT = np.array([0., .3, 1.1, 0., -1.4, 0.])
MAIN_HIGH = np.array([0., 1.2, 1.1, 0., -1.4, 0.])
AUX_EXIT = np.zeros(3)
# Interior bounds of the outboard main case and the auxiliary case.
CASE_BOUNDS = {'main': np.array([[-.04,.333,.12],[.32,.447,.46]]),
               'aux': np.array([[.11,.20,-.033],[.40,.32,.265]])}
LID_OPEN = np.array([.39,.32])


def add_cases(root):
    base = root.find(".//body[@name='base_link']")
    actuator = root.find('actuator')
    def panel(body, name, low, high, mass=.04):
        low, high = np.asarray(low), np.asarray(high)
        ET.SubElement(body,'geom',name=name,type='box',pos=' '.join(map(str,(low+high)/2)),
                      size=' '.join(map(str,(high-low)/2)),mass=str(mass),group='5',
                      rgba='.22 .27 .33 1',contype='1',conaffinity='1')
    for kind,bounds in CASE_BOUNDS.items():
        low,high = bounds
        x0,y0,z0=low; x1,y1,z1=high; wall=.006
        body=ET.SubElement(base,'body',name=f'{kind}_storage_case')
        panel(body,f'{kind}_case_back',[x0-wall,y0-wall,z0-wall],[x0,y1+wall,z1])
        panel(body,f'{kind}_case_outer',[x0,y1,z0-wall],[x1,y1+wall,z1])
        if kind=='main':
            panel(body,'main_case_inner_front',[x0,y0-wall,z0],[x1,y0,z1])
            panel(body,'main_case_floor_front',[x0,y0-wall,z0-wall],[x1,y1,z0])
        else:
            panel(body,'aux_case_inner',[x0,y0-wall,z0-wall],[x1,y0,z1])
            panel(body,'aux_case_floor',[x0,y0,z0-wall],[x1,y1,z0])
        # A roof cover slides clear of the workspace. The front is a deployment
        # port, not a sealed enclosure. Explicit sliding-cover mass and motor.
        lid=ET.SubElement(base,'body',name=f'{kind}_case_lid')
        if kind=='main':
            # Roof/back/outboard side form a sliding hood. Retracting the
            # side frees the main waist workspace; it never passes through
            # the folded arm or a fixed back panel.
            for name in ('main_case_back','main_case_outer','main_case_inner_front'):
                g=body.find(f"geom[@name='{name}']");body.remove(g);lid.append(g)
        axis='-1 0 0'
        distance=LID_OPEN[0 if kind=='main' else 1]
        ET.SubElement(lid,'joint',name=f'{kind}_case_slide',type='slide',axis=axis,
                      range=f'0 {distance}',damping='2',armature='.01',frictionloss='.05')
        panel(lid,f'{kind}_case_cover',[x0-wall,y0-wall,z1],[x1+wall,y1+wall,z1+wall],mass=.10)
        ET.SubElement(actuator,'motor',name=f'{kind}_case_motor',joint=f'{kind}_case_slide',
                      ctrllimited='true',ctrlrange='-10 10',gear='1')


def case_indices(model):
    joints=[model.joint(f'{kind}_case_slide').id for kind in ('main','aux')]
    return (model.jnt_qposadr[joints],model.jnt_dofadr[joints],
            np.array([model.actuator(f'{kind}_case_motor').id for kind in ('main','aux')]))


def arm_geoms(model, kind):
    root=model.body('arm_base' if kind=='main' else 'secondary_arm_base').id
    result=[]
    for g in range(model.ngeom):
        if not model.geom_contype[g]: continue
        b=int(model.geom_bodyid[g])
        while b and b!=root: b=int(model.body_parentid[b])
        if b==root: result.append(g)
    return result


def extents(model, state, ids, frame):
    """Analytic collision-envelope AABBs in the chassis frame (NumPy)."""
    R=state.xmat[frame].reshape(3,3)
    centres=(state.geom_xpos[ids]-state.xpos[frame])@R
    rotations=np.einsum('ij,njk->nik',R.T,state.geom_xmat[ids].reshape(-1,3,3))
    boxes=[]
    for g,M in zip(ids,rotations):
        size=model.geom_size[g];kind=int(model.geom_type[g])
        if kind==int(mujoco.mjtGeom.mjGEOM_BOX): e=np.abs(M)@size
        elif kind==int(mujoco.mjtGeom.mjGEOM_SPHERE): e=np.full(3,size[0])
        elif kind==int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            axis=M[:,2];e=size[0]*np.sqrt(np.maximum(0.,1-axis*axis))+size[1]*np.abs(axis)
        elif kind==int(mujoco.mjtGeom.mjGEOM_CAPSULE): e=size[0]+size[1]*np.abs(M[:,2])
        else: raise ValueError('Unsupported storage collision envelope')
        boxes.append(e)
    return centres-np.asarray(boxes),centres+np.asarray(boxes)


def storage_status(model,state,refs,groups=None):
    groups=groups or {k:arm_geoms(model,k) for k in CASE_BOUNDS}
    margins={}
    for kind,bounds in CASE_BOUNDS.items():
        low,high=extents(model,state,groups[kind],refs['body'])
        margins[kind]=float(min(np.min(low-bounds[0]),np.min(bounds[1]-high)))
    lp,ld,_=case_indices(model)
    error=float(np.max(np.abs(np.r_[state.qpos[refs['arm_p']]-MAIN_FOLD,
                                     state.qpos[refs['sec_p']]-AUX_FOLD])))
    speed=float(np.max(np.abs(state.qvel[np.r_[refs['arm_d'],refs['sec_d']]])))
    closed=bool(np.max(np.abs(state.qpos[lp]))<.0015 and np.max(np.abs(state.qvel[ld]))<.005)
    ready=bool(min(margins.values())>=.002 and error<np.deg2rad(.5) and speed<.03 and closed)
    return {'ready':ready,'case_margins_m':margins,'fold_error_rad':error,'arm_speed_rad_s':speed,'covers_closed':closed}


def lid_control(model,state,target,velocity=None):
    p,d,a=case_indices(model)
    velocity=np.zeros(2) if velocity is None else velocity
    state.ctrl[a]=np.clip(300*(target-state.qpos[p])+6*(velocity-state.qvel[d]),-10.,10.)


class StorageMotion:
    """Actual physics under wheel brakes; all joint paths checked before motion."""
    def __init__(self,model,state,controller,thread):
        self.m,self.s,self.c,self.thread=model,state,controller,thread
        self.r=controller.refs
        self.main=SerialArm(model,ARM_NAMES,'grasp_center',state)
        self.aux=SerialArm(model,SEC_NAMES,'aux_grasp_center',state)
        self.qmain=state.qpos[self.r['arm_p']].copy();self.qaux=state.qpos[self.r['sec_p']].copy()
        self.jaw_target=np.full(4,.03)
        self.jp=model.jnt_qposadr[[model.joint(f'{p}_{side}_slide').id for p in ('grasp','aux') for side in ('left','right')]]
        self.jd=model.jnt_dofadr[[model.joint(f'{p}_{side}_slide').id for p in ('grasp','aux') for side in ('left','right')]]
        self.ja=[model.actuator(f'{p}_{side}_motor').id for p in ('grasp','aux') for side in ('left','right')]
        self.frames=[];self.times=[];self.mocaps=[];self.rows=[];self.events=[];self.steps=0
        self.require_recipient=False
        self.minimum_recipient_force=np.inf
        self.max_reference_speed=0.;self.minimum_passenger_gap=.1
        self.case_geoms=[g for g in range(model.ngeom) if model.geom_group[g]==5]
        self.moving=arm_geoms(model,'main')+arm_geoms(model,'aux')
        self.objects=[g for g in range(model.ngeom) if (model.geom(g).name or '').startswith(('bottle_','cap_collision','cap_rib_'))]

    def validate(self):
        for i,c in enumerate(self.s.contact):
            names=(self.m.geom(c.geom1).name or '',self.m.geom(c.geom2).name or '')
            f=np.zeros(6);mujoco.mj_contactForce(self.m,self.s,i,f)
            if not storage_legal(names) and (c.dist<-.001 or f[0]>1.):
                raise RuntimeError(f'Storage motion contact: {names}, {f[0]:.3f} N')
        if self.steps%10: return
        for proxy in ('passenger_keepout','passenger_head_keepout'):
            p=self.m.geom(proxy).id
            for g in self.moving+self.case_geoms+self.objects:
                gap=conservative_gap(self.m,self.s,p,g)
                self.minimum_passenger_gap=min(self.minimum_passenger_gap,gap)
                if gap<.01: raise RuntimeError('Storage motion violates passenger clearance')

    def tick(self,vm=None,va=None):
        m,s,r=self.m,self.s,self.r
        vm=np.zeros(6) if vm is None else vm;va=np.zeros(3) if va is None else va
        self.max_reference_speed=max(self.max_reference_speed,float(np.max(np.abs(np.r_[vm,va]))))
        track_arm(s,self.main,r['arm_p'],r['arm_d'],r['arm_a'],self.qmain,vm,ARM_KP,ARM_KD,ARM_MAX)
        track_arm(s,self.aux,r['sec_p'],r['sec_d'],r['sec_a'],self.qaux,va,SEC_KP,SEC_KD,SEC_MAX)
        s.ctrl[self.ja]=np.clip(400*(self.jaw_target-s.qpos[self.jp])-3*s.qvel[self.jd],-8.,8.)
        self.thread(m,s);self.c.park(m,s)
        physics_tick(m,s)
        if not np.isfinite(np.r_[s.qpos,s.qvel]).all():raise RuntimeError('Nonfinite storage physics')
        self.validate()
        recipient_force=0.
        for i,c in enumerate(s.contact):
            if {m.geom(c.geom1).name,m.geom(c.geom2).name}=={'bottle_collision','recipient_pad'}:
                force=np.zeros(6);mujoco.mj_contactForce(m,s,i,force);recipient_force+=max(0.,force[0])
        if self.require_recipient:
            self.minimum_recipient_force=min(self.minimum_recipient_force,recipient_force)
            tilt=np.rad2deg(np.arccos(np.clip(s.xmat[m.body('bottle').id].reshape(3,3)[2,2],-1.,1.)))
            if recipient_force<.8*m.body_mass[m.body('bottle').id]*9.81 or recipient_force>6. or tilt>10.:
                raise RuntimeError('Recipient support/force/upright gate failed during idle stow')
        if self.steps%50==0:
            self.frames.append(s.qpos.copy());self.times.append(float(s.time));self.mocaps.append(s.mocap_pos.copy())
            status=storage_status(m,s,r)
            p,_,_=case_indices(m)
            self.rows.append([s.time,*s.qpos[r['arm_p']],*s.qpos[r['sec_p']],*s.qpos[p],
                              *status['case_margins_m'].values(),float(status['ready']),self.minimum_passenger_gap,
                              *s.xpos[m.body('bottle').id],recipient_force])
        self.steps+=1

    def hold(self,duration):
        for _ in range(int(round(duration/self.m.opt.timestep))):self.tick()

    def joint_move(self,main,aux,duration=4.,settle=.5):
        startmain=self.s.qpos[self.r['arm_p']].copy();startaux=self.s.qpos[self.r['sec_p']].copy()
        duration=max(duration,1.875*max(np.max(abs(main-startmain)),np.max(abs(aux-startaux)))/.5)
        saved=self.s.qpos.copy()
        try:
            count=max(2,int(np.ceil(max(np.max(abs(main-startmain)),np.max(abs(aux-startaux)))/.015)))
            for u in np.linspace(0,1,count+1):
                self.s.qpos[self.r['arm_p']]=startmain+u*(main-startmain)
                self.s.qpos[self.r['sec_p']]=startaux+u*(aux-startaux)
                mujoco.mj_forward(self.m,self.s)
                for c in self.s.contact:
                    names=(self.m.geom(c.geom1).name or '',self.m.geom(c.geom2).name or '')
                    if c.dist<.001 and not storage_legal(names):
                        raise RuntimeError(f'Fold/deploy path blocked: {names}')
                for proxy in ('passenger_keepout','passenger_head_keepout'):
                    if any(conservative_gap(self.m,self.s,self.m.geom(proxy).id,g)<.01 for g in self.moving):
                        raise RuntimeError('Fold/deploy path violates passenger clearance')
        finally:
            self.s.qpos[:]=saved;mujoco.mj_forward(self.m,self.s)
        for k in range(int(np.ceil(duration/self.m.opt.timestep))):
            self.qmain,vm=quintic_segment(startmain,main,k*self.m.opt.timestep,duration)
            self.qaux,va=quintic_segment(startaux,aux,k*self.m.opt.timestep,duration)
            self.tick(vm,va)
        self.qmain,self.qaux=main.copy(),aux.copy();self.hold(settle)

    def move_main(self,target,duration=3.):
        initial=self.main.forward(self.s.qpos[self.r['arm_p']])
        vector=rotation_vector(target[:3,:3]@initial[:3,:3].T);angle=np.linalg.norm(vector)
        count=max(1,int(np.ceil(np.linalg.norm(target[:3,3]-initial[:3,3])/.015)),int(np.ceil(angle/.1)))
        seed=self.s.qpos[self.r['arm_p']].copy()
        for u in np.linspace(0.,1.,count+1)[1:]:
            R=initial[:3,:3] if angle<1e-10 else axis_rotation(vector/angle,u*angle)@initial[:3,:3]
            T=transform(initial[:3,3]+u*(target[:3,3]-initial[:3,3]),R)
            q,info=self.main.inverse(T,seed)
            if not info['success']:raise RuntimeError(f'Idle staging IK rejected: {info}')
            self.joint_move(q,self.qaux,duration/count,settle=.02);seed=q

    def covers(self,opened):
        if np.linalg.norm(self.s.qvel[:3])>.008:raise RuntimeError('Cover motion prohibited while chair moves')
        if not opened:
            status=storage_status(self.m,self.s,self.r)
            if min(status['case_margins_m'].values())<.002 or status['fold_error_rad']>np.deg2rad(.5):
                raise RuntimeError('Cover closure prohibited: arms not contained/folded')
        p,_,_=case_indices(self.m);start=self.s.qpos[p].copy()
        goal=LID_OPEN if opened else np.zeros(2);duration=7.
        for k in range(int(duration/self.m.opt.timestep)):
            self.c.lid_target,self.c.lid_velocity=quintic_segment(start,goal,k*self.m.opt.timestep,duration)
            self.tick()
        self.c.lid_target=goal.copy();self.c.lid_velocity=np.zeros(2);self.hold(.5)
        if np.max(abs(self.s.qpos[p]-goal))>.0015:raise RuntimeError('Storage cover position not confirmed')
        self.events.append({'event':'covers_opened' if opened else 'covers_closed','time_s':float(self.s.time)})

    def deploy(self):
        if np.linalg.norm(self.s.qvel[:3])>.008:raise RuntimeError('Deployment prohibited while chair moves')
        self.c.park(self.m,self.s)
        self.covers(True)
        self.joint_move(MAIN_EXIT,AUX_FOLD)
        self.joint_move(START,AUX_FOLD)
        for q in auxiliary_route(self.m,self.s,self.r,AUX_WAIT)[1:]:
            self.joint_move(START,q)
        # Grasp planning expects open tools, commanded gradually under brakes.
        initial=self.s.qpos[self.jp].copy()
        for k in range(1000):
            self.jaw_target,_=quintic_segment(initial,np.zeros(4),k*.002,2.)
            self.tick()
        self.hold(.5)
        self.events.append({'event':'arms_deployed_after_stop','time_s':float(self.s.time)})
        return self

    def fold(self):
        self.require_recipient=True
        for contact in self.s.contact:
            names=(self.m.geom(contact.geom1).name or '',self.m.geom(contact.geom2).name or '')
            if (any(n.startswith(('bottle_','cap_collision','cap_rib_')) for n in names)
                    and any(n.startswith(('grasp_','aux_')) and n.endswith('_pad') for n in names)):
                raise RuntimeError('Folding prohibited while a tool contacts the bottle or cap')
        if np.linalg.norm(self.s.qvel[:3])>.008:raise RuntimeError('Folding prohibited while chair moves')
        # The completed task already withdrew from the recipient. Reposition
        # through the tested staging poses before entering either narrow case.
        # Lift away from the recipient first. A direct joint interpolation to
        # START can sweep the forearm through the receiving hand platform.
        raised=self.main.forward(self.qmain);raised[2,3]+=.04
        self.move_main(raised)
        stage=self.main.forward(START);stage[2,3]+=.07
        self.move_main(stage)
        self.move_main(self.main.forward(START))
        self.joint_move(START,self.qaux)
        for q in auxiliary_route(self.m,self.s,self.r,AUX_FOLD)[1:]:
            self.joint_move(START,q)
        # Keep the forearm above the receiving hand while turning the waist
        # back into the case. The deployment exit pose is obstructed after
        # the recipient arrives, even though it was free before pickup.
        high=MAIN_HIGH.copy();high[0]=START[0]
        self.joint_move(high,AUX_FOLD)
        self.joint_move(MAIN_HIGH,AUX_FOLD)
        self.joint_move(MAIN_FOLD,AUX_FOLD)
        self.hold(1.)
        self.covers(False)
        self.hold(3.)
        status=storage_status(self.m,self.s,self.r)
        if not status['ready']:raise RuntimeError(f'Folded idle state not confirmed: {status}')
        self.events.append({'event':'arms_stowed_for_idle_or_travel','time_s':float(self.s.time)})
        return self


def storage_legal(names):
    """Empty tools must not touch the bottle/cap during fold or deployment."""
    return (('ground' in names and any(n.startswith('wheel_') for n in names)) or
            ('bottle_collision' in names and any(n in ('grasp_table','recipient_pad') for n in names)) or resting_distractor(names) or
            (any(n.startswith(('cap_collision','cap_rib_')) for n in names) and
             any(n.startswith('cap_bin_') or n=='bottle_neck' for n in names)))


def auxiliary_route(model,state,refs,goal,seed=21,configuration_check=None):
    """Student NumPy bidirectional joint-space RRT, with engine collision checks.

    Engine kinematics/collisions verify candidates; sampling, nearest neighbor,
    extension, connection and shortcutting are implemented here, not outsourced.
    Every checked proxy pose is restored before returning to physics.
    """
    qindex=refs['sec_p'];start=state.qpos[qindex].copy();saved=state.qpos.copy()
    limits=model.jnt_range[[model.joint(n).id for n in SEC_NAMES]].copy()
    rng=np.random.default_rng(seed)
    moving=arm_geoms(model,'aux')
    table=model.geom('grasp_table').id if any((model.geom(g).name or '')=='grasp_table' for g in range(model.ngeom)) else None
    def valid(q):
        state.qpos[qindex]=q;mujoco.mj_forward(model,state)
        if configuration_check is not None:
            if not configuration_check():return False
        else:
            for c in state.contact:
                names=(model.geom(c.geom1).name or '',model.geom(c.geom2).name or '')
                if c.dist<.001 and not storage_legal(names):return False
            # Contact tests alone let a route graze the table between samples (the
            # level rigid frame exposed a 1.2 N touch): require 5 mm of clearance.
            if table is not None and any(mujoco.mj_geomDistance(model,state,table,g,.006,None)<.005 for g in moving):
                return False
        return all(conservative_gap(model,state,model.geom(proxy).id,g)>=.01
                   for proxy in ('passenger_keepout','passenger_head_keepout') for g in moving)
    def edge(a,b):
        steps=max(1,int(np.ceil(np.max(abs(b-a))/.025)))
        return all(valid(a+u*(b-a)) for u in np.linspace(0,1,steps+1))
    def root_path(nodes,parents,i):
        out=[]
        while i>=0:out.append(nodes[i]);i=parents[i]
        return out[::-1]
    try:
        if not valid(start) or not valid(goal):raise RuntimeError('Auxiliary storage route endpoint obstructed')
        if edge(start,goal):return [start,goal.copy()]
        A,B=[start],[goal.copy()];pa,pb=[-1],[-1];swapped=False;path=None
        for iteration in range(2500):
            sample=B[0] if rng.random()<.2 else rng.uniform(limits[:,0]+.01,limits[:,1]-.01)
            i=int(np.argmin(np.linalg.norm(np.asarray(A)-sample,axis=1)))
            delta=sample-A[i];q=A[i]+delta*min(1.,.18/max(np.linalg.norm(delta),1e-12))
            if edge(A[i],q):
                A.append(q);pa.append(i);ia=len(A)-1
                j=int(np.argmin(np.linalg.norm(np.asarray(B)-q,axis=1)))
                for _ in range(100):
                    delta=q-B[j];q2=B[j]+delta*min(1.,.18/max(np.linalg.norm(delta),1e-12))
                    if not edge(B[j],q2):break
                    B.append(q2);pb.append(j);j=len(B)-1
                    if np.linalg.norm(q2-q)<1e-9:
                        path=root_path(A,pa,ia)+root_path(B,pb,j)[::-1][1:]
                        if swapped:path=path[::-1]
                        break
                if path is not None:break
            A,B,pa,pb=B,A,pb,pa;swapped=not swapped
        if path is None:raise RuntimeError('No checked auxiliary fold/deploy route found')
        result=[path[0]];i=0
        while i<len(path)-1:
            j=len(path)-1
            while j>i+1 and not edge(path[i],path[j]):j-=1
            result.append(path[j]);i=j
        return result
    finally:
        state.qpos[:]=saved;mujoco.mj_forward(model,state)


def motion_metrics(motion):
    return {'events':motion.events,'maximum_reference_speed_rad_s':motion.max_reference_speed,
            'minimum_passenger_clearance_m':motion.minimum_passenger_gap,
            'minimum_recipient_support_N':None if not motion.require_recipient else motion.minimum_recipient_force,
            'final_storage':storage_status(motion.m,motion.s,motion.r)}


def save_motion(motion,output,label):
    names=['time']+['main_q'+str(i) for i in range(6)]+['aux_q'+str(i) for i in range(3)]
    names+=['main_cover_m','aux_cover_m','main_case_margin_m','aux_case_margin_m','travel_ready','passenger_gap_m',
            'bottle_x','bottle_y','bottle_z','recipient_N']
    np.savetxt(output/f'{label}.csv',np.asarray(motion.rows),delimiter=',',header=','.join(names))


def plot_storage(deployment,idle,filename):
    from .run_experiments import plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for motion,label in ((deployment,'deploy'),(idle,'idle stow')):
        a=np.asarray(motion.rows);t=a[:,0]
        axes[0,0].plot(t,a[:,10],label=label+' main cover')
        axes[0,0].plot(t,a[:,11],ls='--',label=label+' auxiliary cover')
        axes[0,1].plot(t,a[:,12]*1000,label=label+' main')
        axes[0,1].plot(t,a[:,13]*1000,ls='--',label=label+' auxiliary')
        axes[1,0].plot(t,a[:,14],label=label+' travel-ready')
        axes[1,1].plot(t,a[:,19],label=label+' recipient support')
    axes[0,0].set_title('Measured storage cover travel [m]')
    axes[0,1].set_title('Arm envelope margin inside cases [mm]')
    axes[0,1].axhline(2.,color='r',ls=':',label='required containment')
    axes[1,0].set_title('Measured readiness interlock')
    axes[1,1].set_title('Recipient support during final folding [N]')
    for ax in axes.flat:ax.set_xlabel('Continuous simulation time [s]');ax.legend(fontsize=7);ax.grid(True)
    fig.tight_layout();fig.savefig(filename,dpi=120);plt.close(fig)
