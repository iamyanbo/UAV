"""One receipt-driven, real end-to-end integration cycle on DGX Spark.

Every learner gets one optimizer update; DAgger and PPO first collect fresh
physics. Preference ties remain explicit blocking outcomes, never fake labels.
The existing campaign budgets and scientific acceptance gates are unchanged.
"""
import argparse
import json
from pathlib import Path
from program_scheduler import run


def specification(visual_pack,safety,episode,demonstrations=None):
    root=Path.home()/'uav-rgb-flight';stages=[]
    visual=json.loads((visual_pack/'checkpoints.json').read_text())
    def artifact(role):return str(visual_pack/visual['artifacts'][role]['path'])
    def job(name,suffix):return '{stage:'+name+':job}/'+suffix
    def stage(name,kind,command,receipt,outputs,depends=(),peak=24,seconds=1800,inputs=()):
        item=dict(id=name,kind=kind,peak_gib=peak,seconds=seconds,
            depends_on=[dict(stage=x,status='completed') for x in depends],inputs=list(inputs),
            command=command,receipt=receipt,outputs=outputs)
        if name.endswith('update'):
            directory='ppo' if name=='ppo-update' else 'configurator' if name.startswith('qwen-') else 'training'
            item['checkpoint']='{job}/'+directory+'/latest.pt'
            resume=list(command)
            if '--initialize-from' in resume:
                offset=resume.index('--initialize-from');del resume[offset:offset+2]
            item['resume_command']=resume+['--resume','{checkpoint}']
        stages.append(item)
    def pack(name,policy,world,qwen,depends):
        destination='{round}/'+name;command=['python3','{source}/package_navigation.py']
        for role in ('goal','odometry','projection'):command+=['--'+role,artifact(role)]
        command+=['--safety',str(safety),'--policy',policy,'--world',world,'--qwen',qwen,
            '--goal-match-threshold',str(visual['goal_match_threshold']),'--output',destination]
        stage(name,'cpu',command,destination+'/checkpoints.json',[destination+'/checkpoints.json'],depends,peak=4,
              inputs=[policy,world,qwen,str(safety),str(visual_pack/'checkpoints.json')])
        return destination
    def flight(name,pack_path,depends,sample=False,combined=False):
        command=['python3','{source}/collect_learning_round.py','--episode-id',episode,'--controller-checkpoints',pack_path,
            '--goal-collection',collection]
        if sample:command+=['--sample-policy']
        if combined:command+=['--with-deliberation']
        stage(name,'flight',command,'{job}/collection/result.json',['{job}/collection/flights.json'],depends,peak=72,seconds=1200,
              inputs=[pack_path+'/checkpoints.json'])
    def trajectories(name,depends):
        destination='{round}/'+name
        stage(name,'cpu',[str(root/'envs/airsim/bin/python'),'{source}/trajectory_bundle.py',
            '--collection-root',str(root/'launches'),'--output',destination],destination+'/result.json',
            [destination+'/manifest.json'],depends,peak=8)
        return destination
    def update(name,module,dataset,depends,initialize=None):
        command=['python3','{source}/stage_worker.py','train-'+module,'--dataset',dataset,'--integration-only','--updates','1']
        if module=='policy':command+=['--goal-checkpoint','goal.pt']
        if initialize:command+=['--initialize-from',initialize]
        stage(name,'gpu',command,'{job}/training/result.json',['{job}/training/final.pt'],depends,peak=36,
              inputs=[dataset+'/manifest.json'])
        return job(name,'training/final.pt')
    if demonstrations:
        collection=str(demonstrations)
        stage('demonstrations-complete','cpu',['python3','{source}/await_demonstrations.py','--collection',collection],
            '{job}/demonstrations-ready.json',['{job}/demonstrations-ready.json'],peak=1,seconds=1800)
        bootstrap_dependencies=['demonstrations-complete']
    else:
        stage('bootstrap-flight','flight',['python3','{source}/collect_learning_round.py','--demonstration-batch',
            '--controller-checkpoints',str(visual_pack)],'{job}/collection/result.json',
            ['{job}/collection/flights.json'],peak=72,seconds=1800)
        collection=job('bootstrap-flight','collection/flights.json');bootstrap_dependencies=['bootstrap-flight']
    bundle=trajectories('trajectories',bootstrap_dependencies)
    stage('world-data','gpu',['python3','{source}/learning_data_job.py','--phase','world',
        '--bundle',bundle,'--collection',collection,'--checkpoints',str(visual_pack),'--output','{job}/world-data'],
        '{job}/world-data/result.json',['{job}/world-data/manifest.json'],['trajectories'])
    world_data=job('world-data','world-data');replay=job('world-data','world-data-replays/0')
    world=update('world-update','world',world_data,['world-data'])
    stage('policy-data','gpu',['python3','{source}/learning_data_job.py','--phase','policy',
        '--bundle',bundle,'--collection',collection,'--checkpoints',str(visual_pack),'--world',world,'--output','{job}/policy-data'],
        '{job}/policy-data/result.json',['{job}/policy-data/manifest.json'],['world-update'])
    policy_data=job('policy-data','policy-data')
    policy=update('policy-update','policy',policy_data,['policy-data']);config_data='{round}/configuration-data'
    stage('configuration-data','gpu',['python3','{source}/stage_worker.py','build-configurator-view','--dataset',bundle,
        '--collection',collection,'--runtime-replay',replay,'--artifact-output',config_data],
        config_data+'/result.json',[config_data+'/configurator.json'],['policy-update'],peak=12)
    stage('qwen-supervised-update','gpu',['python3','{source}/stage_worker.py','train-configurator','--dataset',config_data,
        '--integration-only','--updates','1'],'{job}/configurator/result.json',['{job}/configurator/latest.pt'],
        ['configuration-data'],peak=40)
    qwen=job('qwen-supervised-update','configurator/latest.pt')
    trained=pack('trained-pack',policy,world,qwen,['qwen-supervised-update'])
    flight('dagger-flight',trained,['trained-pack']);online=trajectories('online-trajectories',['dagger-flight'])
    dagger='{round}/dagger-data'
    stage('dagger-data','gpu',['python3','{source}/stage_worker.py','build-online-view','--learning-phase','dagger',
        '--dataset',online,'--navigation-pack',trained,'--world-checkpoint',world,
        '--collection',job('dagger-flight','collection/flights.json'),'--base-policy-data',policy_data,'--artifact-output',dagger],
        dagger+'/result.json',[dagger+'/dataset/manifest.json'],['online-trajectories'])
    corrected=update('dagger-update','policy',dagger+'/dataset',['dagger-data'],'initial-policy.pt')
    dagger_pack=pack('dagger-pack',corrected,world,qwen,['dagger-update'])
    flight('fresh-ppo-flight',dagger_pack,['dagger-pack'],sample=True);ppo='{round}/ppo-data'
    stage('ppo-data','gpu',['python3','{source}/stage_worker.py','build-online-view','--learning-phase','ppo',
        '--dataset',online,'--navigation-pack',dagger_pack,'--collection',job('fresh-ppo-flight','collection/flights.json'),
        '--artifact-output',ppo],ppo+'/result.json',[ppo+'/ppo.json'],['fresh-ppo-flight'],peak=16)
    stage('ppo-update','gpu',['python3','{source}/stage_worker.py','ppo','--dataset',ppo,'--goal-checkpoint','goal.pt',
        '--integration-only','--updates','1'],'{job}/ppo/result.json',['{job}/ppo/final.pt'],['ppo-data'],
        inputs=[ppo+'/ppo.json'])
    updated=pack('ppo-pack',job('ppo-update','ppo/final.pt'),world,qwen,['ppo-update'])
    flight('updated-policy-flight',updated,['ppo-pack'])
    flight('final-reload-flight',updated,['updated-policy-flight'],combined=True)
    stage('development-flights','flight',['python3','{source}/collect_learning_round.py','--development-batch',
        '--controller-checkpoints',updated],'{job}/collection/result.json',['{job}/collection/flights.json'],
        ['final-reload-flight'],peak=72,seconds=1800)
    stage('matched-configuration-flights','flight',['python3','{source}/collect_configuration_preferences.py',
        '--checkpoints',updated,'--qwen',qwen,'--goal-collection',collection,
        '--episode-id',episode],'{job}/configuration-preferences/result.json',['{job}/configuration-preferences/outcomes.json'],
        ['development-flights'],peak=72,seconds=7200)
    preference='{round}/preference-data'
    stage('preference-data','gpu',['python3','{source}/stage_worker.py','build-preference-view','--dataset',online,
        '--preference-outcomes',job('matched-configuration-flights','configuration-preferences/result.json'),
        '--supervised-adapter',qwen,'--artifact-output',preference],preference+'/result.json',
        [preference+'/configurator.json',preference+'/matched-pairs.jsonl'],['matched-configuration-flights'],peak=12)
    stage('qwen-preference-update','gpu',['python3','{source}/stage_worker.py','train-configurator','--dataset',preference,
        '--configurator-phase','preference','--initialize-from','supervised.pt','--integration-only','--updates','1'],
        '{job}/configurator/result.json',[],['preference-data'],peak=40)
    final=pack('preference-pack',job('ppo-update','ppo/final.pt'),world,job('qwen-preference-update','configurator/latest.pt'),
        ['qwen-preference-update'])
    flight('preference-updated-flight',final,['preference-pack'],combined=True)
    return dict(schema='training-dependencies/v1',scope='One real update through every eligible stage; acceptance never inferred',stages=stages)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--visual-pack',type=Path,required=True);parser.add_argument('--safety-profile',type=Path,required=True)
    parser.add_argument('--episode-id',default='train-00000');parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--demonstrations',type=Path)
    parser.add_argument('--hours',type=float,default=8);args=parser.parse_args()
    if not 0<args.hours<=8:parser.error('At most eight-hour resumable windows')
    spec=specification(args.visual_pack.resolve(),args.safety_profile.resolve(),args.episode_id,args.demonstrations)
    args.output.mkdir(parents=True,exist_ok=True);path=args.output/'programme-spec.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise ValueError('Changed cycle inputs require a new output round')
    if not path.exists():path.write_text(json.dumps(spec,indent=2))
    raise SystemExit(run(path,args.output/'execution',args.hours))
