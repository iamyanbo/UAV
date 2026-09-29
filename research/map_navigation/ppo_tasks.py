"""Offline observed-volume task generation. No route or cost field enters inference.

Unknown voxels are never free. DepthPerspective receipts provide radial depth;
each admitted voxel has all eight corners supported by one eroded depth image.
The centre graph is eroded by a one metre cube including the camera mount.
"""
import argparse
from collections import Counter
import itertools
import math
import time
import shutil
from pathlib import Path
import numpy as np
from .common import read,write,digest
from .ppo_geometry import VoxelIndex,CostField,ParentField,GeometryCapacityError,admit_bytes,build_graph,save_costs

DIFFICULTIES=('direct','detour','multiple_decisions','altitude_alternatives')
DISTANCES=('40-100','100-200','200-300')
NEIGHBORS=tuple(v for v in itertools.product((-1,0,1),repeat=3) if any(v))


def distance_bin(a,b):
    d=math.dist(a,b)
    return '40-100' if 40<=d<100 else '100-200' if 100<=d<200 else '200-300' if 200<=d<=300 else None


def nominal_time(path):
    return sum(max(math.dist(a[:2],b[:2])/1.5,abs(a[2]-b[2])/.5) for a,b in zip(path,path[1:]))


class ObservedVolume:
    def __init__(self,free,occupied,resolution=.5):
        from scipy.ndimage import binary_erosion
        if resolution!=.5:raise ValueError('This certificate requires half-metre voxels')
        self.resolution=resolution
        points=np.asarray(list(free) if isinstance(free,set) else free,dtype=np.int32).reshape(-1,3)
        self.occupied=np.asarray(list(occupied) if isinstance(occupied,set) else occupied,dtype=np.int32).reshape(-1,3)
        if not len(points):raise ValueError('No observed free volume')
        lo=points.min(0).astype(np.int64)-2;hi=points.max(0).astype(np.int64)+3
        shape=hi-lo;cells=int(np.prod(shape,dtype=np.int64))
        # Account for dense erosion buffers and coordinate/index construction.
        # Admission is memory-based, not an arbitrary two-million-node ceiling.
        admit_bytes(cells*3+len(points)*80+len(self.occupied)*24)
        grid=np.zeros(tuple(shape),dtype=bool);grid[tuple((points-lo).T)]=True
        occupied_local=self.occupied.astype(np.int64)-lo
        inside=((occupied_local>=0)&(occupied_local<shape)).all(1)
        grid[tuple(occupied_local[inside].T)]=False
        self.free=(np.argwhere(grid)+lo).astype(np.int32)
        eroded=binary_erosion(grid,structure=np.ones((5,5,5),bool))
        keys=(np.argwhere(eroded)+lo).astype(np.int32)
        if not len(keys):raise ValueError('No body-and-camera clearance after erosion')
        self.centres=VoxelIndex(keys)
        self._graph=None;self._components=None;self._component_order=None;self._component_offsets=None

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:return cls(d['free'],d['occupied'],float(d['resolution']))

    def position(self,key):return (np.asarray(key,dtype=float)*self.resolution).tolist()
    def key(self,position):return tuple(np.rint(np.asarray(position)/self.resolution).astype(int))

    def edge(self,a,b):
        keys=list(itertools.product(*(range(min(x,y),max(x,y)+1) for x,y in zip(a,b))))
        return bool((self.centres.find_many(keys)>=0).all())

    def clear(self,a,b):
        pts=np.linspace(a,b,max(2,math.ceil(math.dist(a,b)/(.5/4))+1))
        keys=np.rint(pts/.5).astype(np.int32)
        if (self.centres.find_many(keys)<0).any():return False
        # Consecutive samples differ by at most one cell on each axis. Every
        # mixed vertex of their swept grid box must retain eroded support.
        lo=np.minimum(keys[:-1],keys[1:]);hi=np.maximum(keys[:-1],keys[1:])
        for corner in itertools.product((0,1),repeat=3):
            if (self.centres.find_many(np.where(corner,hi,lo))<0).any():return False
        return True

    def prepare_graph(self):
        if self._graph is None:self._graph=build_graph(self.centres)
        return self._graph

    def component_nodes(self,goal):
        from scipy.sparse.csgraph import connected_components
        if self._components is None:
            _,self._components=connected_components(self.prepare_graph(),directed=False)
            self._component_order=np.argsort(self._components,kind='stable')
            self._component_offsets=np.r_[0,np.cumsum(np.bincount(self._components))]
        i=self.centres.find(self.key(goal))
        if i<0:raise ValueError('Goal lacks observed clearance')
        label=self._components[i]
        return self._component_order[self._component_offsets[label]:self._component_offsets[label+1]]

    def costs(self,goal,zrange=None):
        from scipy.sparse.csgraph import dijkstra
        index=self.centres;graph=self.prepare_graph()
        if zrange is not None:
            selected=np.flatnonzero((index.keys[:,2]*.5>=zrange[0])&(index.keys[:,2]*.5<=zrange[1]))
            if not len(selected):raise ValueError('Empty altitude band')
            graph=graph[selected][:,selected];index=VoxelIndex(index.keys[selected])
        end=index.find(self.key(goal))
        if end<0:raise ValueError('Goal lacks observed clearance in requested altitude band')
        distances,parents=dijkstra(graph,directed=True,indices=end,return_predecessors=True)
        return CostField(index,distances),ParentField(index,parents)

    def route(self,start,goal,parents):
        key=self.key(start);end=self.key(goal);path=[self.position(key)]
        while key!=end:
            try:key=parents[key]
            except KeyError:raise CandidateRejected('Disconnected observed endpoints') from None
            path.append(self.position(key))
        # Probe long shortcuts exponentially, then bisect. Each accepted segment
        # is checked against the unchanged full-resolution swept-volume rule.
        simplified=[path[0]];i=0
        while i<len(path)-1:
            good=i+1;step=2;bad=len(path)
            while i+step<len(path):
                probe=i+step
                if not self.clear(path[i],path[probe]):bad=probe;break
                good=probe;step*=2
            if bad==len(path):
                if self.clear(path[i],path[-1]):good=len(path)-1
                else:bad=len(path)-1
            while bad-good>1:
                probe=(good+bad)//2
                if self.clear(path[i],path[probe]):good=probe
                else:bad=probe
            simplified.append(path[good]);i=good
        return simplified

    def potential(self,position,costs,reference_s):
        key=self.key(position)
        if key not in costs or not self.clear(position,self.position(key)):
            raise LookupError('Position left the supported cost field')
        return -min(float(costs[key])/max(reference_s,1.),2.)


def fuse(receipts,output):
    """Offline only. All corners, finite radial depths and a pixel erosion required."""
    from scipy.ndimage import minimum_filter
    free=set();occupied=set();sources=[]
    corner=np.asarray(list(itertools.product((-.25,.25),repeat=3)))
    for receipt in receipts:
        r=read(receipt)
        if r['depth_type']!='DepthPerspective' or digest(r['path'])!=r['sha256']:raise ValueError('Unverified radial depth')
        depth=np.load(r['path'],allow_pickle=False);t=np.asarray(r['camera_to_ned'])
        depth=np.where(np.isfinite(depth)&(depth>.2)&(depth<150),depth,0.)
        eroded=minimum_filter(depth,size=3,mode='constant',cval=0)
        v,u=np.mgrid[0:r['height']:2,0:r['width']:2]
        rays=np.stack(((u-r['cx'])/r['fx'],(v-r['cy'])/r['fx'],np.ones_like(u)),-1).reshape(-1,3)
        rays/=np.linalg.norm(rays,axis=1,keepdims=True)
        lengths=depth[v,u].reshape(-1);valid=lengths>.2
        rays=rays[valid];lengths=lengths[valid]
        world=rays@t[:3,:3].T
        ends=t[:3,3]+world*lengths[:,None]
        occupied.update(map(tuple,np.rint(ends/.5).astype(int)))
        proposed=set()
        for direction,length in zip(world,lengths):
            points=t[:3,3]+np.arange(.5,max(.5,length-.5),.25)[:,None]*direction
            proposed.update(map(tuple,np.rint(points/.5).astype(int)))
        keys=np.asarray(sorted(proposed),dtype=int).reshape(-1,3)
        for begin in range(0,len(keys),8192):
            k=keys[begin:begin+8192];points=k[:,None,:]*.5+corner
            optical=(points-t[:3,3])@t[:3,:3]
            z=optical[...,2];safe=np.maximum(z,.001)
            px=np.rint(r['fx']*optical[...,0]/safe+r['cx']).astype(int)
            py=np.rint(r['fx']*optical[...,1]/safe+r['cy']).astype(int)
            inside=(z>.2)&(px>=1)&(px<r['width']-1)&(py>=1)&(py<r['height']-1)
            observed=eroded[np.clip(py,0,r['height']-1),np.clip(px,0,r['width']-1)]
            supported=(inside&(np.linalg.norm(optical,axis=-1)<observed-.35)).all(axis=1)
            free.update(map(tuple,k[supported]))
        sources.append(dict(path=str(Path(receipt).resolve()),sha256=digest(receipt)))
        write(str(output)+'.progress.json',dict(stage='fusion',processed=len(sources),total=len(receipts),
            free_voxels=len(free),occupied_voxels=len(occupied),complete=False))
    volume=ObservedVolume(free,occupied)
    np.savez_compressed(output,free=volume.free,occupied=volume.occupied,resolution=.5)
    write(str(output)+'.json',dict(schema='ppo-observed-volume/v1',sources=sources,sha256=digest(output),
        radius_m=1,unknown_is_free=False,free_voxels=len(volume.free),clear_centres=len(volume.centres)))
    write(str(output)+'.progress.json',dict(stage='fusion',processed=len(sources),total=len(receipts),complete=True,
        free_voxels=len(volume.free),clear_centres=len(volume.centres)))


def decisions(path):
    turns=0
    for a,b,c in zip(path,path[1:],path[2:]):
        x=np.asarray(b)-a;y=np.asarray(c)-b
        if min(np.linalg.norm(x[:2]),np.linalg.norm(y[:2]))<5:continue
        angle=math.degrees(math.acos(float(np.clip(np.dot(x[:2],y[:2])/(np.linalg.norm(x[:2])*np.linalg.norm(y[:2])),-1,1))))
        turns+=angle>=45
    return turns,max(p[2] for p in path)-min(p[2] for p in path)


class CandidateRejected(ValueError):
    """An expected endpoint/route rejection; infrastructure exceptions propagate."""


def generate(field,output,scene_id,split,seed=0,per_cell=20,max_candidates=20000,hours=2):
    if not 0<hours<=8 or per_cell<1 or max_candidates<1:raise ValueError('Invalid generation bounds')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();deadline=started+hours*3600;rng=np.random.default_rng(seed)
    counts=Counter();tasks=[];rejected=Counter();seen=set();attempted=0;goals_solved=0
    stage='loading_geometry';status='running';graph_info={};goal=None;costs=parents=None;accepted_goal=0
    def save_candidates(complete=False):
        write(root/'candidates.json',dict(schema='ppo-task-candidates/v2',tasks=tasks,seed=seed,
            coverage={f'{d}/{c}':counts[(d,c)] for d in DISTANCES for c in DIFFICULTIES},rejected=dict(rejected),
            generation_complete=complete,status=status,stage=stage,attempted_pairs=attempted,
            goal_fields_solved=goals_solved,elapsed_s=time.monotonic()-started,graph=graph_info,
            sampling='uniform component-conditioned endpoints; up to 32 pairs and 3 tasks per goal',
            training_qualified=False,missing='endpoint images, physical route checks, reviewed geography and scene qualification'))
    def check_deadline():
        if time.monotonic()>=deadline:raise TimeoutError('Task generation window expired; partial candidates preserved')
    try:
        save_candidates()
        volume=ObservedVolume.load(field);keys=volume.centres.keys
        bounds=[(keys.min(0)*.5).tolist(),(keys.max(0)*.5).tolist()];field_sha=digest(field)
        stage='building_graph';save_candidates();check_deadline()
        graph=volume.prepare_graph()
        graph_info.update(nodes=len(keys),directed_edges=graph.nnz,
            csr_bytes=graph.data.nbytes+graph.indices.nbytes+graph.indptr.nbytes,
            voxel_resolution_m=.5,clearance_half_extent_m=1,construction_s=time.monotonic()-started)
        stage='sampling_routes';save_candidates()
        for index in range(max_candidates):
            check_deadline();attempted=index+1
            if all(counts[(d,c)]>=per_cell for d in DISTANCES for c in DIFFICULTIES):break
            if index%32==0 or goal is None or accepted_goal>=3:
                goal=volume.position(keys[int(rng.integers(len(keys)))])
                connected=volume.component_nodes(goal);costs=parents=None;cost_receipt=None;accepted_goal=0
                # Skip components whose extent cannot contain even a 40 m pair.
                pts=keys[connected]
                if np.linalg.norm((pts.max(0)-pts.min(0))*.5)<40:
                    rejected['component_extent_under_40m']+=1;goal=None;continue
            start=volume.position(keys[int(connected[int(rng.integers(len(connected)))])])
            d=distance_bin(start,goal)
            if d is None:rejected['outside_distance_bins']+=1;continue
            pair=tuple(sorted((volume.key(start),volume.key(goal))))
            if pair in seen:rejected['duplicate_endpoint_pair']+=1;continue
            seen.add(pair)
            try:
                direct=volume.clear(start,goal)
                if direct and counts[(d,'direct')]>=per_cell:
                    raise CandidateRejected('direct_cell_full')
                if costs is None:
                    stage='solving_goal_field';save_candidates()
                    costs,parents=volume.costs(goal);goals_solved+=1
                    check_deadline();stage='sampling_routes';save_candidates()
                route=[start,goal] if direct else volume.route(start,goal,parents)
                length=sum(math.dist(a,b) for a,b in zip(route,route[1:]));turns,height=decisions(route)
                if length>600:raise CandidateRejected('reference_over_600m')
                reference=nominal_time(route);timeout=max(120.,2*reference+30)
                if timeout>900:raise CandidateRejected('timeout_over_900s')
                alternative=None
                category='direct' if direct else 'multiple_decisions' if turns>=2 else 'detour' if turns>=1 else None
                if not direct and height>=4 and abs(start[2]-goal[2])<=2:
                    lo=min(start[2],goal[2])-1;hi=max(start[2],goal[2])+1
                    _,low_parents=volume.costs(goal,(lo,hi))
                    try:alternative=volume.route(start,goal,low_parents)
                    except CandidateRejected:pass
                    if alternative and abs(nominal_time(alternative)-reference)>1:category='altitude_alternatives'
                if category is None:raise CandidateRejected('no_measured_decision_class')
                if counts[(d,category)]>=per_cell:raise CandidateRejected('difficulty_cell_full')
                check_deadline()
                ident=f'{scene_id}-{seed}-{index:06d}'
                if cost_receipt is None:
                    stage='saving_goal_field';save_candidates()
                    costpath=root/(ident+'-costs.npz')
                    # Uncompressed payload plus temporary archive must fit before writing.
                    if shutil.disk_usage(root).free<len(costs)*32:
                        raise GeometryCapacityError('Insufficient disk for goal cost field')
                    save_costs(costpath,costs)
                    cost_receipt=dict(cost_field=str(costpath.resolve()),cost_sha256=digest(costpath))
                    stage='sampling_routes'
                regions=sorted(tuple(np.floor(np.asarray(p)[:2]/10).astype(int)) for p in (start,goal))
                group=':'.join(','.join(map(str,r)) for r in regions)
                tasks.append(dict(id=ident,scene_id=scene_id,split=split,kind='mission',start=start,goal=goal,
                    start_yaw_deg=float(rng.uniform(-180,180)),goal_yaw_deg=float(rng.uniform(-180,180)),
                    distance_bin=d,difficulty=category,turns=turns,vertical_range_m=height,reference_length_m=length,
                    reference_path=route,alternative_path=alternative,reference_s=reference,timeout_s=timeout,
                    start_region=np.floor(np.asarray(start)[:2]/10).astype(int).tolist(),group_id=group,
                    field=str(Path(field).resolve()),field_sha256=field_sha,bounds=bounds,**cost_receipt,
                    start_yaw_sampling='uniform_360_independent',goal_yaw_sampling='uniform_360_independent'))
                counts[(d,category)]+=1;accepted_goal+=1;save_candidates()
            except CandidateRejected as error:rejected[str(error)]+=1
            if index%32==0:save_candidates()
        coverage_complete=all(counts[(d,c)]>=per_cell for d in DISTANCES for c in DIFFICULTIES)
        status='coverage_complete' if coverage_complete else 'candidate_budget_exhausted'
        stage='finished';save_candidates(complete=True)
    except BaseException as error:
        status='deadline' if isinstance(error,(TimeoutError,KeyboardInterrupt)) else 'failed'
        write(root/'failure.json',dict(type=type(error).__name__,reason=str(error),stage=stage,
            scope='scene generation; never retried as an endpoint rejection',attempted_pairs=attempted))
        save_candidates();raise


def validate_tasks(manifest,cfg):
    """Fail closed on coverage, independent geography, evidence and sealed splits."""
    if manifest.get('schema')!='photo-map-ppo-tasks/v2':raise ValueError('New task schema required; corridors cannot migrate')
    scenes={s['scene_id']:s for s in manifest['scenes']};groups={};ids=set()
    for scene in scenes.values():
        if scene['split'] not in ('train','validation'):raise ValueError('Sealed test geography cannot enter pilot')
        for key in ('qualification','geography_review'):
            if digest(scene[key])!=scene[key+'_sha256']:raise ValueError('Changed scene evidence')
        q=read(scene['qualification']);review=read(scene['geography_review'])
        if not q.get('qualified') or not all(q.get('checks',{}).get(k) for k in ('camera','geometry','motion','collision','stop','stale_frame','timing')):
            raise ValueError('Missing physical scene qualification')
        if not review.get('independent') or not review.get('reviewer') or not review.get('notes'):raise ValueError('Geography not independently reviewed')
        group=review['geography_id']
        if group in groups and groups[group]!=scene['split']:raise ValueError('Geography leakage')
        groups[group]=scene['split']
        if q['config']['camera']!=cfg['camera']:raise ValueError('Camera qualification differs')
        if len(q.get('resets',[]))<20 or not all(r[-1]['passed'] for r in q['resets']):raise ValueError('Twenty qualified resets required')
        if len(q.get('camera_attitudes',[]))!=9 or not all(r['pose_agreement'] for r in q['camera_attitudes']):raise ValueError('Nine camera attitudes required')
        if q['scene_sha256']!=digest(scene['descriptor']):raise ValueError('Changed scene descriptor')
        for asset in read(scene['descriptor'])['assets']:
            if digest(asset['path'])!=asset['sha256']:raise ValueError('Changed scene asset')
        routes=scene['physical_routes']
        if len(routes)<12:raise ValueError('Twelve representative physical routes required per scene')
        cells=set()
        for receipt in routes:
            if digest(receipt['path'])!=receipt['sha256']:raise ValueError('Changed route qualification')
            r=read(receipt['path'])
            if not r.get('passed'):raise ValueError('Failed route qualification')
            cells.add((r['distance_bin'],r['difficulty']))
        if cells!={(d,c) for d in DISTANCES for c in DIFFICULTIES}:raise ValueError('Physical checks lack difficulty/distance coverage')
    for split,minimum in [('train',cfg['minimum_train_scenes']),('validation',cfg['minimum_validation_scenes'])]:
        if sum(s==split for s in groups.values())<minimum:raise ValueError('Insufficient independent '+split+' geography')
    fields={}
    for t in manifest['tasks']:
        if t['id'] in ids:raise ValueError('Duplicate task')
        ids.add(t['id']);scene=scenes[t['scene_id']]
        if t['split']!=scene['split'] or t['camera']!=cfg['camera']:raise ValueError('Task split/camera mismatch')
        for name in ('goal_image','endpoint_evidence','field','cost_field'):
            sha={'goal_image':'goal_sha256','endpoint_evidence':'endpoint_sha256','field':'field_sha256','cost_field':'cost_sha256'}[name]
            if digest(t[name])!=t[sha]:raise ValueError('Task evidence changed: '+name)
        endpoint=read(t['endpoint_evidence'])
        if not endpoint.get('passed') or endpoint['start']!=t['start'] or endpoint['goal']!=t['goal']:raise ValueError('Endpoint qualification mismatch')
        if endpoint['camera']!=cfg['camera'] or endpoint['start_yaw_deg']!=t['start_yaw_deg'] or endpoint['goal_yaw_deg']!=t['goal_yaw_deg']:raise ValueError('Endpoint camera/orientation changed')
        if not np.isfinite(t['start']+t['goal']+[t['start_yaw_deg'],t['goal_yaw_deg'],t['reference_s'],t['timeout_s']]).all():raise ValueError('Nonfinite task')
        volume=fields.setdefault(t['field'],None)
        if volume is None:volume=fields[t['field']]=ObservedVolume.load(t['field'])
        with np.load(t['cost_field'],allow_pickle=False) as costs:
            goal_key=volume.key(t['goal']);matches=np.all(costs['keys']==goal_key,axis=1)
            if matches.sum()!=1 or costs['seconds'][matches][0]!=0 or not np.isfinite(costs['seconds']).all() or (costs['seconds']<0).any():
                raise ValueError('Invalid goal-bound cost field')
        if t.get('kind','mission')=='mission':
            if distance_bin(t['start'],t['goal'])!=t['distance_bin']:raise ValueError('Distance label mismatch')
            if t['start_yaw_sampling']!='uniform_360_independent':raise ValueError('Start heading biased toward goal')
            route=t['reference_path']
            if route[0]!=t['start'] or route[-1]!=t['goal'] or not all(volume.clear(a,b) for a,b in zip(route,route[1:])):raise ValueError('Unsupported reference route')
            length=sum(math.dist(a,b) for a,b in zip(route,route[1:]))
            reference=nominal_time(route)
            if length>600 or abs(reference-t['reference_s'])>1e-6 or abs(t['timeout_s']-max(120,2*reference+30))>1e-6:
                raise ValueError('Reference length/time labels disagree with route')
            turns,height=decisions(route)
            if t['difficulty'] not in DIFFICULTIES:raise ValueError('Unknown decision class')
            direct=volume.clear(t['start'],t['goal'])
            if (t['difficulty']=='direct' and not direct) or (t['difficulty'] in ('detour','multiple_decisions') and direct):raise ValueError('Artificial route turns are not task difficulty')
            if t['difficulty']=='detour' and turns<1 or t['difficulty']=='multiple_decisions' and turns<2:raise ValueError('Unsubstantiated difficulty')
            if t['difficulty']=='altitude_alternatives':
                alternative=t['alternative_path']
                if (height<4 or not alternative or alternative[0]!=t['start'] or alternative[-1]!=t['goal'] or
                    not all(volume.clear(a,b) for a,b in zip(alternative,alternative[1:]))):raise ValueError('Unsubstantiated altitude choice')
            if not 120<=t['timeout_s']<=900:raise ValueError('Invalid mission horizon')
        elif t['kind'] not in ('execution','arrival'):raise ValueError('Unknown auxiliary stream')
        elif not volume.clear(t['start'],t['goal']):raise ValueError('Unsupported auxiliary execution')
        if t.get('kind')=='execution' and ('exercise' not in t or 'reference_depth' not in endpoint):raise ValueError('Execution target lacks visible evidence')
    for ident,scene in scenes.items():
        rows=[t for t in manifest['tasks'] if t['scene_id']==ident and t.get('kind','mission')=='mission']
        counts=Counter((t['distance_bin'],t['difficulty']) for t in rows)
        minimum=cfg['qualification']['train_tasks_per_cell' if scene['split']=='train' else 'validation_tasks_per_cell']
        if any(counts[(d,c)]<minimum for d in DISTANCES for c in DIFFICULTIES):raise ValueError('Task coverage incomplete: '+str(ident))
        if scene['split']=='train':
            if len({tuple(t['start_region']) for t in rows})<60 or len({tuple(t['goal']) for t in rows})<60:raise ValueError('Too few distinct starts/goals')
            kinds={t.get('kind','mission') for t in manifest['tasks'] if t['scene_id']==ident}
            if not {'execution','arrival'}<=kinds:raise ValueError('Missing execution/arrival exercises')
    return scenes


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='stage',required=True)
    a=sub.add_parser('fuse');a.add_argument('--receipts',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('generate');a.add_argument('--field',required=True);a.add_argument('--output',required=True)
    a.add_argument('--scene-id',required=True);a.add_argument('--split',choices=['train','validation'],required=True)
    a.add_argument('--hours',type=float,default=2)
    a.add_argument('--seed',type=int,default=0);a.add_argument('--per-cell',type=int,default=20);a.add_argument('--max-candidates',type=int,default=20000)
    args=p.parse_args()
    if args.stage=='fuse':fuse(read(args.receipts)['receipts'],args.output)
    else:generate(args.field,args.output,args.scene_id,args.split,args.seed,args.per_cell,args.max_candidates,args.hours)
