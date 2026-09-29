"""Offline observed-volume task generation. No route or cost field enters inference.

Unknown voxels are never free. DepthPerspective receipts provide radial depth;
each admitted voxel has all eight corners supported by one eroded depth image.
The centre graph is eroded by a one metre cube including the camera mount.
"""
import argparse
from collections import Counter
import itertools
import math
from pathlib import Path
import numpy as np
from .common import read,write,digest

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
        if resolution!=.5:raise ValueError('This certificate requires half-metre voxels')
        self.resolution=resolution
        self.occupied={tuple(map(int,r)) for r in occupied}
        self.free={tuple(map(int,r)) for r in free}-self.occupied
        if not self.free:raise ValueError('No observed free volume')
        # A centred one-metre cube intersects +/-2 neighbours on each axis.
        points=np.asarray(list(self.free),dtype=np.int32);lo=points.min(0)-2;hi=points.max(0)+3
        if int(np.prod(hi-lo,dtype=np.int64))<=256_000_000:
            from scipy.ndimage import binary_erosion
            grid=np.zeros(tuple(hi-lo),dtype=bool);grid[tuple((points-lo).T)]=True
            eroded=binary_erosion(grid,structure=np.ones((5,5,5),bool))
            self.centres=set(map(tuple,np.argwhere(eroded)+lo))
        else:
            offsets=tuple(itertools.product(range(-2,3),repeat=3))
            self.centres={p for p in self.free if all(tuple(p[j]+v[j] for j in range(3)) in self.free for v in offsets)}
        if not self.centres:raise ValueError('No body-and-camera clearance after erosion')
        self._graph=None

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:return cls(d['free'],d['occupied'],float(d['resolution']))

    def position(self,key):return (np.asarray(key,dtype=float)*self.resolution).tolist()
    def key(self,position):return tuple(np.rint(np.asarray(position)/self.resolution).astype(int))

    def edge(self,a,b):
        # Diagonals require every vertex of the swept grid box: no corner cuts.
        return all(p in self.centres for p in itertools.product(*(range(min(x,y),max(x,y)+1) for x,y in zip(a,b))))

    def clear(self,a,b):
        pts=np.linspace(a,b,max(2,math.ceil(math.dist(a,b)/(.5/4))+1))
        keys=[self.key(p) for p in pts]
        return all(k in self.centres for k in keys) and all(self.edge(x,y) for x,y in zip(keys,keys[1:]))

    def costs(self,goal,zrange=None,max_nodes=2000000):
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
        end=self.key(goal)
        if end not in self.centres:raise ValueError('Goal lacks observed clearance')
        if self._graph is None:
            keys=sorted(self.centres)
            if len(keys)>max_nodes:raise ValueError('Survey graph exceeds bounded cost-field budget')
            index={k:i for i,k in enumerate(keys)};source=[];target=[];weights=[]
            # Six-connected swept edges cannot cut voxel corners. Construct once;
            # repeated goal fields use compiled sparse shortest paths.
            for i,key in enumerate(keys):
                for axis in range(3):
                    for sign in (-1,1):
                        neighbour=list(key);neighbour[axis]+=sign;j=index.get(tuple(neighbour))
                        if j is not None:source.append(i);target.append(j);weights.append(1. if axis==2 else 1/3)
            graph=csr_matrix((weights,(source,target)),shape=(len(keys),len(keys)))
            self._graph=(keys,index,graph)
        keys,index,graph=self._graph
        if zrange:
            selected=np.asarray([i for i,k in enumerate(keys) if zrange[0]<=k[2]*.5<=zrange[1]])
            subset=graph[selected][:,selected];local_keys=[keys[i] for i in selected];local_index={k:i for i,k in enumerate(local_keys)}
            if end not in local_index:raise ValueError('Goal outside altitude band')
            graph=subset;keys=local_keys;index=local_index
        distances,predecessors=dijkstra(graph,directed=False,indices=index[end],return_predecessors=True)
        valid=np.flatnonzero(np.isfinite(distances))
        return ({keys[i]:float(distances[i]) for i in valid},
            {keys[i]:keys[int(predecessors[i])] for i in valid if predecessors[i]>=0})

    def route(self,start,goal,parents):
        key=self.key(start);end=self.key(goal);path=[self.position(key)]
        while key!=end:
            if key not in parents:raise ValueError('Disconnected observed endpoints')
            key=parents[key];path.append(self.position(key))
        # Preserve the shortest graph route; simplify only through certified free edges.
        simplified=[path[0]];i=0
        while i<len(path)-1:
            j=i+1
            while j+1<len(path) and self.clear(path[i],path[j+1]):j+=1
            simplified.append(path[j]);i=j
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
    volume=ObservedVolume(free,occupied)
    np.savez_compressed(output,free=np.asarray(sorted(volume.free)),occupied=np.asarray(sorted(occupied)),resolution=.5)
    write(str(output)+'.json',dict(schema='ppo-observed-volume/v1',sources=sources,sha256=digest(output),
        radius_m=1,unknown_is_free=False,free_voxels=len(volume.free),clear_centres=len(volume.centres)))


def decisions(path):
    turns=0
    for a,b,c in zip(path,path[1:],path[2:]):
        x=np.asarray(b)-a;y=np.asarray(c)-b
        if min(np.linalg.norm(x[:2]),np.linalg.norm(y[:2]))<5:continue
        angle=math.degrees(math.acos(float(np.clip(np.dot(x[:2],y[:2])/(np.linalg.norm(x[:2])*np.linalg.norm(y[:2])),-1,1))))
        turns+=angle>=45
    return turns,max(p[2] for p in path)-min(p[2] for p in path)


def generate(field,output,scene_id,split,seed=0,per_cell=20,max_candidates=20000):
    volume=ObservedVolume.load(field);rng=np.random.default_rng(seed)
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    keys=sorted(volume.centres);counts=Counter();tasks=[];rejected=Counter();seen=set()
    for index in range(max_candidates):
        if all(counts[(d,c)]>=per_cell for d in DISTANCES for c in DIFFICULTIES):break
        start,goal=[volume.position(keys[i]) for i in rng.choice(len(keys),2,replace=False)]
        d=distance_bin(start,goal)
        if d is None:continue
        pair=tuple(sorted((volume.key(start),volume.key(goal))))
        if pair in seen:continue
        seen.add(pair)
        try:
            costs,parents=volume.costs(goal);route=volume.route(start,goal,parents)
            length=sum(math.dist(a,b) for a,b in zip(route,route[1:]));turns,height=decisions(route)
            if length>600:raise ValueError('reference_over_600m')
            reference=nominal_time(route);timeout=max(120.,2*reference+30)
            if timeout>900:raise ValueError('timeout_over_900s')
            direct=volume.clear(start,goal);alternative=None
            category='direct' if direct else 'multiple_decisions' if turns>=2 else 'detour' if turns>=1 else None
            if height>=4 and abs(start[2]-goal[2])<=2:
                lo=min(start[2],goal[2])-1;hi=max(start[2],goal[2])+1
                _,low_parents=volume.costs(goal,(lo,hi))
                try:alternative=volume.route(start,goal,low_parents)
                except ValueError:pass
                if alternative and abs(nominal_time(alternative)-reference)>1:category='altitude_alternatives'
            if category is None:raise ValueError('no_measured_decision_class')
            if counts[(d,category)]>=per_cell:continue
            ident=f'{scene_id}-{seed}-{index:06d}';costpath=root/(ident+'-costs.npz')
            np.savez_compressed(costpath,keys=np.asarray(list(costs)),seconds=np.asarray(list(costs.values())))
            # Reversals and altitude variants share an undirected endpoint-region ID.
            regions=sorted(tuple(np.floor(np.asarray(p)[:2]/10).astype(int)) for p in (start,goal))
            group=':'.join(','.join(map(str,r)) for r in regions)
            tasks.append(dict(id=ident,scene_id=scene_id,split=split,kind='mission',start=start,goal=goal,
                start_yaw_deg=float(rng.uniform(-180,180)),goal_yaw_deg=float(rng.uniform(-180,180)),
                distance_bin=d,difficulty=category,turns=turns,vertical_range_m=height,reference_length_m=length,
                reference_path=route,alternative_path=alternative,reference_s=reference,timeout_s=timeout,
                start_region=list(regions[0] if tuple(np.floor(np.asarray(start)[:2]/10).astype(int))==regions[0] else regions[1]),
                group_id=group,field=str(Path(field).resolve()),field_sha256=digest(field),
                cost_field=str(costpath.resolve()),cost_sha256=digest(costpath),
                bounds=[(np.min(np.asarray(keys),axis=0)*.5).tolist(),(np.max(np.asarray(keys),axis=0)*.5).tolist()],
                start_yaw_sampling='uniform_360_independent',goal_yaw_sampling='uniform_360_independent'))
            counts[(d,category)]+=1
        except ValueError as error:rejected[str(error)]+=1
    write(root/'candidates.json',dict(schema='ppo-task-candidates/v2',tasks=tasks,seed=seed,
        coverage={f'{d}/{c}':counts[(d,c)] for d in DISTANCES for c in DIFFICULTIES},rejected=dict(rejected),
        training_qualified=False,missing='endpoint images, physical route checks, reviewed geography and scene qualification'))


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
    a.add_argument('--seed',type=int,default=0);a.add_argument('--per-cell',type=int,default=20);a.add_argument('--max-candidates',type=int,default=20000)
    args=p.parse_args()
    if args.stage=='fuse':fuse(read(args.receipts)['receipts'],args.output)
    else:generate(args.field,args.output,args.scene_id,args.split,args.seed,args.per_cell,args.max_candidates)
