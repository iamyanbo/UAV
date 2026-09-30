"""Private native occupancy, conservative vehicle clearance and route labels."""
from collections import OrderedDict
import itertools
import math
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter
from .common import read, write, digest, contained, FlightLock
from .mission_contracts import identity, city_config
from .mission_resources import RunWindow, Resources
from .mission_space import reserve_write


class RouteUnavailable(RuntimeError):pass


class Geometry:
    def __init__(self,manifest):
        self.path=Path(manifest).resolve();self.spec=read(self.path)
        if self.spec.get('schema')!='photo-goal-private-geometry/v1' or not self.spec.get('qualified'):
            raise ValueError('Measured native geometry qualification required')
        receipt=contained(self.path.parent,self.spec['qualification']['path'])
        if digest(receipt)!=self.spec['qualification']['sha256']:raise ValueError('Geometry receipt changed')
        self.origin=np.asarray(self.spec['origin'],float);self.res=float(self.spec['resolution_m'])
        shape=tuple(self.spec['shape']);occupied=np.ones(shape,dtype=bool)
        for tile in self.spec['tiles']:
            path=contained(self.path.parent,tile['path'])
            if digest(path)!=tile['sha256']:raise ValueError('Geometry tile changed')
            data=np.load(path,allow_pickle=False)
            start=np.asarray(tile['grid_start'],int);stop=start+np.asarray(data.shape)
            occupied[tuple(slice(a,b) for a,b in zip(start,stop))]=data
        # A whole voxel, not only its center, must clear the bounding vehicle box.
        # This conservative dilation also protects diagonal edges from corner cuts.
        radii=np.asarray([1.,1.,.65])+.5*self.res
        widths=2*np.ceil(radii/self.res).astype(int)+1
        self.clear=~maximum_filter(occupied,size=tuple(widths),mode='constant',cval=1)
        self.cache=OrderedDict();self.cache_bytes=0

    def point(self,index):return self.origin+(np.asarray(index)+.5)*self.res

    def index(self,point):return np.floor((np.asarray(point)-self.origin)/self.res).astype(int)

    def safe(self,point,bounds=None):
        point=np.asarray(point)
        if bounds is not None and (np.any(point<bounds[0]) or np.any(point>bounds[1])):return False
        idx=self.index(point)
        return bool(np.all(idx>=0) and np.all(idx<self.clear.shape) and self.clear[tuple(idx)])

    def swept(self,a,b,bounds=None):
        count=max(2,math.ceil(math.dist(a,b)/(.25*self.res))+1)
        return all(self.safe(p,bounds) for p in np.linspace(a,b,count))

    def connectors(self,point,bounds):
        if not self.safe(point,bounds):return []
        center=self.index(point);result=[]
        for delta in itertools.product(range(-2,3),repeat=3):
            idx=center+delta;p=self.point(idx);distance=math.dist(point,p)
            if distance<=2. and self.swept(point,p,bounds):result.append((tuple(idx),distance))
        return result

    def prepare_field(self,task):
        from skimage.graph import MCP_Geometric
        bounds=np.asarray(task['bounds'],float);goal=np.asarray(task['goal'],float)
        key=identity(dict(geometry=digest(self.path),goal=goal.tolist(),bounds=bounds.tolist()))
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
        lo=np.maximum(self.index(bounds[0]),0);hi=np.minimum(self.index(bounds[1])+1,self.clear.shape)
        if np.any(hi<=lo):raise RouteUnavailable('Empty task field')
        free=self.clear[tuple(slice(a,b) for a,b in zip(lo,hi))].copy()
        for axis in range(3):
            centers=self.origin[axis]+(np.arange(lo[axis],hi[axis])+.5)*self.res
            outside=(centers<bounds[0,axis])|(centers>bounds[1,axis])
            selection=[slice(None)]*3;selection[axis]=outside
            free[tuple(selection)]=False
        starts=[]
        for idx,_ in self.connectors(goal,bounds):
            starts.append(tuple(np.asarray(idx)-lo))
        # Multi-source arrival position region; heading/speed are separate labels.
        center=self.index(goal)
        for delta in itertools.product(range(-4,5),range(-4,5),range(-3,4)):
            idx=center+delta;p=self.point(idx)
            if np.linalg.norm((p-goal)[:2])<=3 and abs(p[2]-goal[2])<=2 and self.safe(p,bounds):
                starts.append(tuple(idx-lo))
        starts=list(set(s for s in starts if all(0<=x<n for x,n in zip(s,free.shape))))
        if not starts:raise RouteUnavailable('No clear goal-region vertices')
        mcp=MCP_Geometric(np.where(free,1.,np.inf),fully_connected=True,sampling=(self.res,)*3)
        costs,_=mcp.find_costs(starts)
        result=dict(id=key,costs=costs,origin=lo,bounds=bounds,mcp=mcp,goal=goal)
        # Include traceback/workspace conservatively in admission, not just costs.
        size=int(costs.nbytes*8)
        if size>2*2**30:raise RouteUnavailable('Route field exceeds 2 GiB cache budget')
        while self.cache and (len(self.cache)>=8 or self.cache_bytes+size>2*2**30):
            _,old=self.cache.popitem(last=False);self.cache_bytes-=old['bytes']
        result['bytes']=size;self.cache[key]=result;self.cache_bytes+=size
        return result

    def distance(self,field,point):
        delta=np.asarray(point)-field['goal']
        if np.linalg.norm(delta[:2])<=3 and abs(delta[2])<=2 and self.safe(point,field['bounds']):return 0.
        values=[]
        for idx,length in self.connectors(point,field['bounds']):
            local=np.asarray(idx)-field['origin']
            if all(0<=x<n for x,n in zip(local,field['costs'].shape)):
                distance=float(field['costs'][tuple(local)])+length
                if math.isfinite(distance):values.append((distance,tuple(local)))
        if not values:raise RouteUnavailable('Nonterminal position has no connected route label')
        return min(values)[0]

    def validate_pair(self,task):
        field=self.prepare_field(task)
        distance=self.distance(field,task['start'])
        if distance>600:raise RouteUnavailable('Route exceeds 600 m')
        connectors=self.connectors(task['start'],field['bounds'])
        best=min(connectors,key=lambda row: float(field['costs'][tuple(np.asarray(row[0])-field['origin'])])+row[1])
        path=field['mcp'].traceback(tuple(np.asarray(best[0])-field['origin']))
        points=[task['start']]+[self.point(np.asarray(p)+field['origin']).tolist() for p in reversed(path)]
        if not all(self.swept(a,b,field['bounds']) for a,b in zip(points,points[1:])):
            raise RouteUnavailable('Swept route clearance failed')
        seconds=sum(max(math.dist(a[:2],b[:2])/3,abs(a[2]-b[2])) for a,b in zip(points,points[1:]))+4
        if seconds>.8*task['timeout_s']:raise RouteUnavailable('Route cannot fit mission deadline')
        return dict(geometry_sha256=digest(self.path),field_id=field['id'],route_length_m=distance,
                    optimistic_route_s=seconds,swept_clear=True)


def acquire(args):
    from .mission_environment import CityEnvironment
    from projectairsim.types import Pose
    root=Path(args.root).resolve();scene=read(args.scene);cfg=city_config(args.config)
    q=read(args.qualification)
    required=('axis_verified','scale_verified','occupancy_verified','surface_verified','contact_verified')
    if not all(q.get(k) is True for k in required) or q.get('scene_sha256')!=digest(args.scene):
        raise ValueError('Actual axis/surface/contact evidence for this native scene required')
    for evidence in q.get('evidence',[]):
        if digest(evidence['path'])!=evidence['sha256']:raise ValueError('Geometry evidence changed')
    if not q.get('evidence'):raise ValueError('Geometry qualification needs native recordings')
    axes=q.get('array_axes');order=q.get('array_order');surface=q.get('surface_units')
    if sorted(axes or [])!=['x','y','z'] or order not in ('C','F') or surface not in ('ned_z','world_height'):
        raise ValueError('Measured occupancy layout and surface units required')
    output=Path(args.output).resolve()
    if not output.is_relative_to(root):raise ValueError('Geometry output escapes project root')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'geometry.json').exists():raise ValueError('Geometry identity is immutable')
    write(output/'qualification.json',q)
    origin=np.asarray([-384,-384,-192]);shape=np.asarray([768,768,192]);tiles=[]
    window=RunWindow(args.hours)
    resources=Resources(root,cfg,'cuda');resources.check()
    with FlightLock(root,'native-geometry-acquisition'):
        with CityEnvironment(scene,output/'native-capture',cfg) as env:
            if scene.get('backend')!='projectairsim':raise RuntimeError('This geometry adapter requires qualified ProjectAirSim')
            world=env.owned.session.world
            for offset in itertools.product(range(0,shape[0],64),range(0,shape[1],64),range(0,shape[2],64)):
                resources.check()
                if not window.admits(30):raise RuntimeError('Geometry window ended; complete tiles retained for resume')
                path=output/('tile-'+'-'.join(map(str,offset))+'.npy')
                meta=path.with_suffix('.json')
                if path.exists() and meta.exists():
                    tile=read(meta)
                    if tile['sha256']!=digest(path) or tile['qualification_sha256']!=digest(output/'qualification.json'):
                        raise ValueError('Resumable tile identity mismatch')
                else:
                    center=(origin+offset+32).tolist()
                    pose=Pose(dict(translation=dict(zip(('x','y','z'),center)),rotation=dict(w=1,x=0,y=0,z=0),frame_id='DEFAULT_ID'))
                    raw=np.asarray(world.create_voxel_grid(pose,64,64,64,1,actors_to_ignore=['drone_1'],write_file=False))
                    if raw.size!=64**3 or raw.dtype!=np.bool_:raise RuntimeError('Malformed native occupancy; no free-space fallback')
                    array=raw.reshape((64,)*3,order=order).transpose(tuple(axes.index(a) for a in ('x','y','z')))
                    reserve_write(path,array.nbytes+65536)
                    with path.with_suffix('.pending').open('wb') as stream:np.save(stream,array,allow_pickle=False)
                    path.with_suffix('.pending').replace(path)
                    tile=dict(path=path.name,sha256=digest(path),grid_start=list(offset),
                              qualification_sha256=digest(output/'qualification.json'))
                    write(meta,tile)
                tiles.append(tile)
    write(output/'geometry.json',dict(schema='photo-goal-private-geometry/v1',qualified=True,
        scene_sha256=digest(args.scene),origin=origin.tolist(),shape=shape.tolist(),resolution_m=1,
        qualification=dict(path='qualification.json',sha256=digest(output/'qualification.json')),
        array_axes=axes,array_order=order,surface_units=surface,tiles=tiles))
