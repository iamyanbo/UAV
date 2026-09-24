"""Linux thread/process affinity for two fast and six slow logical CPUs."""
import os


def reserve_fast():
    available=sorted(os.sched_getaffinity(0))
    if len(available)<8:raise RuntimeError('Eight logical CPUs required for navigation runtime')
    fast,slow=available[:2],available[2:8]
    os.environ['RGB_SLOW_CPUS']=','.join(map(str,slow))
    os.sched_setaffinity(0,fast)
    return dict(fast_cpus=fast,slow_cpus=slow)


def slow_worker():
    value=os.environ.get('RGB_SLOW_CPUS')
    if value:os.sched_setaffinity(0,[int(x) for x in value.split(',')])


def slow_command(command):
    value=os.environ.get('RGB_SLOW_CPUS')
    return ['taskset','-c',value,*command] if value else command
