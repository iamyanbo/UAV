"""Vehicle/camera dispatch windows, never relabelled actuator timestamps."""
from bisect import bisect_right
import numpy as np
from action_intervals import executed_slots, SLOT_NS


class IndexedCommands:
    """Validate a flight once, then join each window in logarithmic time."""
    def __init__(self, rows):
        self.rows=rows
        self.stamps=[r.get('dispatch_sim_ns',r.get('sim_ns')) for r in rows]
        self.valid=bool(rows) and all(s is not None for s in self.stamps)
        self.valid=self.valid and all(a<b for a,b in zip(self.stamps,self.stamps[1:]))

    def window(self, start_ns):
        if not self.valid:return []
        left=max(0,bisect_right(self.stamps,start_ns)-1)
        right=min(len(self.rows),bisect_right(self.stamps,start_ns+4*SLOT_NS)+1)
        return self.rows[left:right]


def aerial_slots(commands, start_ns, fixed_pitch=0.):
    if isinstance(commands,IndexedCommands):commands=commands.window(start_ns)
    vehicle=executed_slots(commands,start_ns)
    if vehicle['values'] is None:
        return dict(vehicle,values=None)
    stamps=[r.get('dispatch_sim_ns',r.get('sim_ns')) for r in commands]
    if any(s is None for s in stamps) or any(a>=b for a,b in zip(stamps,stamps[1:])):
        return dict(values=None,valid=[False]*4,reason='invalid_camera_command_clock')
    values=[]
    for slot in range(4):
        begin=start_ns+slot*SLOT_NS;end=begin+SLOT_NS;i=bisect_right(stamps,begin)-1
        if i<0:
            values.append([*vehicle['values'][slot],fixed_pitch]);continue
        cursor=begin;pitch=float(commands[i].get('camera_pitch_deg',fixed_pitch));total=0.
        for j in range(i+1,bisect_right(stamps,end-1)):
            total+=(stamps[j]-cursor)*pitch;cursor=stamps[j]
            pitch=float(commands[j].get('camera_pitch_deg',fixed_pitch))
        total+=(end-cursor)*pitch
        values.append([*vehicle['values'][slot],total/SLOT_NS])
    if not np.isfinite(values).all():raise ValueError('Invalid camera action target')
    return dict(vehicle,values=values,action_semantics='post-safety-vehicle-camera-dispatch/50ms-v1')
