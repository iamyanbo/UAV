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
    if abs(fixed_pitch)>1:raise ValueError('Temporal campaign requires a fixed forward camera')
    if isinstance(commands,IndexedCommands):commands=commands.window(start_ns)
    return executed_slots(commands,start_ns)
