"""Vehicle/camera dispatch windows, never relabelled actuator timestamps."""
from bisect import bisect_right
import numpy as np
from .vision.action_intervals import executed_slots, SLOT_NS


def dispatch_interval(commands, begin, end):
    """Measured frame-to-frame dispatch exposure, including piecewise commands."""
    if not commands.valid or end<=begin:return None
    left=bisect_right(commands.stamps,begin)-1
    if left<0 or max(commands.stamps[-1],getattr(commands,'coverage_end_ns',0))<end:return None
    events=commands.rows[left:bisect_right(commands.stamps,end)+1]
    cursor=begin;current=events[0];segments=[];total=np.zeros(4)
    for following in [r for r in events[1:] if r.get('dispatch_sim_ns',r['sim_ns'])<end]+[None]:
        boundary=following.get('dispatch_sim_ns',following['sim_ns']) if following else end
        if boundary-current.get('dispatch_sim_ns',current['sim_ns'])>250_000_000:return None
        values=np.asarray(current['values'],dtype=float)
        if values.shape!=(4,) or not np.isfinite(values).all():return None
        duration=boundary-cursor
        if duration>0:
            segments.append(dict(begin_ns=cursor,end_ns=boundary,values=values.tolist(),
                decision_id=current.get('decision_id'),watchdog=bool(current.get('watchdog_override'))))
            total+=duration*values
        cursor=boundary;current=following
    return dict(values=[(total/(end-begin)).tolist()],dt_s=(end-begin)/1e9,segments=segments,
                application_timestamps_observed=False)


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
