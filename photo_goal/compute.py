"""One GPU lane with priority at bounded model-step boundaries.

CUDA work is synchronized before releasing the lane. Kernels are not assumed
preemptible; measured slow slices exceeding the budget suspend future planning.
"""
from contextlib import contextmanager
import threading
import time
import torch


class ComputeLane:
    def __init__(self, maximum_slow_slice_s=.05):
        self.condition=threading.Condition();self.busy=False;self.fast_waiting=0
        self.maximum=maximum_slow_slice_s;self.suspend_until=0.;self.essential_waiting=0
        self.slowest_slice_s=0.;self.slow_slices=0
        self.last_fast_queue_s=0.;self.overruns=0

    @property
    def slow_admitted(self):return time.monotonic()>=self.suspend_until

    @contextmanager
    def enter(self, fast=False, essential=False):
        queued=time.monotonic()
        with self.condition:
            if fast:self.fast_waiting+=1
            elif essential:self.essential_waiting+=1
            try:
                self.condition.wait_for(lambda:not self.busy and (fast or self.fast_waiting==0 and (essential or self.essential_waiting==0)))
                if not fast and not essential and not self.slow_admitted:
                    raise RuntimeError('prediction_slice_budget_exceeded')
                self.busy=True
            finally:
                if fast:self.fast_waiting-=1
                elif essential:self.essential_waiting-=1
        started=time.monotonic()
        if fast:self.last_fast_queue_s=started-queued
        try:
            yield
        finally:
            try:
                if torch.cuda.is_available():torch.cuda.synchronize()
            finally:
                elapsed=time.monotonic()-started
                with self.condition:
                    if not fast:
                        self.slow_slices+=1;self.slowest_slice_s=max(self.slowest_slice_s,elapsed)
                        if not essential and elapsed>self.maximum:
                            self.overruns=min(5,self.overruns+1)
                            self.suspend_until=time.monotonic()+min(60.,3.*2**(self.overruns-1))
                        elif not essential:self.overruns=max(0,self.overruns-1)
                    elif started-queued>self.maximum:
                        self.suspend_until=max(self.suspend_until,time.monotonic()+3.)
                    self.busy=False;self.condition.notify_all()

    def fast(self):return self.enter(True)
    def slow(self):return self.enter(False)
    def background(self):return self.enter(False,True)
