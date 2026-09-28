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
        self.maximum=maximum_slow_slice_s;self.slow_admitted=True
        self.slowest_slice_s=0.;self.slow_slices=0

    @contextmanager
    def enter(self, fast=False):
        with self.condition:
            if fast:self.fast_waiting+=1
            try:
                self.condition.wait_for(lambda:not self.busy and (fast or self.fast_waiting==0))
                if not fast and not self.slow_admitted:
                    raise RuntimeError('prediction_slice_budget_exceeded')
                self.busy=True
            finally:
                if fast:self.fast_waiting-=1
        started=time.monotonic()
        try:
            yield
        finally:
            try:torch.cuda.synchronize()
            finally:
                elapsed=time.monotonic()-started
                with self.condition:
                    if not fast:
                        self.slow_slices+=1;self.slowest_slice_s=max(self.slowest_slice_s,elapsed)
                        if elapsed>self.maximum:self.slow_admitted=False
                    self.busy=False;self.condition.notify_all()

    def fast(self):return self.enter(True)
    def slow(self):return self.enter(False)
