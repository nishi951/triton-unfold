from typing import Optional, Literal

from pathlib import Path
import copy
import gc
import logging

import torch
import cupy as cp
from easydict import EasyDict

from ._defaults import default_to

logger = logging.getLogger("torchmri.utils")

__all__ = ["benchmark"]


def benchmark(
    fn,
    *args,
    num_iters: int = 10,
    backend: Literal["torch", "cupy"] = "torch",
    **kwargs,
):
    """Benchmark a function called with some arguments.

    Defaults to torch benchmarking
    """
    if backend == "torch":
        backend = TorchHandler()
    elif backend == "cupy":
        backend = CupyHandler()
    else:
        raise ValueError(f"Unrecognized backend type {backend}")
    fn_result = fn(*args, **kwargs)  # Warmup
    backend.bench_start()
    for _ in range(num_iters):
        backend.trial_start()
        fn(*args, **kwargs)
        backend.trial_end()
    backend.bench_end()
    return backend.result, fn_result


class TorchHandler:
    def __init__(self, memory_snapshot_file: Optional[Path] = "memory.pkl"):
        self.reset()
        self.memory_snapshot_file = memory_snapshot_file

    def reset(self):
        self._start_event = None
        self._end_event = None

        self.result = EasyDict(
            {
                "timings_ms": [],
                "max_mem_bytes": None,
            }
        )

    def bench_start(self, *args, **kwargs):
        self.reset()
        gc.disable()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.memory._record_memory_history(
            max_entries=100000,
        )

    def bench_end(self, *args, **kwargs):
        self.result.max_mem_bytes = torch.cuda.max_memory_allocated()
        gc.enable()
        logger.info(f"Max memory allocated: {self.result.max_mem_bytes}")
        try:
            torch.cuda.memory._dump_snapshot(f"{str(self.memory_snapshot_file)}")
        except Exception as e:
            logger.error(f"Failed to capture memory snapshot {e}")

        # Stop recording memory snapshot history.
        torch.cuda.memory._record_memory_history(enabled=None)

    def trial_start(self, event=None, i=None):
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)
        self._start_event.record()

    def trial_end(self, event=None, i=None):
        self._end_event.record()
        torch.cuda.synchronize()
        time = self._start_event.elapsed_time(self._end_event)
        logger.debug(f"{event}: {time}")
        self.result.timings_ms.append(time)

    def collect_results(self, event, data):
        return {"torch": copy.deepcopy(self.result)}


class CupyHandler:
    """Benchmarking class for CuPy-based functions

    Usage:


    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._start_event = None
        self._end_event = None
        self._mempool = None

        self.result = EasyDict(
            {
                "timings_ms": [],
                "max_mem_bytes": None,
            }
        )

    def bench_start(self, *args, **kwargs):
        self.reset()
        gc.disable()
        self._mempool = cp.get_default_memory_pool()
        self._mempool.free_all_blocks()

    def bench_end(self, *args, **kwargs):
        self.result.max_mem_bytes = self._mempool.total_bytes()
        gc.enable()

    def trial_start(self, event=None, i=None):
        self._start_event = cp.cuda.Event(disable_timing=False)
        self._end_event = cp.cuda.Event(disable_timing=False)
        self._start_event.record()

    def trial_end(self, event=None, i=None):
        self._end_event.record()
        self._end_event.synchronize()
        time = cp.cuda.get_elapsed_time(self._start_event, self._end_event)  # ms
        self.result.timings_ms.append(time)

    def collect_results(self, *args, **kwargs):
        return {"cupy": copy.deepcopy(self.result)}
