import time
import tracemalloc
from typing import Dict, Any, Optional

# Attempt to import PAPI for hardware FLOPS counting
try:
    from pypapi import papi_low, events
    PAPI_AVAILABLE = True
except ImportError:
    PAPI_AVAILABLE = False

class HardwareProfiler:
    """A context manager to profile hardware usage (Time, Memory, and FLOPS).

    This class measures the execution time, peak memory usage, and 
    hardware floating-point operations (if supported) of a code block.

    Attributes:
        label (str): Name of the model or block being profiled.
        results (Dict[str, Any]): Dictionary containing the profiling data
            after the context exits.
    """

    def __init__(self, label: str = "Model"):
        """Initializes the profiler.

        Args:
            label: A descriptive name for the profiling session.
        """
        self.label = label
        self.results: Dict[str, Any] = {}
        self._start_time: float = 0.0
        self._papi_supported: bool = False
        self._evs: Optional[int] = None

        if PAPI_AVAILABLE:
            try:
                papi_low.library_init()
                self._evs = papi_low.create_eventset()
                # PAPI_DP_OPS counts double precision floating point operations
                papi_low.add_event(self._evs, events.PAPI_DP_OPS)
                self._papi_supported = True
            except Exception:
                self._papi_supported = False

    def __enter__(self) -> "HardwareProfiler":
        """Starts the profiling session (timer, memory tracing, and PAPI).

        Returns:
            HardwareProfiler: The instance itself.
        """
        tracemalloc.start()
        if self._papi_supported:
            papi_low.start(self._evs)
        
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Finalizes profiling and populates the results dictionary."""
        # 1. Stop time
        duration = time.perf_counter() - self._start_time
        
        # 2. Stop FLOPS counting
        flops = 0
        if self._papi_supported:
            flops = papi_low.stop(self._evs)[0]
        
        # 3. Stop memory tracing
        _, peak = tracemalloc.get_traced_memory()
        peak_mb = peak / (1024 ** 2)
        tracemalloc.stop()

        # Populate the results dictionary
        self.results = {
            "label": self.label,
            "time_sec": round(duration, 6),
            "memory_peak_mb": round(peak_mb, 4),
            "flops": flops,
            "gflops_per_sec": round((flops / duration) / 1e9, 4) if duration > 0 and flops > 0 else 0.0,
            "status": "Success" if exc_type is None else f"Error: {exc_type.__name__}"
        }

        # Visual feedback
        self._print_summary()

    def _print_summary(self) -> None:
        """Prints a formatted summary of the results to the console."""
        print(f"\n--- [ {self.results['label']} Profiling ] ---")
        print(f"  Time        : {self.results['time_sec']} s")
        print(f"  Peak Memory : {self.results['memory_peak_mb']} MB")
        if self._papi_supported:
            print(f"  Total FLOPS : {self.results['flops']:,}")
            print(f"  Throughput  : {self.results['gflops_per_sec']} GFLOPS/s")
        else:
            print("  Total FLOPS : Not available (PAPI not supported)")
        print("-" * 30)

    def get_data(self) -> Dict[str, Any]:
        """Returns the profiling data.

        Returns:
            A dictionary containing time, memory, and FLOPS metrics.
        """
        return self.results