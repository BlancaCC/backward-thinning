import numpy as np
from typing import Callable

def make_gaussian_kernel(bandwidth: float) -> Callable:
    """k(x,y) = exp(-‖x-y‖²/(2σ²))."""
    s2 = 2.0 * bandwidth ** 2
    def k(x, y):
        d = np.asarray(x) - np.asarray(y)
        return float(np.exp(-float(d @ d) / s2))
    return k
