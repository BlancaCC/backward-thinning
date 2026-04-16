import numpy as np
from goodpoints.herding import herding
from functools import partial

def kernel_herding(X, m, kernel_eval, params_k):
    """Runs the kernel herding algorithm on the input data X with combination depth m and kernel evaluation function.
    """
    swap_kernel = partial(kernel_eval, params_k=params_k)
    coreset = herding(X, m, swap_kernel)
    return coreset

"""
#A typical usage example would be:
from experiments.utils import kernel_eval
d = 2
var = 1.
m = int(3)
X = np.random.normal(size=(1024, d))

params_k = {"name": "gauss", "var": var, "d": int(d)}
swap_kernel = partial(kernel_eval, params_k=params_k)
coreset = herding(X, m, swap_kernel)
"""
