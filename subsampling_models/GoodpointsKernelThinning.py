from goodpoints import kt
from functools import partial

def kernel_thinning(X, m, delta, kernel_eval, params_k_split, params_k_swap):
    """Runs the kernel thinning algorithm on the input data X with combination depth m and delta parameter.
    """
    split_kernel = partial(kernel_eval, params_k=params_k_split)
    swap_kernel = partial(kernel_eval, params_k=params_k_swap)

    coreset = kt.thin(X, m, split_kernel, swap_kernel, delta=delta)
    return coreset
    
"""
A typical usage example would be:
from utils import kernel_eval
var = 1. 
d = int(2)
params_k_swap = {"name": "gauss", "var": var, "d": int(d)}
params_k_split = {"name": "gauss_rt", "var": var/2., "d": int(d)}
delta = .5
m = int(3)
X = np.random.normal(size=(1024, d))

coresets = kernel_thinning(X, m, delta, kernel_eval, params_k_split, params_k_swap)
"""
