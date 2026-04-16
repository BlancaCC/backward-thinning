import numpy as np
from goodpoints import kt
from functools import partial

def kernel_eval(x, y, params_k):
    """Returns matrix of kernel evaluations kernel(xi, yi) for each row index i.
    x and y should have the same number of columns, and x should either have the
    same shape as y or consist of a single row, in which case, x is broadcasted 
    to have the same shape as y.
    """
    if params_k["name"] in ["gauss", "gauss_rt"]:
        k_vals = np.sum((x-y)**2,axis=1)
        scale = -.5/params_k["var"]
        return(np.exp(scale*k_vals))
    
    raise ValueError("Unrecognized kernel name {}".format(params_k["name"]))

var = 1. # Variance
d = int(2)
params_p = {"name": "gauss", "var": var, "d": int(d), "saved_samples": False}
params_k_swap = {"name": "gauss", "var": var, "d": int(d)}
params_k_split = {"name": "gauss_rt", "var": var/2., "d": int(d)}
delta = .5
m = int(3)
X = np.random.normal(size=(1024, d))


split_kernel = partial(kernel_eval, params_k=params_k_split)
swap_kernel = partial(kernel_eval, params_k=params_k_swap)
coresets = kt.thin(X, m, split_kernel, swap_kernel, delta=delta)
print("Coreset size: {}".format(coresets.shape[0]))
print("Coreset points:\n{}".format(coresets))