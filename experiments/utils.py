import numpy as np

def kernel_eval(x, y, params_k):
    """Returns matrix of kernel evaluations kernel(xi, yi)
    params_k should contain:
    - name: kernel name (e.g., "gauss")
    - var: variance parameter for Gaussian kernel
    - d: dimensionality of the input data

    Used for the goodpoints implementations of kernel thinning and herding.
    """
    
    if params_k["name"] in ["gauss", "gauss_rt"]:
        k_vals = np.sum((x-y)**2,axis=1)
        scale = -.5/params_k["var"]
        return(np.exp(scale*k_vals))
    
    raise ValueError("Unrecognized kernel name {}".format(params_k["name"]))