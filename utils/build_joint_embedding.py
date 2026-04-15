"""
Joint Embedding Module for Supervised Kernel Thinning.

This module provides the core functionality for projecting data into a 
Reproducing Kernel Hilbert Space (RKHS). It implements a supervised joint 
embedding approach-

Key Components:
1.  'scale' Heuristic: Automatically determines the RBF kernel bandwidth (gamma) 
    based on data variance and dimensionality, mimicking scikit-learn's behavior if no gamma is provided.
2.  Feature Mapping (phi): Uses Random Fourier Features (RBFSampler) with 
    200 components to approximate Gaussian kernels for input features X.
3.  Target Mapping (psi): 
    - For Classification: Uses One-Hot Encoding to treat labels as 
      orthonormal vectors.
    - For Regression: Applies a secondary RBF mapping to the continuous 
      target variable y.
"""

import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import OneHotEncoder

def compute_gamma_scale(X):
    """
    Computes gamma using sklearn's 'scale' heuristic.

    gamma = 1 / (d * var(X))

    Args:
        X (np.ndarray): Input data.

    Returns:
        float: Gamma value.
    """
    var = np.var(X)
    d = X.shape[1]
    return 1.0 / (d * var + 1e-12)


def build_joint_embedding(X, y, problem, gamma_x=None, gamma_y=None, n_components=50):
    """
    Builds joint embedding φ(X)ψ(y).

    Args:
        X (np.ndarray): Features.
        y (np.ndarray): Labels.
        problem (str): 'classification' or 'regression'.
        gamma_x (float, optional): RBF bandwidth for features. If None, computed using 'scale' heuristic.
        gamma_y (float, optional): RBF bandwidth for regression targets. If None, computed using 'scale' heuristic.

    Returns:
        functions: The embedding functions φ and ψ.
    """
    if gamma_x is None:
        gamma_x = compute_gamma_scale(X)
    phi = RBFSampler(gamma=gamma_x, n_components=n_components, random_state=42)
    phi_X = lambda _X: phi.fit_transform(_X)

    if problem == "classification":
        enc = OneHotEncoder(sparse_output=False)
        psi_y = lambda _y: enc.fit_transform(_y.reshape(-1, 1))

    elif problem == "regression":
        y = y.reshape(-1, 1)
        if gamma_y is None:
            gamma_y = compute_gamma_scale(y)
        psi = RBFSampler(gamma=gamma_y, n_components=n_components, random_state=42)
        psi_y = lambda _y: psi.fit_transform(_y)

    else:
        raise ValueError("Invalid problem type")

    return phi_X, psi_y