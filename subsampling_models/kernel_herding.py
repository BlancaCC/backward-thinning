import numpy as np

def kernel_herding(
    m,
    K_x=None,
    K_y=None,
    phi_x=None,
    phi_y=None,
    unique=True
):
    """
    Kernel herding (equivalent to the first implementation)

    Parameters
    ----------
    m : int
        Coreset size

    Input options (provide one of the following)
    -------------------------------------------
    K_x : (n, n) array
        Kernel matrix on X
    K_y : (n, n) array, optional
        Kernel matrix on Y (for supervised setting)

    phi_x : (n, d) array
        Feature map for X
    phi_y : (n, d') array, optional
        Feature map for Y (for supervised setting)

    unique : bool
        If True, prevents reselection of the same index

    Returns
    -------
    coreset : (m,) array of selected indices
    """

    # --- Build kernel matrix ---
    if K_x is not None:
        K = K_x.copy()
        if K_y is not None:
            K = K * K_y  # supervised: product kernel
    elif phi_x is not None:
        K = phi_x @ phi_x.T
        if phi_y is not None:
            K = K * (phi_y @ phi_y.T)
    else:
        raise ValueError("You must provide either K_x or phi_x")

    n = K.shape[0]

    # --- Mean embedding ---
    meanK = K.mean(axis=1)

    # --- Initialization ---
    objective = meanK.copy()
    coreset = np.empty(m, dtype=int)

    for t in range(m):
        # Select argmax
        i_star = np.argmax(objective)
        coreset[t] = i_star

        # Update objective (key step: correct herding update)
        objective = (
            objective * (t + 1) / (t + 2)
            + (meanK - K[i_star]) / (t + 2)
        )

        if unique:
            objective[i_star] = -np.inf

    return coreset


# import numpy as np

# def kernel_herding(K, m):
#     """
#     K: (n, n) kernel matrix (PSD, positive kernel required)
#     m: coreset size
#     """

#     n = K.shape[0]

#     # μ = mean embedding (Bach / Chen notation)
#     mu = K.mean(axis=1)

#     # ψ_0 = 0 (empirical embedding of selected set)
#     psi = np.zeros(n)

#     selected = []

#     for t in range(1, m + 1):

#         # g_t(x) = ⟨φ(x), μ - ψ_{t-1}⟩
#         # kernel form: K[:, i] with residual
#         scores = mu - psi

#         # avoid reselection
#         scores[selected] = -np.inf

#         # x_t = argmax g_t(x)
#         i_star = np.argmax(scores)
#         selected.append(i_star)

#         # ψ update (herding / Frank-Wolfe step)
#         psi += K[i_star] / m   # (Chen et al. 2010)

#     return selected