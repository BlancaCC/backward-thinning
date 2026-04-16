import numpy as np

def kernel_herding(K, m):
    """
    K: (n, n) kernel matrix (PSD, positive kernel required)
    m: coreset size
    """

    n = K.shape[0]

    # μ = mean embedding (Bach / Chen notation)
    mu = K.mean(axis=1)

    # ψ_0 = 0 (empirical embedding of selected set)
    psi = np.zeros(n)

    selected = []

    for t in range(1, m + 1):

        # g_t(x) = ⟨φ(x), μ - ψ_{t-1}⟩
        # kernel form: K[:, i] with residual
        scores = mu - psi

        # avoid reselection
        scores[selected] = -np.inf

        # x_t = argmax g_t(x)
        i_star = np.argmax(scores)
        selected.append(i_star)

        # ψ update (herding / Frank-Wolfe step)
        psi += K[i_star] / m   # (Chen et al. 2010)

    return selected