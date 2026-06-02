import numpy as np


class phiKBKH:
    """
    Backward kernel thinning. Accepts either:

      - Explicit feature maps  phi_x / phi_y  (fast, O(nD))
      - Exact kernel functions kernel_X / kernel_y  (exact, O(n²))

    When feature maps are available they are always preferred.
    """

    def __init__(self, kernel_X=None, kernel_y=None, phi_x=None, phi_y=None):
        if kernel_X is None and phi_x is None:
            raise ValueError("Provide at least one of: kernel_X, phi_x")
        self.kernel_X = kernel_X
        self.kernel_y  = kernel_y
        self.phi_x     = phi_x
        self.phi_y     = phi_y

    # ------------------------------------------------------------------
    # Gram-matrix builder (exact kernel path)
    # ------------------------------------------------------------------

    def _build_gram(self, X, Y=None):
        n = len(X)
        K = np.empty((n, n), dtype=np.float64)

        for i in range(n):
            K[i, i] = self.kernel_X(X[i], X[i])
            for j in range(i + 1, n):
                v = self.kernel_X(X[i], X[j])
                K[i, j] = K[j, i] = v

        if Y is not None and self.kernel_y is not None:
            Ky = np.empty((n, n), dtype=np.float64)
            for i in range(n):
                Ky[i, i] = self.kernel_y(Y[i], Y[i])
                for j in range(i + 1, n):
                    v = self.kernel_y(Y[i], Y[j])
                    Ky[i, j] = Ky[j, i] = v
            K *= Ky

        return K

    # ------------------------------------------------------------------
    # Thinning — feature-map path  O(nD)
    # ------------------------------------------------------------------

    def _thin_feature_map(self, n_to_rm, X, Y=None):
        Phi = self.phi_x(X)          # (n, D)
        n   = len(X)

        # Build product-kernel Gram matrix in feature space
        K = Phi @ Phi.T
        if Y is not None and self.phi_y is not None:
            Phi_y = self.phi_y(Y)
            K    *= Phi_y @ Phi_y.T

        mu     = K.mean(axis=0)      # mean embedding, shape (n,)
        diagK  = np.diag(K).copy()   # k(x_i, x_i)
        inner  = mu.copy()           # tracks <phi_i, residual>

        active      = np.arange(n)
        active_size = n

        for _ in range(n_to_rm):
            cur = active[:active_size]

            # Score: k(x_i, x_i) - 2 <phi_i, residual>
            scores   = diagK[cur] - 2.0 * inner[cur]
            loc_idx  = np.argmin(scores)
            glob_idx = cur[loc_idx]

            # Update residual for remaining points
            active[loc_idx], active[active_size - 1] = (
                active[active_size - 1], active[loc_idx]
            )
            active_size -= 1
            rem = active[:active_size]
            inner[rem] -= K[rem, glob_idx] - mu[rem]

        keep = active[:active_size]
        return (X[keep], Y[keep]) if Y is not None else (X[keep], None)

    # ------------------------------------------------------------------
    # Thinning — exact-kernel path  O(n²)
    # ------------------------------------------------------------------

    def _thin_kernel(self, n_to_rm, X, Y=None):
        n     = len(X)
        K     = self._build_gram(X, Y)
        mu    = K.mean(axis=1)       # mean embedding, shape (n,)
        diagK = np.diag(K).copy()
        inner = mu.copy()

        active      = np.arange(n)
        active_size = n

        for _ in range(n_to_rm):
            cur      = active[:active_size]
            scores   = diagK[cur] - 2.0 * inner[cur]
            loc_idx  = np.argmin(scores)
            glob_idx = cur[loc_idx]

            active[loc_idx], active[active_size - 1] = (
                active[active_size - 1], active[loc_idx]
            )
            active_size -= 1
            rem = active[:active_size]
            inner[rem] -= K[rem, glob_idx] - mu[rem]

        keep = active[:active_size]
        return (X[keep], Y[keep]) if Y is not None else (X[keep], None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def thin(self, n_to_rm, X, Y=None):
        n = len(X)
        if n_to_rm >= n:
            raise ValueError("n_to_rm must be < len(X)")

        if self.phi_x is not None:
            return self._thin_feature_map(n_to_rm, X, Y)
        return self._thin_kernel(n_to_rm, X, Y)