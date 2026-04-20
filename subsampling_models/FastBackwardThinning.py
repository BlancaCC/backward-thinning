import numpy as np

class FastBackwardThinning:
    """Implements backward thinning for dataset reduction in feature space."""

    def __init__(self, feature_map_X, feature_map_y=None):
        self.feature_map_X = feature_map_X
        self.feature_map_y = feature_map_y

    @staticmethod
    def _safe_norms(D):
        """Compute L2 norms row-wise with overflow protection via per-row scaling."""
        # Divide each row by its abs-max before squaring to prevent overflow
        row_max = np.abs(D).max(axis=1, keepdims=True)
        row_max = np.where(row_max == 0, 1.0, row_max)  # avoid div by zero
        D_scaled = D / row_max
        norms = np.sqrt((D_scaled ** 2).sum(axis=1)) * row_max[:, 0]
        return norms

    @staticmethod
    def _normalize(arr):
        """Normalize array to [0,1] for safe multiplication. Preserves argmin."""
        amax = arr.max()
        if amax > 0:
            arr /= amax
        return arr

    def thin(self, n_to_rm, X, Y=None):
        if Y is not None and self.feature_map_y is None:
            raise ValueError("feature_map_y must be provided if Y is not None.")

        use_Y = Y is not None

        # Map to feature space once, ensure float64
        phi_X = self.feature_map_X(X).astype(np.float64)
        psi_Y = self.feature_map_y(Y).astype(np.float64) if use_Y else None

        n = phi_X.shape[0]
        mask = np.ones(n, dtype=bool)

        # Keep errors in float64; clip aggressively to prevent accumulation blow-up
        error_x = np.zeros(phi_X.shape[1], dtype=np.float64)
        error_y = np.zeros(psi_Y.shape[1], dtype=np.float64) if use_Y else None

        for _ in range(n_to_rm):
            phi_active = phi_X[mask]
            mu_x = phi_active.mean(axis=0)
            D_x = phi_active - (mu_x - error_x)

            scores = self._safe_norms(D_x)
            self._normalize(scores)

            if use_Y:
                psi_active = psi_Y[mask]
                mu_y = psi_active.mean(axis=0)
                D_y = psi_active - (mu_y - error_y)

                scores_y = self._safe_norms(D_y)
                self._normalize(scores_y)

                scores *= scores_y

            local_idx = int(scores.argmin())

            # Clip error accumulation to float64 safe range
            new_ex = error_x + D_x[local_idx]
            error_x = np.clip(new_ex, -1e300, 1e300)
            if use_Y:
                new_ey = error_y + D_y[local_idx]
                error_y = np.clip(new_ey, -1e300, 1e300)

            global_idx = np.where(mask)[0][local_idx]
            mask[global_idx] = False

        return X[mask], (Y[mask] if use_Y else None)