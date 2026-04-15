import numpy as np
from itertools import combinations


# ─────────────────────────────────────────────────────────────────────────────
# Epsilon Hash Table (Euclidean tolerance)
# ─────────────────────────────────────────────────────────────────────────────

class EpsilonHashTable:
    """
    Hash table that stores vectors up to Euclidean tolerance epsilon.

    Insertion: O(1) amortized — computes a grid-aligned bucket key and stores
    the vector (plus its group metadata) in that bucket.
    Lookup:    Checks the 3^d neighbouring cells so that vectors within epsilon
    of a cell boundary are still found.  For small d this is cheap.
    """

    def __init__(self, epsilon: float):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = epsilon
        # Cell size = epsilon so two vectors in the same cell are guaranteed
        # to be within epsilon*sqrt(d) of each other.  We check neighbours to
        # handle the boundary case exactly.
        self.cell = epsilon
        # bucket_key -> list of (vector, group_set)
        self._buckets: dict[tuple, list] = {}

    # ------------------------------------------------------------------
    def _grid_key(self, v: np.ndarray) -> tuple:
        """Integer grid coordinates of the cell that contains v."""
        return tuple(int(np.floor(x / self.cell)) for x in v)

    def _neighbour_keys(self, v: np.ndarray):
        """Yield all 3^d neighbouring cell keys (including the cell itself)."""
        base = self._grid_key(v)
        d = len(base)
        for offsets in np.ndindex(*(3,) * d):
            yield tuple(base[i] + offsets[i] - 1 for i in range(d))

    # ------------------------------------------------------------------
    def insert(self, v: np.ndarray, group) -> None:
        """Insert v into the table together with its associated group."""
        key = self._grid_key(v)
        entry = (v.copy(), group if group is not None else set())
        self._buckets.setdefault(key, []).append(entry)

    def lookup(self, v: np.ndarray):
        """
        Return (stored_vector, group) if some stored vector is within epsilon
        of v, else None.
        """
        for key in self._neighbour_keys(v):
            for stored_v, group in self._buckets.get(key, []):
                if np.linalg.norm(v - stored_v) < self.epsilon:
                    return stored_v, group
        return None

    def __contains__(self, v: np.ndarray) -> bool:
        return self.lookup(v) is not None


# ─────────────────────────────────────────────────────────────────────────────
# Weighted Backward Thinning
# ─────────────────────────────────────────────────────────────────────────────

class WeightedBackwardThinning:
    """
    Weighted Backward Thinning coreset construction.

    Parameters
    ----------
    feature_map_X : callable
        Map  X → H  (returns an (n, d) array when called on (n, p) input).
    feature_map_y : callable, optional
        Analogous map for labels / responses.
    depth : int
        Combination size m used in the combinatorial tree step.
    mmd_tolerance : float
        Epsilon tolerance for the epsilon hash table.
    alpha : float
        Exponent used in the kernel-based weight redistribution step.
    """

    def __init__(
        self,
        feature_map_X,
        feature_map_y=None,
        depth: int = 2,
        mmd_tolerance: float = 0.4,
        alpha: float = 1.0,
    ):
        self.feature_map_X = feature_map_X
        self.feature_map_y = feature_map_y
        self.depth = depth
        self.mmd_tolerance = mmd_tolerance
        self.alpha = alpha

        # Fitted attributes (set by fit)
        self.mu_px_ = None
        self.mu_py_ = None
        self._lifted_X = None

        # Result attributes (set by transform)
        self.coreset_indices_ = None
        self.weights_ = None
        self.removed_indices_ = None

    # ------------------------------------------------------------------
    # Step 0 – Fit: compute mean embeddings
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y):
        """
        Compute mean embedding(s) over the full dataset.

        Parameters
        ----------
        X : (n, p) array
        y : (n, q) array, optional
        """
        # ── Step 1 – Global Mean Embedding ────────────────────────────
        lifted_X = self.feature_map_X(X)          # (n, d)
        self._lifted_X = lifted_X
        self.mu_px_ = lifted_X.mean(axis=0)       # (d,)

        if self.feature_map_y is not None and y is not None:
            lifted_y = self.feature_map_y(y)
            self.mu_py_ = lifted_y.mean(axis=0)

        return self

    # ------------------------------------------------------------------
    # Step 1 – Transform: backward thinning + weight redistribution
    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray, y):
        """
        Run the backward thinning algorithm and return the coreset.

        Parameters
        ----------
        X : (n, p) array  — same data passed to fit.
        y : ignored (kept for API symmetry).

        Returns
        -------
        X_final : (m', p) array  — coreset points.
        weights  : (m',)  array  — redistributed weights.
        """
        if self.mu_px_ is None or self._lifted_X is None:
            raise RuntimeError("Call fit() before transform().")

        n = len(X)
        lifted_X = self._lifted_X          # (n, d)
        mu_p = self.mu_px_                 # (d,)
        m = self.depth
        eps = self.mmd_tolerance

        # Initial uniform weights
        weights = np.ones(n) / n

        # assumpt only quit one 

        distances = np.linalg.norm(self._lifted_X - self.mu_px_[np.newaxis, :], axis=1)
        print(f"Distances to mean embedding: {min(distances):.4f} to {max(distances):.4f}")
        index_to_remove = np.where(distances <= eps)[0]
        index_to_keep = np.where(distances > eps)[0]
        

        # Compute weights for the remaining points (those not removed)
        S = lifted_X[index_to_keep]  @ lifted_X[index_to_remove].T  # (m', |R|)
        S_col_normalized = S / np.linalg.norm(S, axis=0, keepdims=True)  # Normalize columns
        

        if len(index_to_remove) > 0:
            weights[index_to_keep] += S_col_normalized @ weights[index_to_remove].T  # Redistribute weights
            weights[index_to_keep] /= weights[index_to_keep].sum()  # Normalize weights
            weights[index_to_remove] = 0.0  # Ensure removed points have zero weight

        return X[index_to_keep], weights[index_to_keep]

        # # ── Step 2 – Combinatorial Tree Population ─────────────────────
        # print('Inicio paso 1')
        # T = EpsilonHashTable(epsilon=eps)   # stores V_S vectors
        # G: dict[int, set] = {}              # id(stored_V) → set of point indices

        # for subset_idx in combinations(range(n), m):
        #     subset_idx = list(subset_idx)
        #     phi_sum = lifted_X[subset_idx].sum(axis=0)   # Σ Φ(x_j)

        #     # V_S = −((m−1)·μ̂_P − Σ Φ(x_j))
        #     #      = Σ Φ(x_j) − (m−1)·μ̂_P
        #     V_S = phi_sum - (m - 1) * mu_p

        #     # Insert only if novel within tolerance ε
        #     if V_S not in T:
        #         T.insert(V_S, group=set(subset_idx))
        #         # Record via the object id of the just-inserted vector
        #         # We retrieve it immediately to grab the stored reference.
        #         stored = T.lookup(V_S)
        #         if stored is not None:
        #             stored_v, stored_group = stored
        #             stored_group.update(subset_idx)
        # # ── Step 3 – Find removable set R ──────────────────────────────
        # R: set[int] = set()

        # for i in range(n):
        #     # Check if μ̂_P − Φ(x_i) ∈ T_ε
        #     query = mu_p - lifted_X[i]
        #     result = T.lookup(query)
        #     if result is not None:
        #         _, group = result
        #         R.add(i)
        #         R.update(group)

        # # ── Step 4 – Define X_final ─────────────────────────────────────
        # final_mask = np.ones(n, dtype=bool)
        # final_mask[list(R)] = False
        # final_indices = np.where(final_mask)[0]
        # removed_indices = np.where(~final_mask)[0]

        # # Edge-case: if everything is removed keep the full dataset
        # if len(final_indices) == 0:
        #     final_indices = np.arange(n)
        #     removed_indices = np.array([], dtype=int)

        # # ── Step 5 – Kernel-based Heuristic Weighting ──────────────────
        # Phi_final = lifted_X[final_indices]    # (m', d)
        # m_prime = len(final_indices)

        # # Inner-product kernel matrix K[i,j] = <Φ(x_i), Φ(x_j)>
        # K_ff = Phi_final @ Phi_final.T         # (m', m')

        # # s_i = Σ_{x_j ∈ X_final} <Φ(x_i), Φ(x_j)>
        # # (not used directly in weight update but available for diagnostics)
        # s = K_ff.sum(axis=1)                   # (m',)

        # # Kernel between final points and removed points
        # Phi_removed = lifted_X[removed_indices]  # (|R|, d)
        # w_final = weights[final_indices].copy()
        # alpha = self.alpha

        # if len(removed_indices) > 0:
        #     K_fr = Phi_final @ Phi_removed.T       # (m', |R|)

        #     # For each removed point r, redistribute w_r to final points
        #     # proportionally to k_{ir}^α / Σ_{j∈X_final} k_{jr}^α
        #     K_fr_alpha = np.abs(K_fr) ** alpha     # (m', |R|)
        #     col_sums = K_fr_alpha.sum(axis=0)      # (|R|,)

        #     # Avoid division by zero
        #     safe_sums = np.where(col_sums > 0, col_sums, 1.0)

        #     # w_i += Σ_{r∈R} w_r * k_{ir}^α / Σ_{j} k_{jr}^α
        #     w_removed = weights[removed_indices]   # (|R|,)
        #     delta = (K_fr_alpha / safe_sums) @ w_removed   # (m',)
        #     w_final = w_final + delta

        # # Normalise so weights sum to 1 (preserves probability interpretation)
        # w_sum = w_final.sum()
        # if w_sum > 0:
        #     w_final = w_final / w_sum

        # ── Store and return ───────────────────────────────────────────
        self.coreset_indices_ = final_indices
        self.removed_indices_ = removed_indices
        self.weights_ = w_final

        return X[final_indices], w_final

    # ------------------------------------------------------------------
    # Convenience: fit + transform in one call
    # ------------------------------------------------------------------

    def fit_transform(self, X: np.ndarray, y):
        return self.fit(X, y).transform(X, y)


# ─────────────────────────────────────────────────────────────────────────────
# Quick demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Toy dataset: 50 points in R^4
    X = rng.standard_normal((50, 4))

    # Random Kitchen Sink (RBF approximation) as feature map
    def rbf_feature_map(X, D=16, gamma=0.5, seed=0):
        rng2 = np.random.default_rng(seed)
        W = rng2.normal(0, np.sqrt(2 * gamma), size=(X.shape[1], D))
        b = rng2.uniform(0, 2 * np.pi, size=D)
        Z = np.cos(X @ W + b) * np.sqrt(2 / D)
        return Z

    wbt = WeightedBackwardThinning(
        feature_map_X=rbf_feature_map,
        depth=2,
        mmd_tolerance=0.5,
        alpha=1.0,
    )

    X_coreset, w = wbt.fit_transform(X)

    print(f"Original size : {len(X)}")
    print(f"Coreset size  : {len(X_coreset)}")
    print(f"Removed points: {len(wbt.removed_indices_)}")
    print(f"Weights sum   : {w.sum():.6f}")
    print(f"Coreset indices (first 10): {wbt.coreset_indices_[:10]}")