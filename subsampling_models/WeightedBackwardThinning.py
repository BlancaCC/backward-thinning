import numpy as np
from itertools import combinations
from collections import defaultdict



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
  
    def _remove_opposite_vectors(self, V, index_to_keep, eps):
        table = EpsilonHashTable(eps)

        to_remove = set()
        pairs = {}

        for i in index_to_keep:
            if not i in to_remove:
                v = V[i]

                # buscar vector opuesto
                result = table.lookup(-v)

                if result is not None:
                    stored_v, group = result

                    j = group  # aquí guardamos el índice
                    to_remove.add(i)
                    pairs[j] = i
                else:
                    # guardamos índice como "group"
                    table.insert(v, i)

        new_index_to_keep = [i for i in index_to_keep if i not in to_remove]

        return new_index_to_keep, pairs
    
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

    def redistribute_weights(self,weights, S, index_to_keep, index_to_remove, alpha):
        """
        Implementa exactamente:
        w_i <- sum_r w_r * k_ir^alpha / sum_j k_jr^alpha + w_i
        """

        if len(index_to_remove) == 0:
            return weights

        W_removed = weights[index_to_remove]  # (m,)

        # Submatriz correcta (índices locales coherentes)
        K = S[np.ix_(index_to_keep, index_to_remove)] ** alpha  # (n_keep, m)

        # Normalización por columna (por cada r)
        denom = K.sum(axis=0)
        denom[denom == 0] = 1.0  # evitar división por cero

        K_normalized = K / denom

        # Redistribución
        weights[index_to_keep] += K_normalized @ W_removed

        # Eliminar peso de los removidos
        weights[index_to_remove] = 0.0

        return weights
    
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
        eps = self.mmd_tolerance

        # Initial uniform weights
        weights = np.ones(n) / n

        # assumpt only quit one 
        V = self._lifted_X - self.mu_px_[np.newaxis, :]  # (n, d)
        distances = np.linalg.norm(V, axis=1)
        print(f"Distances to mean embedding: {min(distances):.4f} to {max(distances):.4f}")
        index_to_keep = np.where(distances > eps)[0]
        print(f"Initial points kept after distance check m=1: {len(index_to_keep)}/{n}")

        
        # m = 2
        index_to_keep, _ = self._remove_opposite_vectors(V, index_to_keep, self.mmd_tolerance)
        index_to_remove = np.setdiff1d(np.arange(n), index_to_keep)
        print(f"Points kept after m=2: {len(index_to_keep)}/{n}")
        S = np.abs(lifted_X @ lifted_X.T)  # (n_keep, n_remove) # hay que hacer el abs porque puede haber valores negativos que den lugar a pesos negativos, por la apoximación d
        weights_to_keep = self.redistribute_weights(weights, S, index_to_keep, index_to_remove, self.alpha)
        print(f"Weights after redistribution ({1/n}): {min(weights_to_keep):.6f} to {max(weights_to_keep):.6f} sum={weights_to_keep.sum():.6f}")
        return X[index_to_keep], weights_to_keep, index_to_keep


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