import numpy as np


class KernelizedBackwardThinning:
    """
    Efficient backward thinning in RKHS.

    Main optimizations:
    -------------------
    1. Avoid repeated boolean masking
    2. Avoid np.where() inside loop
    3. Avoid fancy indexing allocations
    4. Precompute diagonal
    5. In-place updates
    6. Contiguous memory access as much as possible
    """

    def __init__(self, kernel_X, kernel_y=None):
        self.kernel_X = kernel_X
        self.kernel_y = kernel_y

    def _build_gram(self, X, Y=None):
        n = len(X)

        # -------- Build Kx --------
        K = np.empty((n, n), dtype=np.float64)

        for i in range(n):
            K[i, i] = self.kernel_X(X[i], X[i])

            for j in range(i + 1, n):
                val = self.kernel_X(X[i], X[j])
                K[i, j] = val
                K[j, i] = val

        # -------- Supervised product kernel --------
        if Y is not None:
            if self.kernel_y is None:
                raise ValueError(
                    "kernel_y must be provided if Y is not None."
                )

            Ky = np.empty((n, n), dtype=np.float64)

            for i in range(n):
                Ky[i, i] = self.kernel_y(Y[i], Y[i])

                for j in range(i + 1, n):
                    val = self.kernel_y(Y[i], Y[j])
                    Ky[i, j] = val
                    Ky[j, i] = val

            K *= Ky

        return K

    def thin(self, n_to_rm, X, Y=None):
        """
        Remove n_to_rm samples using backward thinning.

        Returns
        -------
        X_thinned, Y_thinned
        """

        n = len(X)

        if n_to_rm >= n:
            raise ValueError("n_to_rm must be < n")

        # ============================================================
        # Build Gram matrix
        # ============================================================

        K = self._build_gram(X, Y)

        # ============================================================
        # Precompute useful quantities
        # ============================================================

        meanK = K.mean(axis=1)

        # Precompute diagonal once
        diagK = np.diag(K).copy()

        # Current residual / inner product tracker
        inner = meanK.copy()

        # ============================================================
        # Active index structure
        # ============================================================

        active = np.arange(n)
        active_size = n

        removed = np.empty(n_to_rm, dtype=np.int64)

        # ============================================================
        # Main loop
        # ============================================================

        for t in range(n_to_rm):

            current = active[:active_size]

            # --------------------------------------------------------
            # Compute scores
            # score_i = K_ii - 2 * inner_i
            # --------------------------------------------------------

            scores = diagK[current] - 2.0 * inner[current]

            # --------------------------------------------------------
            # Select removal candidate
            # --------------------------------------------------------

            local_idx = np.argmin(scores)
            global_idx = current[local_idx]

            removed[t] = global_idx

            # --------------------------------------------------------
            # Remove from active set
            # Swap-remove trick O(1)
            # --------------------------------------------------------

            active[local_idx], active[active_size - 1] = (
                active[active_size - 1],
                active[local_idx],
            )

            active_size -= 1

            remaining = active[:active_size]

            # --------------------------------------------------------
            # In-place residual update
            # inner_j <- inner_j - (K_jr - meanK_j)
            # --------------------------------------------------------

            inner[remaining] -= (
                K[remaining, global_idx] - meanK[remaining]
            )

        # ============================================================
        # Build final mask
        # ============================================================

        keep = active[:active_size]

        if Y is None:
            return X[keep], None

        return X[keep], Y[keep]

# import numpy as np

# class KernelizedBackwardThinning:
#     """Implements backward thinning for dataset reduction in feature space.
    
#     This class reduces a dataset by iteratively removing points that minimize
#     the product of squared L2 norms (L @ L.T) between the features and the 
#     current cumulative error/residual in the feature space.
#     """

#     def __init__(self, kernel_X, kernel_y=None):
#         """Initializes the thinner with kernel functions.

#         Args:
#             kernel_X (callable): Kernel function for X data.
#             kernel_y (callable, optional): Kernel function for Y data.
#                 feature space. Defaults to None.
#         """
#         self.kernel_X = kernel_X
#         self.kernel_y = kernel_y

#     def thin(self, n_to_rm, X, Y=None):
#         """Reduces the dataset by removing n_to_rm samples.

#         Args:
#             n_to_rm (int): Number of samples to remove from the dataset.
#             X (np.ndarray): Input features of shape (n_samples, n_dims).
#             Y (np.ndarray, optional): Input targets of shape (n_samples, n_dims).
#                 Defaults to None.

#         Returns:
#             tuple: (Reduced X, Reduced Y) or (Reduced X, None).
        
#         Raises:
#             ValueError: If Y is provided but kernel_y is not initialized.
#         """
    
#         if Y is not None and self.kernel_y is None:
#             raise ValueError("kernel_y must be provided if Y is not None.")

#         use_Y = Y is not None
#         n = X.shape[0]

#         # Build Gram matrix for X
#         gram_X = np.array([[self.kernel_X(X[i], X[j]) for j in range(n)]
#                             for i in range(n)])

#         if use_Y:
#             gram_y = np.array([[self.kernel_y(Y[i], Y[j]) for j in range(n)]
#                                 for i in range(n)])
#             gram = gram_X * gram_y
#         else:
#             gram = gram_X

#         # ✓ Compute means AFTER combining gram matrices
#         means = np.mean(gram, axis=1)

#         mask = np.ones(n, dtype=bool)
#         inner_product = np.copy(means)

#         for _ in range(n_to_rm):
#             active_indices = np.where(mask)[0]

#             scores = gram[active_indices, active_indices] - 2 * inner_product[active_indices]
#             local_idx = np.argmin(scores)
#             global_idx = active_indices[local_idx]

#             # Update only remaining points
#             mask[global_idx] = False
#             inner_product[mask] -= gram[mask, global_idx] - means[mask]

#         return X[mask], (Y[mask] if use_Y else None)