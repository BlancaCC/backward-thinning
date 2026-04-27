import numpy as np


class KernelHerding:
    """
    Kernel Herding class.

    This implementation supports:
    - Kernel functions k(x, x') or k((x,y), (x',y'))
    - Feature maps phi(x) or phi(x, y)
    - Supervised setting via product kernels
    """

    def __init__(
        self,
        m,
        k_x=None,
        k_y=None,
        phi_x=None,
        phi_y=None,
        unique=True
    ):
        """
        Parameters
        ----------
        m : int
            Size of the coreset

        k_x : callable, optional
            Kernel function for X: k_x(x_i, x_j)

        k_y : callable, optional
            Kernel function for Y: k_y(y_i, y_j)

        phi_x : callable, optional
            Feature map for X: phi_x(x_i)

        phi_y : callable, optional
            Feature map for Y: phi_y(y_i)

        unique : bool
            If True, prevents selecting the same index multiple times
        """

        self.m = m
        self.k_x = k_x
        self.k_y = k_y
        self.phi_x = phi_x
        self.phi_y = phi_y
        self.unique = unique

        # Basic validation
        if k_x is None and phi_x is None:
            raise ValueError("You must provide either k_x or phi_x")

    def _compute_kernel_matrix(self, X, Y=None):
        """
        Compute the full kernel matrix.

        Supports:
        - Kernel functions
        - Feature maps
        - Supervised product kernels
        """

        n = len(X)

        # --- Case 1: kernel functions ---
        if self.k_x is not None:
            K = np.zeros((n, n))

            for i in range(n):
                for j in range(n):
                    K[i, j] = self.k_x(X[i], X[j])

            # Supervised case
            if self.k_y is not None and Y is not None:
                K_y = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        K_y[i, j] = self.k_y(Y[i], Y[j])

                K = K * K_y

        # --- Case 2: feature maps ---
        else:
            Phi_x = self.phi_x(X)
            K = Phi_x @ Phi_x.T

            if self.phi_y is not None and Y is not None:
                Phi_y = self.phi_y(Y)
                K = K * (Phi_y @ Phi_y.T)

        return K

    def get_coreset(self, X, Y=None):
        """
        Compute the coreset indices.

        Parameters
        ----------
        X : array-like
            Input data

        Y : array-like, optional
            Labels (for supervised setting)

        Returns
        -------
        coreset : (m,) array
            Selected indices
        """

        # --- Build kernel matrix ---
        K = self._compute_kernel_matrix(X, Y)

        n = K.shape[0]

        # --- Mean embedding ---
        meanK = K.mean(axis=1)

        # --- Initialization ---
        objective = meanK.copy()
        coreset = np.empty(self.m, dtype=int)

        for t in range(self.m):
            # Select argmax
            i_star = np.argmax(objective)
            coreset[t] = i_star

            # Herding update
            objective = (
                objective * (t + 1) / (t + 2)
                + (meanK - K[i_star]) / (t + 2)
            )

            if self.unique:
                objective[i_star] = -np.inf

        return coreset