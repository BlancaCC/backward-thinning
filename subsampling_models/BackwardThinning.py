from numpy import mean, zeros, newaxis, delete
from numpy.linalg import norm


class BackwardThinning:
    """Implements backward thinning for dataset reduction in feature space.

    This algorithm iteratively removes samples that minimally affect the
    empirical mean in a transformed feature space. Optionally, it can
    operate on joint feature representations of (X, Y).

    Attributes:
        feature_map_X (callable): Mapping from input space X to feature space.
        feature_map_y (callable, optional): Mapping from output space Y to feature space.
    """

    def __init__(self, feature_map_X, feature_map_y=None):
        """Initializes the BackwardThinning instance.

        Args:
            feature_map_X (callable): Feature map for input data X.
            feature_map_y (callable, optional): Feature map for output data Y.
        """
        self.feature_map_X = feature_map_X
        self.feature_map_y = feature_map_y

    def thin(self, n_to_rm, X, Y=None):
        """Performs backward thinning by removing samples iteratively.

        At each iteration, the sample whose removal minimally perturbs the
        empirical mean in feature space is removed. If Y is provided, a joint
        criterion based on both X and Y feature maps is used.

        Args:
            n_to_rm (int): Number of samples to remove.
            X (ndarray): Input data of shape (n_samples, d_x).
            Y (ndarray, optional): Output data of shape (n_samples, d_y).

        Returns:
            tuple:
                - X (ndarray): Reduced input dataset.
                - Y (ndarray or None): Reduced output dataset if provided, else None.

        Raises:
            ValueError: If Y is provided but feature_map_y is not defined.
        """
        if Y is not None and self.feature_map_y is None:
            raise ValueError("feature_map_y must be provided if Y is not None.")

        # Map X to feature space
        phi_X = self.feature_map_X(X)

        # Initialize accumulated error in X feature space
        error_x = zeros(phi_X.shape[1])

        if Y is not None:
            # Map Y to feature space
            psi_Y = self.feature_map_y(Y)

            # Initialize accumulated error in Y feature space
            error_y = zeros(psi_Y.shape[1])

        # Iteratively remove samples
        for i in range(n_to_rm):
            # Compute empirical mean in X feature space
            mu_x = mean(phi_X, axis=0)

            # Compute deviation from adjusted mean
            D_x = phi_X - (mu_x - error_x)[newaxis, :]

            # Compute norm of each sample's contribution
            norm_x = norm(D_x, axis=1)

            if Y is not None:
                # Same computations for Y
                mu_y = mean(psi_Y, axis=0)
                D_y = psi_Y - (mu_y - error_y)[newaxis, :]
                norm_y = norm(D_y, axis=1)

                # Combine X and Y contributions multiplicatively
                norm_x = norm_x * norm_y

            # Select index of sample to remove (smallest impact)
            idx_to_remove = norm_x.argmin()

            # Update accumulated error
            error_x += D_x[idx_to_remove]

            if Y is not None:
                error_y += D_y[idx_to_remove]

                # Remove selected sample from Y and its feature map
                psi_Y = delete(psi_Y, idx_to_remove, axis=0)
                Y = delete(Y, idx_to_remove, axis=0)

            # Remove selected sample from X and its feature map
            phi_X = delete(phi_X, idx_to_remove, axis=0)
            X = delete(X, idx_to_remove, axis=0)

        return X, Y if Y is not None else None