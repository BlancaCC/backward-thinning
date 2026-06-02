import numpy as np

class BackwardThinning:
    """Implements backward thinning for dataset reduction in feature space.
    
    This class reduces a dataset by iteratively removing points that minimize
    the product of squared L2 norms (L @ L.T) between the features and the 
    current cumulative error/residual in the feature space.
    """

    def __init__(self, feature_map_X, feature_map_y=None):
        """Initializes the thinner with feature mapping functions.

        Args:
            feature_map_X (callable): Function to map X data to feature space.
            feature_map_y (callable, optional): Function to map Y data to 
                feature space. Defaults to None.
        """
        self.feature_map_X = feature_map_X
        self.feature_map_y = feature_map_y

    def thin(self, n_to_rm, X, Y=None):
        """Reduces the dataset by removing n_to_rm samples.

        Args:
            n_to_rm (int): Number of samples to remove from the dataset.
            X (np.ndarray): Input features of shape (n_samples, n_dims).
            Y (np.ndarray, optional): Input targets of shape (n_samples, n_dims).
                Defaults to None.

        Returns:
            tuple: (Reduced X, Reduced Y) or (Reduced X, None).
        
        Raises:
            ValueError: If Y is provided but feature_map_y is not initialized.
        """
        if Y is not None and self.feature_map_y is None:
            raise ValueError("feature_map_y must be provided if Y is not None.")

        use_Y = Y is not None

        # Map to feature space [cite: 12]
        phi_X = self.feature_map_X(X).astype(np.float64)
        psi_Y = self.feature_map_y(Y).astype(np.float64) if use_Y else None

        n = phi_X.shape[0]
        mask = np.ones(n, dtype=bool)

        # Initialize targets with the mean of the features 
        error_x = np.mean(phi_X, axis=0)
        error_y = np.mean(psi_Y, axis=0) if use_Y else None

        for i in range(n_to_rm):
            # Calculate L = F - e for active indices 
            active_indices = np.where(mask)[0]
            L_x = phi_X[active_indices] - error_x
            
            # Compute squared norms (L @ L.T per row) 
            scores = np.sum(L_x * L_x, axis=1)

            if use_Y:
                L_y = psi_Y[active_indices] - error_y
                scores_y = np.sum(L_y * L_y, axis=1)
                # The norm of the tensor is the product of norms
                scores *= scores_y

            # Find the index that minimizes the residual 
            local_idx = np.argmin(scores)
            global_idx = active_indices[local_idx]

            # Update the error/target with the selected point's features 
            error_x -= phi_X[global_idx]
            
            if use_Y:
                    error_y -= psi_Y[global_idx]
            
            # Mark the point as removed [cite: 19, 23]
            mask[global_idx] = False

        return X[mask], (Y[mask] if use_Y else None)