import numpy as np
import itertools
from typing import List, Any, Callable, Dict, Optional
from utils import EuclideanSpatialHash

class BackwardThinning:
    """Implements the Backward Thinning algorithm for coreset construction.
    
    This version delegates all spatial proximity logic to EuclideanSpatialHash,
    focusing strictly on the combinatorial logic and point filtering.
    """

    def __init__(self, epsilon: float):
        """Initializes the optimizer with a specific tolerance.

        Args:
            epsilon: The tolerance for the L2 distance uniqueness check.
        """
        self.epsilon = epsilon
        # We use our specialized structure as the 'Tree' T_epsilon
        self.tree = EuclideanSpatialHash(epsilon)
        # G[V_S] storage: Maps a unique hash of the vector to the point group
        self.group_store: Dict[int, List[Any]] = {}

    def _get_vector_id(self, vector: np.ndarray) -> int:
        """Generates a stable hash for a vector to use as a dictionary key."""
        return hash(vector.tobytes())

    def run(self, 
            X: List[Any], 
            phi: Callable[[Any], np.ndarray], 
            m: int) -> List[Any]:
        """Executes the removal algorithm with tree-based filtering.

        Complexity Analysis:
            Time: O(binom(n, m) * (d + 3^d)) for population. 
                  O(n * (d + 3^d)) for filtering.
            Memory: O(N_unique * d) for the spatial hash storage.

        Args:
            X: Input set of points {x_1, ..., x_n}.
            phi: Feature map Φ: X -> H.
            m: Combination depth.

        Returns:
            X_final: The optimal coreset.
        """
        n = len(X)
        embeddings = [phi(x) for x in X]
        
        # 1. Global Mean Embedding
        # mu_p = (1/n) * sum(phi(x_i))
        mu_p = np.mean(embeddings, axis=0)

        # 2. Combinatorial Tree Population
        for subset_indices in itertools.combinations(range(n), m):
            S_points = [X[i] for i in subset_indices]
            S_embeddings = [embeddings[i] for i in subset_indices]
            
            v_s = np.sum(S_embeddings, axis=0) -(m * mu_p)
            
            # Insert into the spatial hash (T_epsilon)
            # EuclideanSpatialHash.insert returns True only if unique within epsilon
            if self.tree.insert(v_s.tolist()):
                # If unique, we store the metadata G[V_S]
                v_id = self._get_vector_id(v_s)
                self.group_store[v_id] = S_points

        # 3. Identification of points to remove (R)
        R = set()
        for i, x in enumerate(X):
            target_v = mu_p - embeddings[i]
            
            # Here we check if mu_p - phi(x) exists in T_epsilon
            # Note: We need a find/query method in EuclideanSpatialHash 
            # to retrieve the original V_S that matched within epsilon.
            match_v = self.tree.find_nearby(target_v)
            
            if match_v is not None:
                # Add the point x and its generating group G[V_S] to R
                R.add(tuple(X[i]))
                v_id = self._get_vector_id(match_v)
                for member in self.group_store.get(v_id, []):
                    R.add(tuple(member))
        # 4. Final Coreset: X \ R
        # Using identity check or hashable check for points in X
        return [x for x in X if tuple(x) not in R]