import numpy as np
from typing import List, Tuple, Generator

class EuclideanSpatialHash:
    """A spatial hashing structure for efficient vector insertion and lookup.
    
    This class discretizes n-dimensional space into cells of size 'epsilon'.
    It ensures that vectors are only inserted if no existing vector is within
    the Euclidean distance (L2 norm) specified by epsilon.

    Attributes:
        epsilon: The minimum Euclidean distance required between vectors.
        grid: A dictionary mapping cell coordinates to lists of vectors.
    """

    def __init__(self, epsilon: float):
        """Initializes the spatial hasher.

        Args:
            epsilon: The minimum Euclidean distance required between vectors.
        """
        self.epsilon = epsilon
        self.grid = {}

    def _get_cell_coords(self, vector: np.ndarray) -> Tuple[int, ...]:
        """Maps a continuous vector to discrete grid coordinates.

        Args:
            vector: The input n-dimensional vector.

        Returns:
            A tuple of integers representing the cell's multi-dimensional index.
        """
        return tuple(np.floor(vector / self.epsilon).astype(int))

    def insert(self, vector_list: List[float]) -> bool:
        """Inserts a vector if no other vector exists within epsilon distance.

        The algorithm checks the vector's current cell and all immediate 
        neighboring cells to guarantee the L2 distance constraint.

        Complexity Analysis:
            Time Complexity: O(3^d * K), where 'd' is the dimensionality and 'K' 
                is the average number of vectors per cell. In low dimensions 
                (d < 10), this is effectively O(1) relative to total vectors N.
            Memory Complexity: O(N * d), where 'N' is the number of successfully 
                inserted vectors. We only store points in occupied cells, 
                minimizing overhead for sparse data.

        Args:
            vector_list: The vector to be inserted.

        Returns:
            True if the vector was unique (distance >= epsilon) and inserted, 
            False otherwise.
        """
        vector = np.array(vector_list, dtype=float)
        cell_key = self._get_cell_coords(vector)
        dim = len(vector)

        # Iterate through all 3^d neighboring cells (current cell + neighbors)
        # to ensure the |x - z|_2 < epsilon condition is checked globally.
        for offset in np.ndindex(*(3,) * dim):
            # Shift offset from [0, 1, 2] to [-1, 0, 1] relative to center cell
            neighbor_cell = tuple(np.array(cell_key) + (np.array(offset) - 1))

            if neighbor_cell in self.grid:
                for candidate in self.grid[neighbor_cell]:
                    # Strict Euclidean distance check: ||x - z||_2 < epsilon
                    if np.linalg.norm(vector - candidate) < self.epsilon:
                        return False

        # If no neighbors are within epsilon, add to the grid
        if cell_key not in self.grid:
            self.grid[cell_key] = []
        self.grid[cell_key].append(vector)
        return True

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """Yields all stored vectors.

        Complexity Analysis:
            Time Complexity: O(M + N), where 'M' is the number of occupied 
                cells and 'N' is the total number of vectors.
            Memory Complexity: O(1) extra space beyond the generator state.

        Yields:
            The next stored np.ndarray vector in the structure.
        """
        for cell_list in self.grid.values():
            for vector in cell_list:
                yield vector

    def find_nearby(self, vector_input):
        """Searches for an existing vector within epsilon distance.

        Complexity Analysis:
            Time Complexity: O(3^d * K), where 'd' is dimensionality and 'K' 
                is average points per cell.
            Memory Complexity: O(1) additional space.

        Args:
            vector_input: The vector to check (list or np.ndarray).

        Returns:
            The first np.ndarray found within epsilon distance, or None.
        """
        vector = np.array(vector_input, dtype=float)
        cell_key = self._get_cell_coords(vector)
        dim = len(vector)

        # Check the current cell and all 3^d - 1 neighbors
        for offset in np.ndindex(*(3,) * dim):
            neighbor_cell = tuple(np.array(cell_key) + (np.array(offset) - 1))

            if neighbor_cell in self.grid:
                for candidate in self.grid[neighbor_cell]:
                    # Strict L2 check: ||x - z||_2 < epsilon
                    if np.linalg.norm(vector - candidate) < self.epsilon:
                        return candidate
        return None

# Example Usage:
# hasher = EuclideanSpatialHash(epsilon=0.1)
# hasher.insert([1.0, 1.0])