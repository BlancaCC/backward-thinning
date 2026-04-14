import unittest
import numpy as np
from utils import EuclideanSpatialHash 

class TestEuclideanSpatialHash(unittest.TestCase):
    """Unit tests for EuclideanSpatialHash focusing on L2 distance and efficiency."""

    def setUp(self):
        """Initialize a hasher with a fixed epsilon for testing."""
        self.epsilon = 1.0
        self.hasher = EuclideanSpatialHash(epsilon=self.epsilon)

    def test_initial_insertion(self):
        """Test that a new vector is successfully inserted."""
        self.assertTrue(self.hasher.insert([0.0, 0.0]))

    def test_rejection_within_epsilon(self):
        """Test that vectors within L2 distance < epsilon are rejected."""
        self.hasher.insert([0.0, 0.0])
        # Distance is 0.5, which is < epsilon (1.0)
        self.assertFalse(self.hasher.insert([0.5, 0.0]))

    def test_acceptance_outside_epsilon(self):
        """Test that vectors with L2 distance >= epsilon are accepted."""
        self.hasher.insert([0.0, 0.0])
        # Distance is 1.1, which is > epsilon (1.0)
        self.assertTrue(self.hasher.insert([1.1, 0.0]))

    def test_euclidean_precision(self):
        """Test strict L2 norm (diagonal distance).
        
        A point at (0.7, 0.7) has an L1 distance of 1.4 but an L2 distance 
        of ~0.99. It should be rejected if epsilon = 1.0.
        """
        self.hasher.insert([0.0, 0.0])
        # L2 distance = sqrt(0.7^2 + 0.7^2) = 0.9899...
        self.assertFalse(self.hasher.insert([0.7, 0.7]))

    def test_neighboring_cells(self):
        """Test that the hasher correctly checks adjacent grid cells.
        
        Two points can be very close but fall into different integer grid cells.
        Point A: (0.9, 0.9) -> Cell (0, 0)
        Point B: (1.1, 1.1) -> Cell (1, 1)
        Dist: ~0.28, should be rejected.
        """
        self.hasher.insert([0.9, 0.9])
        self.assertFalse(self.hasher.insert([1.1, 1.1]))

    def test_iteration(self):
        """Test that the __iter__ method yields all inserted vectors."""
        points = [[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]]
        for p in points:
            self.hasher.insert(p)
        
        extracted = list(self.hasher)
        self.assertEqual(len(extracted), 3)
        # Verify one of the points exists in the output
        np.testing.assert_array_equal(extracted[0], np.array([0.0, 0.0]))

    def test_high_dimensionality(self):
        """Test that the structure handles higher dimensions (e.g., 4D)."""
        hasher_4d = EuclideanSpatialHash(epsilon=1.0)
        v1 = [0, 0, 0, 0]
        v2 = [2, 2, 2, 2]
        self.assertTrue(hasher_4d.insert(v1))
        self.assertTrue(hasher_4d.insert(v2))
        # v3 is very close to v1 in 4D space
        v3 = [0.1, 0.1, 0.1, 0.1]
        self.assertFalse(hasher_4d.insert(v3))
        
    def test_find_nearby_returns_correct_vector(self):
        """Verifies that find_nearby returns the actual stored array."""
        v_orig = np.array([10.0, 10.0])
        self.hasher.insert(v_orig.tolist())
        
        # Query with a slightly different vector
        v_query = [10.05, 10.05]
        v_match = self.hasher.find_nearby(v_query)
        
        self.assertIsNotNone(v_match)
        np.testing.assert_array_equal(v_match, v_orig)

    def test_find_nearby_returns_none_if_far(self):
        """Verifies that find_nearby returns None if no vector is within epsilon."""
        self.hasher.insert([0.0, 0.0])
        self.assertIsNone(self.hasher.find_nearby([5.0, 5.0]))

if __name__ == "__main__":
    unittest.main()