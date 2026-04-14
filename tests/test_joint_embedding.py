import unittest
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from utils import compute_gamma_scale, build_joint_embedding

class JointEmbeddingTest(unittest.TestCase):
    """Unit tests for the Joint Embedding Module."""

    def setUp(self):
        """Sets up sample data for classification and regression tasks."""
        # 4 samples, 2 features
        self.X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [1.0, 1.0]])
        # 3 distinct classes
        self.y_class = np.array([0, 1, 2, 0])
        # Continuous targets
        self.y_reg = np.array([0.5, 1.5, 2.5, 3.5])

    def test_compute_gamma_scale_returns_correct_heuristic(self):
        """Verifies the 'scale' gamma calculation: 1 / (n_features * var(X))."""
        variance = np.var(self.X)
        n_features = self.X.shape[1]
        expected_gamma = 1.0 / (n_features * variance + 1e-12)
        
        actual_gamma = compute_gamma_scale(self.X)
        self.assertAlmostEqual(actual_gamma, expected_gamma, places=7)

    def test_classification_psi_is_orthogonal(self):
        """Validates that target embeddings for classification are orthonormal."""
        _, psi_y = build_joint_embedding(
            self.X, self.y_class, problem="classification"
        )
        
        # Test unique classes to verify the basis vectors
        unique_labels = np.array([0, 1, 2])
        embeddings = psi_y(unique_labels)
        
        # A set of vectors is orthonormal if G = Psi * Psi^T is the Identity Matrix
        # This confirms dot(v_i, v_j) = 0 for i != j AND dot(v_i, v_i) = 1
        gramian_matrix = np.matmul(embeddings, embeddings.T)
        identity_matrix = np.eye(len(unique_labels))
        
        np.testing.assert_array_almost_equal(
            gramian_matrix, 
            identity_matrix, 
            decimal=7,
            err_msg="Classification embeddings (psi) are not orthonormal."
        )

    def test_regression_psi_output_dimension(self):
        """Ensures regression target mapping projects to the RFF space (200)."""
        _, psi_y = build_joint_embedding(
            self.X, self.y_reg, problem="regression"
        )
        
        embeddings = psi_y(self.y_reg)
        # Expected components based on RBFSampler(n_components=200)
        self.assertEqual(embeddings.shape, (len(self.y_reg), 200))

    def test_invalid_problem_type_raises_value_error(self):
        """Checks that unsupported problem types trigger a ValueError."""
        with self.assertRaisesRegex(ValueError, "Invalid problem type"):
            build_joint_embedding(self.X, self.y_class, problem="clustering")

if __name__ == "__main__":
    unittest.main()