import unittest
import socket
import numpy as np
from datasets import get_data

def is_connected():
    """Checks for an active internet connection.

    Returns:
        bool: True if connected to the internet, False otherwise.
    """
    try:
        # Attempt to connect to Google's DNS to verify network availability
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

class TestGetDataReal(unittest.TestCase):
    """Integration tests for the get_data function using real OpenML calls."""

    @unittest.skipIf(not is_connected(), "No internet connection: skipping real OpenML download test")
    def test_valid_call_real(self):
        """Tests successful data retrieval and preprocessing from OpenML.

        Verifies that the returned features and labels are correctly formatted 
        as NumPy arrays.
        """
        try:
            # Attempt to download a small dataset (e.g., classification index 1)
            X, y, name, tid = get_data('classification', 1, verbose=False) 
            
            self.assertIsNotNone(name)
            # Verify that both X and y are returned as NumPy arrays
            self.assertIsInstance(X, np.ndarray)
            self.assertIsInstance(y, np.ndarray)
            
        except Exception as e:
            self.fail(f"get_data raised an exception unexpectedly: {e}")

    def test_invalid_problem(self):
        """Tests that an invalid problem type correctly triggers a ValueError.

        This test does not require internet connectivity as it validates 
        internal logic.
        """
        with self.assertRaises(ValueError):
            get_data('clustering', 1, verbose=False)  # Invalid problem type

if __name__ == '__main__':
    unittest.main()