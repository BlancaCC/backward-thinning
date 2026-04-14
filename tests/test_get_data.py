import unittest
import socket
from datasets import get_data
import numpy as np

def is_connected():
    try:
        # Intenta conectar al DNS de Google para ver si hay red
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

class TestGetDataReal(unittest.TestCase):

    @unittest.skipIf(not is_connected(), "No internet connection: skipping real OpenML download test")
    def test_valid_call_real(self):
        """Prueba real: descarga datos reales de OpenML."""
        # Intentamos bajar un dataset pequeño (ej. ID 1 de clasificación)
        try:
            X, y, name, tid = get_data('classification', 0) 
            
            self.assertIsNotNone(name)
            self.assertIsInstance(X, np.ndarray)
            self.assertIsInstance(y, np.ndarray)
            print(f"\n✅ Success! Downloaded: {name} (Task ID: {tid})")
            
        except Exception as e:
            self.fail(f"get_data raised an exception unexpectedly: {e}")

    def test_invalid_problem(self):
        """Esta sigue siendo válida sin internet porque falla por lógica interna."""
        with self.assertRaises(ValueError):
            get_data('clustering', 1)

if __name__ == '__main__':
    unittest.main()