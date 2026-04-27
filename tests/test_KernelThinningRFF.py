import unittest
import numpy as np
from typing import Callable
from subsampling_models import KernelThinningRff, make_rff, make_gaussian_kernel
from subsampling_models.KernelThinningRFF import _as_matrix

# Asumiendo que el código original está en un archivo llamado kernel_thinning.py
# o integrado en el mismo script. 
# Si es un archivo aparte: from kernel_thinning import KernelThinningRff, make_rff, make_gaussian_kernel

class TestKernelThinningRff(unittest.TestCase):

    def setUp(self):
        """Configuración de datos sintéticos y kernels para los tests."""
        self.n = 16  # Potencia de 2 para facilitar divisiones m=1, 2, 3
        self.d = 2
        self.rng = np.random.default_rng(42)
        
        # Datos X (features) e y (labels para modo joint)
        self.X = self.rng.standard_normal((self.n, self.d))
        self.y = self.rng.standard_normal((self.n, 1))
        
        # Parámetros de kernels
        self.bw = 1.0
        self.D = 50  # Dimensión RFF
        
        # Generadores
        self.k_rt = make_gaussian_kernel(self.bw)
        self.phi_rt = make_rff(self.bw, self.D, self.d, rng=self.rng)
        self.k_star = make_gaussian_kernel(self.bw)
        self.phi_star = make_rff(self.bw, self.D, self.d, rng=self.rng)

    def test_initialization(self):
        """Verifica que el constructor asigne correctamente los componentes."""
        kt = KernelThinningRff(self.k_rt, self.phi_rt, self.k_star)
        self.assertFalse(kt._joint)
        self.assertIsNone(kt.phi_star)
        
        # Test con modo joint
        kt_joint = KernelThinningRff(self.k_rt, self.phi_rt, self.k_star, k_y=self.k_rt)
        self.assertTrue(kt_joint._joint)

    def test_kernel_halving_shapes(self):
        """Verifica que kernel_halving devuelva dos sets disjuntos de tamaño n/2."""
        kt = KernelThinningRff(self.k_rt, self.phi_rt, self.k_star, rng=42)
        s_plus, s_minus = kt.kernel_halving(self.X)
        
        self.assertEqual(len(s_plus), self.n // 2)
        self.assertEqual(len(s_minus), self.n // 2)
        
        # Verificar que son disjuntos
        intersection = np.intersect1d(s_plus, s_minus)
        self.assertEqual(len(intersection), 0)
        
        # Verificar que los índices son válidos
        self.assertTrue(np.all(s_plus < self.n))

    def test_thin_output_size(self):
        """Verifica que thin devuelva el tamaño de coreset esperado (n / 2^m)."""
        m = 2
        kt = KernelThinningRff(self.k_rt, self.phi_rt, self.k_star, rng=42)
        indices = kt.thin(self.X, m=m)
        
        expected_size = self.n // (2**m)
        self.assertEqual(len(indices), expected_size)

    def test_joint_mode_execution(self):
        """Verifica que el modo joint (lifting tensors) funcione sin errores de dimensiones."""
        phi_rt_y = make_rff(self.bw, self.D, 1, rng=self.rng)
        kt = KernelThinningRff(
            k_rt=self.k_rt, 
            phi_rt=self.phi_rt, 
            k_star=self.k_star,
            k_y=self.k_rt,
            phi_rt_y=phi_rt_y,
            rng=42
        )
        
        # Debe correr sin lanzar excepciones de producto tensorial
        indices = kt.thin(self.X, y=self.y, m=1)
        self.assertEqual(len(indices), self.n // 2)

    def test_greedy_selection_modes(self):
        """Compara que el greedy funcione tanto con phi_star (RFF) como con k_star (Exacto)."""
        # Caso 1: Greedy exacto (k_star)
        kt_exact = KernelThinningRff(self.k_rt, self.phi_rt, self.k_star, rng=42)
        idx_exact = kt_exact.thin(self.X, m=1)
        
        # Caso 2: Greedy aproximado (phi_star)
        kt_approx = KernelThinningRff(self.k_rt, self.phi_rt, self.k_star, phi_star=self.phi_star, rng=42)
        idx_approx = kt_approx.thin(self.X, m=1)
        
        self.assertIsInstance(idx_exact, np.ndarray)
        self.assertIsInstance(idx_approx, np.ndarray)
        self.assertEqual(len(idx_exact), self.n // 2)

    def test_validation_errors(self):
        """Verifica que se lancen errores si falta 'y' en modo joint."""
        kt = KernelThinningRff(self.k_rt, self.phi_rt, self.k_star, k_y=self.k_rt)
        with self.assertRaises(ValueError):
            kt.thin(self.X, y=None)

    def test_reproducibility(self):
        """Verifica que con el mismo seed el resultado sea idéntico."""
        kt1 = KernelThinningRff(self.k_rt, self.phi_rt, self.k_star, rng=123)
        kt2 = KernelThinningRff(self.k_rt, self.phi_rt, self.k_star, rng=123)
        
        idx1 = kt1.thin(self.X, m=1)
        idx2 = kt2.thin(self.X, m=1)
        
        np.testing.assert_array_equal(idx1, idx2)

    def test_as_matrix_utility(self):
        """Verifica la robustez de la utilidad _as_matrix frente a funciones vectorizadas o no."""
        
        # Función que solo acepta un vector a la vez
        def scalar_phi(x):
            return np.array([x[0]**2, x[1]**2])
            
        # Función que acepta matrices (vectorizada)
        def vector_phi(X):
            return X**2
            
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        res_s = _as_matrix(scalar_phi, data)
        res_v = _as_matrix(vector_phi, data)
        
        self.assertEqual(res_s.shape, (2, 2))
        self.assertEqual(res_v.shape, (2, 2))
        np.testing.assert_array_almost_equal(res_s, res_v)

if __name__ == '__main__':
    unittest.main()