import numpy as np

class IterativeWeightedBackwardThinning:
    """
    Algoritmo iterativo de Backward Thinning con redistribución de pesos.
    Elimina el punto más cercano a la media actual y recalcula.
    """

    def __init__(
        self,
        feature_map_X,
        feature_map_y=None,
        target_size: int = 10,
        alpha: float = 1.0,
    ):
        self.feature_map_X = feature_map_X
        self.feature_map_y = feature_map_y
        self.target_size = target_size
        self.alpha = alpha
        
        # Atributos de estado
        self.mu_px_ = None
        self._lifted_X = None
        self._lifted_y = None

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Calcula el embedding inicial y la media global."""
        self._lifted_X = self.feature_map_X(X)  # (n, d)
        self.mu_px_ = self._lifted_X.mean(axis=0)
        if self.feature_map_y is not None and y is not None:
            self._lifted_y = self.feature_map_y(y)
            self.mu_py_ = self._lifted_y.mean(axis=0)
        return self

    def _compute_kernel_matrix(self, A, B):
        """
        Calcula el kernel lineal en el espacio proyectado (lifted).
        k(x, y) = <phi(x), phi(y)>
        """
        return A @ B.T

    def transform(self, X: np.ndarray, y: np.ndarray = None):
        if self._lifted_X is None:
            raise RuntimeError("Debes llamar a fit() antes de transform().")

        n = len(X)
        indices_activos = list(range(n))
        weights = np.ones(n) / n
        
        # Copia del embedding para no modificar el original
        Phi_x = self._lifted_X.copy()
        Phi_y = self._lifted_y.copy() if self._lifted_y is not None else None

        while len(indices_activos) > self.target_size:
            # 1. Calcular la media pesada actual
            # mu_w = sum(w_i * phi_i)
            
            mu_w = np.mean(Phi_x[indices_activos], axis=0)
            # 2. Identificar el punto con "menor gradiente" (más cercano a la media)
            # Calculamos la norma de la diferencia entre cada punto activo y la media
            diffs_x = Phi_x[indices_activos] - mu_w
            distancias = np.linalg.norm(diffs_x, axis=1) 
            if Phi_y is not None:
                mu_w_y = np.mean(Phi_y[indices_activos], axis=0)
                diff_y = (Phi_y[indices_activos] - mu_w_y) 
                distancias += np.linalg.norm(diff_y, axis=1)  # Combinamos distancias de X e y


            # Índice local (dentro de indices_activos) del punto a eliminar
            idx_local_remove = np.argmin(distancias)
            idx_global_remove = indices_activos.pop(idx_local_remove)
            
            # 3. Redistribuir el peso del punto eliminado
            # w_r = weights[idx_global_remove]
            # if w_r > 0:
            #     # Calculamos el kernel entre los que se quedan (i) y el que se va (r)
            #     # k_ir = <phi_i, phi_r>
            #     phi_r = Phi_x[idx_global_remove]
            #     phi_y_r = Phi_y[idx_global_remove] if Phi_y is not None else None
            #     phi_activos = Phi_x[indices_activos]
            #     phi_y_activos = Phi_y[indices_activos] if Phi_y is not None else None
                
            #     # Aplicamos la fórmula: w_i = w_i + w_r * (k_ir^alpha / sum(k_jr^alpha))
            #     kx_ir = (phi_activos @ phi_r) ** self.alpha
            #     ky_ir = (phi_y_activos @ phi_y_r) ** self.alpha if Phi_y is not None else 0
            #     k_ir = kx_ir * ky_ir  if Phi_y is not None else kx_ir  # Si no hay y, solo usamos el kernel de X
            #     # Evitar división por cero si el kernel es muy pequeño
            #     denom = np.sum(k_ir)
            #     if abs(denom) > 1e-12:
            #         weights[indices_activos] += w_r * (k_ir / denom)
            #     else:
            #         # Si falla el kernel, redistribución uniforme simple
            #         weights[indices_activos] += w_r / len(indices_activos)

            # weights[idx_global_remove] = 0.0

        # Resultados finales
        self.coreset_indices_ = np.array(indices_activos)
        self.weights_ = weights[self.coreset_indices_]
        if y is not None:
            return X[self.coreset_indices_],  y[self.coreset_indices_], self.weights_,
        return X[self.coreset_indices_], None, self.weights_

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None):
        return self.fit(X, y).transform(X, y)

# ─────────────────────────────────────────────────────────────────────────────
# Ejemplo de uso
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X_toy = rng.standard_normal((50, 2))

    def simple_feature_map(X):
        # Una proyección simple a 5D para el ejemplo
        return np.hstack([X, X**2, (X[:, 0] * X[:, 1]).reshape(-1, 1)])

    # Queremos reducir de 50 a 10 puntos
    wbt = IterativeWeightedBackwardThinning(
        feature_map_X=simple_feature_map,
        target_size=10,
        alpha=1.0
    )

    X_core, w_core = wbt.fit_transform(X_toy)

    print("\n--- Resultados ---")
    print(f"Tamaño Coreset: {len(X_core)}")
    print(f"Suma de pesos: {w_core.sum():.4f} (Debe ser aprox 1.0)")
    print(f"Índices seleccionados: {wbt.coreset_indices_}")