import numpy as np

class KernelThinning:
    """
    Implementación de Kernel Thinning (Dwivedi & Mackey).
    Compatible con configuraciones supervisadas (Phi_X, Psi_y) y 
    no supervisadas (solo Phi_X).
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.indices_ = None

    def _get_kernel_val(self, i, j, Phi_X, Psi_y=None):
        """
        Calcula el valor del kernel. Si Psi_y es None, actúa como 
        kernel no supervisado sobre Phi_X.
        """
        # Kernel sobre el espacio proyectado X
        k_val = np.dot(Phi_X[i], Phi_X[j])
        
        # Si hay labels/proyecciones de y, aplicamos el producto (Joint Kernel)
        if Psi_y is not None:
            k_val *= np.dot(Psi_y[i], Psi_y[j])
            
        return k_val

    def _halve(self, current_indices, Phi_X, Psi_y=None):
        """
        Ronda de reducción estocástica (Kernel Halving).
        """
        n = len(current_indices)
        if n % 2 != 0:
            last_idx = current_indices[-1]
            current_indices = current_indices[:-1]
            n = len(current_indices)
        else:
            last_idx = None

        perm = np.random.permutation(n)
        shuffled_idx = current_indices[perm]
        pairs = shuffled_idx.reshape(-1, 2)
        
        selected_indices = []
        delta_accum = 0.0 

        for i in range(len(pairs)):
            idx1, idx2 = pairs[i]
            
            # Cálculo de similitudes en el RKHS
            k11 = self._get_kernel_val(idx1, idx1, Phi_X, Psi_y)
            k22 = self._get_kernel_val(idx2, idx2, Phi_X, Psi_y)
            k12 = self._get_kernel_val(idx1, idx2, Phi_X, Psi_y)
            
            # Distancia al cuadrado en el RKHS entre los dos puntos:
            # ||phi(x1) - phi(x2)||^2 = k11 + k22 - 2*k12
            sq_dist = max(0, k11 + k22 - 2 * k12)
            
            # Moneda sesgada para balancear el error acumulado (delta_accum)
            # Intentamos que la esperanza del error sea cero.
            denom = np.sqrt(sq_dist + 1e-9)
            prob = 0.5 * (1.0 - (delta_accum / denom))
            prob = np.clip(prob, 0.05, 0.95) 

            if np.random.rand() < prob:
                selected_indices.append(idx1)
                delta_accum += (k11 - k12)
            else:
                selected_indices.append(idx2)
                delta_accum += (k12 - k22)
            
            # Factor de olvido para evitar saturación del error acumulado
            delta_accum *= 0.9

        if last_idx is not None:
            selected_indices.append(last_idx)

        return np.array(selected_indices)

    def thin(self, Phi_X, Psi_y=None, target_size=None):
        """
        Reduce el conjunto hasta alcanzar el target_size.
        
        Args:
            Phi_X: Mapeo de características de X (n, d).
            Psi_y: Mapeo de características de y (n, q) o None.
            target_size: Tamaño final deseado.
        """
        n = Phi_X.shape[0]
        if target_size is None:
            target_size = n // 2
            
        current_indices = np.arange(n)
        
        # El halving reduce el tamaño a la mitad en cada paso: O(n log n)
        while len(current_indices) > target_size:
            current_indices = self._halve(current_indices, Phi_X, Psi_y)
            
            # Si el siguiente halving nos deja muy por debajo del target, 
            # hacemos un recorte aleatorio final o paramos.
            if len(current_indices) / 2 < target_size * 0.7:
                break
                
        # Ajuste final exacto si es necesario
        if len(current_indices) > target_size:
            current_indices = np.random.choice(current_indices, target_size, replace=False)
            
        self.indices_ = np.sort(current_indices)
        return self.indices_