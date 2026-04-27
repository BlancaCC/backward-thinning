"""
Kernel Thinning — implementación completa con soporte de fracción arbitraria
============================================================================

Extensión principal: thin_fraction(X, p, ...)
─────────────────────────────────────────────
Dado p ∈ (0, 1), selecciona un coreset de tamaño aproximadamente ⌊p·n⌋
usando la representación binaria de p:

    p = Σ_i  b_i / 2^i      b_i ∈ {0, 1}

Algoritmo de extracción de bits:
    a_i = 1/2^i
    Para cada nivel i = 1, 2, ...:
        si p_restante ≥ a_i:  b_i = 1,  p_restante -= a_i
        si no:                b_i = 0
    Parar cuando p_restante < tol  o  nivel máximo alcanzado

Selección en el árbol KT:
    • En el nivel i, si b_i = 1:
        – los hijos IZQUIERDOS de todos los nodos activos de ese nivel
          se añaden al coreset final.
        – los hijos DERECHOS siguen siendo explorados en el siguiente nivel.
    • Si b_i = 0:
        – todos los nodos del nivel se siguen explorando (ambos hijos
          bajan un nivel más), pero ninguno se añade al coreset en este paso.

Esto garantiza:
    |coreset| ≈ n · (b_1/2 + b_2/4 + ...) ≈ n · p   (exacto en media)

Sigue Dwivedi & Mackey (2024) con las sustituciones RFF selectivas
discutidas en el análisis teórico previo.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Union

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Utilidades de bajo nivel
# ═══════════════════════════════════════════════════════════════════════════════

def _as_matrix(feat: Callable, Z: np.ndarray) -> np.ndarray:
    """Aplica feat a Z y devuelve (n, D). Acepta callables vectorizados o por fila."""
    Z = np.atleast_2d(Z)
    try:
        out = feat(Z)
        if out.ndim == 1:
            out = out[np.newaxis, :]
    except Exception:
        out = np.stack([feat(z) for z in Z])
    return np.asarray(out, dtype=np.float64)


def _kval(kern_or_gram: Union[Callable, np.ndarray],
          X: np.ndarray, i: int, j: int) -> float:
    """k(X[i], X[j]) o Gram[i, j]."""
    if isinstance(kern_or_gram, np.ndarray):
        return float(kern_or_gram[i, j])
    return float(kern_or_gram(X[i], X[j]))


# ═══════════════════════════════════════════════════════════════════════════════
# b² — siempre kernel exacto
# ═══════════════════════════════════════════════════════════════════════════════

def _b2(k_rt, X: np.ndarray, i: int, j: int,
        k_y=None, Y: Optional[np.ndarray] = None) -> float:
    """
    b² = k_rt(xi,xi) + k_rt(xj,xj) - 2·k_rt(xi,xj)          [estándar]
    b²_joint = k_rt(xi,xi)·k_y(yi,yi) + k_rt(xj,xj)·k_y(yj,yj)
               - 2·k_rt(xi,xj)·k_y(yi,yj)                     [joint]
    """
    kii = _kval(k_rt, X, i, i)
    kjj = _kval(k_rt, X, j, j)
    kij = _kval(k_rt, X, i, j)
    if k_y is not None and Y is not None:
        yii = float(k_y(Y[i], Y[i]))
        yjj = float(k_y(Y[j], Y[j]))
        yij = float(k_y(Y[i], Y[j]))
        return kii * yii + kjj * yjj - 2.0 * kij * yij
    return kii + kjj - 2.0 * kij


# ═══════════════════════════════════════════════════════════════════════════════
# Δ — siempre phi_rt + acumuladores
# ═══════════════════════════════════════════════════════════════════════════════

def _delta_std(phi_i: np.ndarray, phi_j: np.ndarray,
               acc_S: np.ndarray, acc_Sp: np.ndarray) -> float:
    """Δ estándar con acumuladores — O(D)."""
    d_phi = phi_i - phi_j
    norm  = float(phi_j @ phi_j) - float(phi_i @ phi_i)
    return norm + float((acc_S - acc_Sp) @ d_phi)


def _delta_joint(phi_i: np.ndarray, phi_j: np.ndarray,
                 psi_i: np.ndarray, psi_j: np.ndarray,
                 PHI_S:  np.ndarray, PSI_S:  np.ndarray,
                 PHI_Sp: np.ndarray, PSI_Sp: np.ndarray) -> float:
    """Δ joint sin materializar el tensor phi_x ⊗ phi_y."""
    norm_i = float(phi_i @ phi_i) * float(psi_i @ psi_i)
    norm_j = float(phi_j @ phi_j) * float(psi_j @ psi_j)
    delta  = norm_j - norm_i

    if len(PHI_S):
        cxi = PHI_S @ phi_i;  cyi = PSI_S @ psi_i
        cxj = PHI_S @ phi_j;  cyj = PSI_S @ psi_j
        delta += float(np.dot(cxi, cyi) - np.dot(cxj, cyj))

    if len(PHI_Sp):
        cxi = PHI_Sp @ phi_i; cyi = PSI_Sp @ psi_i
        cxj = PHI_Sp @ phi_j; cyj = PSI_Sp @ psi_j
        delta -= float(np.dot(cxi, cyi) - np.dot(cxj, cyj))

    return delta


# ═══════════════════════════════════════════════════════════════════════════════
# get_swap_params — idéntico al paper
# ═══════════════════════════════════════════════════════════════════════════════

def _swap_params(psi: float, b: float, varpi: float):
    b2      = b * b
    a       = max(b * psi - 2.0 * math.log(2.0 / max(varpi, 1e-300)), b2)
    psi_new = psi + b2 * (1.0 + (b2 - 2.0 * a) * psi / (a * a + 1e-300))
    return a, psi_new


def _p_swap(delta: float, a: float) -> float:
    return min(1.0, 0.5 * max(0.0, 1.0 - delta / (a + 1e-300)))


# ═══════════════════════════════════════════════════════════════════════════════
# Estructura de coreset con acumuladores
# ═══════════════════════════════════════════════════════════════════════════════

class _Coreset:
    """
    Un único coreset con sus acumuladores RFF.

    Modo estándar : acc (D,)  = Σ phi_rt(z)
    Modo joint    : acc_x (Dx,), acc_y (Dy,)
                    mat_x (k, Dx), mat_y (k, Dy)  ← filas para delta_joint
    """

    __slots__ = ("indices", "joint", "acc", "acc_x", "acc_y",
                 "_rows_x", "_rows_y")

    def __init__(self, D: int, joint: bool = False, Dy: int = 0):
        self.indices: list[int] = []
        self.joint = joint
        if joint:
            self.acc_x = np.zeros(D,  dtype=np.float64)
            self.acc_y = np.zeros(Dy, dtype=np.float64)
            self._rows_x: list[np.ndarray] = []
            self._rows_y: list[np.ndarray] = []
            self.acc = None
        else:
            self.acc = np.zeros(D, dtype=np.float64)
            self.acc_x = self.acc_y = None
            self._rows_x = self._rows_y = None

    def push(self, idx: int, phi: np.ndarray,
             psi: Optional[np.ndarray] = None):
        self.indices.append(idx)
        if self.joint:
            self.acc_x += phi
            self.acc_y += psi
            self._rows_x.append(phi)
            self._rows_y.append(psi)
        else:
            self.acc += phi

    @property
    def mat_x(self) -> np.ndarray:
        return (np.stack(self._rows_x) if self._rows_x
                else np.empty((0, len(self.acc_x))))

    @property
    def mat_y(self) -> np.ndarray:
        return (np.stack(self._rows_y) if self._rows_y
                else np.empty((0, len(self.acc_y))))


# ═══════════════════════════════════════════════════════════════════════════════
# Extracción de la representación binaria de p
# ═══════════════════════════════════════════════════════════════════════════════

def _binary_levels(p: float, tol: float = 0.01, max_levels: int = 20
                   ) -> list[int]:
    """
    Devuelve la lista de bits b_i ∈ {0,1} tal que p ≈ Σ b_i / 2^i.

    Algoritmo:
        a_i = 1 / 2^i  (i = 1, 2, ...)
        residuo = p
        Para cada nivel i:
            si residuo ≥ a_i:  b_i = 1, residuo -= a_i
            si no:             b_i = 0
        Parar cuando residuo < tol  o  i > max_levels

    Examples
    --------
    p = 0.75 → [1, 1]         (1/2 + 1/4)
    p = 0.50 → [1]            (1/2)
    p = 0.25 → [0, 1]         (1/4)
    p = 0.60 → [1, 0, 0, 1]  (1/2 + 1/16 = 0.5625 ≈ 0.60, con tol ajustada)
    """
    if not (0 < p < 1):
        raise ValueError(f"p debe estar en (0,1), recibido p={p}")

    bits   = []
    resid  = p
    for i in range(1, max_levels + 1):
        a_i = 1.0 / (2 ** i)
        if resid >= a_i:
            bits.append(1)
            resid -= a_i
        else:
            bits.append(0)
        if resid < tol:
            break
    return bits


# ═══════════════════════════════════════════════════════════════════════════════
# Clase principal
# ═══════════════════════════════════════════════════════════════════════════════

class FlexibleKernelThinning:
    """
    Kernel Thinning con sustituciones RFF selectivas y soporte de joint kernel.

    Parameters
    ----------
    k_rt     : callable | np.ndarray
               Square-root kernel exacto de k_star. Usado SOLO en b².
    phi_rt   : callable
               RFF de k_rt. Usado SOLO en Δ (con acumuladores).
    k_star   : callable | np.ndarray
               Kernel objetivo para el greedy refinement final.
    phi_star : callable | None
               RFF de k_star para greedy aproximado (opcional).
    k_y      : callable | None
               Kernel de etiquetas; activa modo joint si no es None.
    phi_rt_y : callable | None
               RFF del square-root kernel de k_y (para Δ joint).
    rng      : int | np.random.Generator | None
    """

    def __init__(
        self,
        k_rt:     Union[Callable, np.ndarray],
        phi_rt:   Callable,
        k_star:   Union[Callable, np.ndarray],
        phi_star: Optional[Callable] = None,
        k_y:      Optional[Callable] = None,
        phi_rt_y: Optional[Callable] = None,
        rng:      Union[int, np.random.Generator, None] = None,
    ):
        self.k_rt     = k_rt
        self.phi_rt   = phi_rt
        self.k_star   = k_star
        self.phi_star = phi_star
        self.k_y      = k_y
        self.phi_rt_y = phi_rt_y
        self._joint   = k_y is not None
        self.rng = (rng if isinstance(rng, np.random.Generator)
                    else np.random.default_rng(rng))

    # ──────────────────────────────────────────────────────────────────────────
    # API pública: thin clásico (tamaños n/2^m)
    # ──────────────────────────────────────────────────────────────────────────

    def thin(self, X: np.ndarray,
             y: Optional[np.ndarray] = None,
             m: int = 1) -> np.ndarray:
        """
        Ejecuta Kernel Thinning clásico: coreset de tamaño ⌊n/2^m⌋.

        Parameters
        ----------
        X : (n, d)
        y : (n, p) | None   requerido si k_y is not None
        m : int              número de divisiones

        Returns
        -------
        indices : (⌊n/2^m⌋,)
        """
        X, y = self._validate(X, y)
        n    = len(X)

        PHI = _as_matrix(self.phi_rt, X)
        PSI = (_as_matrix(self.phi_rt_y, y)
               if self._joint and self.phi_rt_y is not None else None)

        candidates = self._kt_split(X, y, PHI, PSI, m)
        return self._greedy_select(candidates, X, y, PHI)

    # ──────────────────────────────────────────────────────────────────────────
    # API pública: thin_fraction (tamaño arbitrario ≈ p·n)
    # ──────────────────────────────────────────────────────────────────────────

    def thin_fraction(self, X: np.ndarray,
                      p: float,
                      y: Optional[np.ndarray] = None,
                      tol: float = 0.01,
                      max_levels: int = 20) -> np.ndarray:
        """
        Kernel Thinning con fracción arbitraria p ∈ (0, 1).

        Selecciona un coreset de tamaño aproximadamente ⌊p·n⌋ usando la
        representación binaria de p:

            p ≈ b_1/2 + b_2/4 + b_3/8 + ...

        En el árbol KT de nivel i:
          • Si b_i = 1: los hijos IZQUIERDOS de los nodos activos se añaden
                        al coreset final; los hijos DERECHOS siguen bajando.
          • Si b_i = 0: ambos hijos de cada nodo activo siguen bajando al
                        siguiente nivel (sin añadir nada al coreset todavía).

        Esto permite que el tamaño resultante sea cualquier múltiplo de n/2^L
        (donde L es la profundidad), cubriendo densamente (0, 1).

        Parameters
        ----------
        X          : (n, d)
        p          : fracción objetivo en (0, 1)
        y          : (n, q) | None   requerido si k_y is not None
        tol        : tolerancia para detener la expansión binaria de p
        max_levels : número máximo de niveles del árbol

        Returns
        -------
        indices : array de enteros, tamaño ≈ ⌊p·n⌋
        """
        X, y = self._validate(X, y)
        n    = len(X)

        bits = _binary_levels(p, tol=tol, max_levels=max_levels)
        L    = len(bits)

        print(f"   Representación binaria de p={p:.4f}: "
              f"bits={bits}  →  p_aprox={sum(b/2**(i+1) for i,b in enumerate(bits)):.4f}"
              f"  →  coreset objetivo ≈ {int(sum(b*n/2**(i+1) for i,b in enumerate(bits)))}")

        PHI = _as_matrix(self.phi_rt, X)
        PSI = (_as_matrix(self.phi_rt_y, y)
               if self._joint and self.phi_rt_y is not None else None)

        D   = PHI.shape[1]
        Dy  = PSI.shape[1] if PSI is not None else 0
        joint = self._joint and PSI is not None

        # ── Construir el árbol hasta profundidad L ─────────────────────────
        # levels[j] = lista de _Coreset (nodos activos en el nivel j)
        # Al comienzo solo hay un nodo raíz (nivel 0 = todos los puntos).
        levels: list[list[_Coreset]] = [
            [_Coreset(D, joint=joint, Dy=Dy)]
        ]
        psi_arr: list[list[float]] = [[0.0]]  # psi_{j}[nodo]

        for j in range(1, L + 1):
            # Doblar el número de nodos
            levels.append([
                _Coreset(D, joint=joint, Dy=Dy)
                for _ in range(2 * len(levels[j - 1]))
            ])
            psi_arr.append([0.0] * (2 * len(psi_arr[j - 1])))

        # ── Alimentar puntos uno a uno (pares) ────────────────────────────
        for step in range(n // 2):
            ia, ib = 2 * step, 2 * step + 1

            phi_a = PHI[ia]; phi_b = PHI[ib]
            psi_a = PSI[ia] if joint else None
            psi_b = PSI[ib] if joint else None

            # Añadir par al nivel 0
            levels[0][0].push(ia, phi_a, psi_a)
            levels[0][0].push(ib, phi_b, psi_b)

            # Propagar hacia abajo en el árbol
            for j in range(1, L + 1):
                period = 2 ** (j - 1)
                if (step + 1) % period != 0:
                    break

                n_parents = len(levels[j - 1])
                for lp in range(n_parents):
                    parent  = levels[j - 1][lp]
                    child_l = levels[j][2 * lp]
                    child_r = levels[j][2 * lp + 1]

                    p_idx = parent.indices
                    if len(p_idx) < 2:
                        continue

                    gi, gj = p_idx[-2], p_idx[-1]

                    # b²: kernel EXACTO
                    b2v = _b2(self.k_rt, X, gi, gj, k_y=self.k_y, Y=y)
                    b   = math.sqrt(max(b2v, 0.0))
                    a, psi_arr[j][lp] = _swap_params(psi_arr[j][lp], b, 0.5)

                    # Δ: phi_rt + acumuladores / lifting
                    phi_i = PHI[gi]; phi_j = PHI[gj]

                    if joint:
                        delta = _delta_joint(
                            phi_i, phi_j,
                            PSI[gi], PSI[gj],
                            parent.mat_x,  parent.mat_y,
                            child_l.mat_x, child_l.mat_y,
                        )
                    else:
                        delta = _delta_std(
                            phi_i, phi_j,
                            parent.acc, child_l.acc
                        )

                    # Swap
                    if self.rng.random() < _p_swap(delta, a):
                        gi, gj   = gj, gi
                        phi_i, phi_j = phi_j, phi_i

                    psi_i = PSI[gi] if joint else None
                    psi_j = PSI[gj] if joint else None

                    child_l.push(gi, phi_i, psi_i)
                    child_r.push(gj, phi_j, psi_j)

        # ── Recolección final usando los bits ─────────────────────────────
        #
        # "active" = lista de índices de nodo en el nivel j que siguen
        # siendo explorados (no han contribuido al coreset todavía).
        #
        # Nivel j corresponde al bit bits[j-1] (indexado desde 1).
        #
        # En el nivel j (bit b_j):
        #   • Si b_j = 1:
        #       – El hijo IZQUIERDO (2*lp) del nodo activo lp
        #         se añade al coreset.
        #       – El hijo DERECHO   (2*lp+1) pasa a ser activo en j+1.
        #   • Si b_j = 0:
        #       – Ambos hijos pasan a ser activos en j+1.
        #
        # El árbol en el nivel j tiene 2^j nodos (índices 0..2^j-1).
        # Los nodos activos siempre son índices válidos en levels[j].

        coreset_indices: list[int] = []
        active: list[int] = [0]   # índice de nodo en levels[0] (raíz)

        for j, bit in enumerate(bits, start=1):
            next_active: list[int] = []
            for lp in active:
                left_child  = 2 * lp
                right_child = 2 * lp + 1

                if bit == 1:
                    # Hijo izquierdo → va al coreset
                    coreset_indices.extend(levels[j][left_child].indices)
                    # Hijo derecho → sigue activo
                    next_active.append(right_child)
                else:
                    # Ambos hijos siguen activos
                    #next_active.append(left_child)
                    next_active.append(right_child)

            active = next_active

        return np.array(coreset_indices, dtype=int)

    # ──────────────────────────────────────────────────────────────────────────
    # Kernel Halving
    # ──────────────────────────────────────────────────────────────────────────

    def kernel_halving(self, X: np.ndarray,
                       y: Optional[np.ndarray] = None,
                       indices: Optional[np.ndarray] = None,
                       ) -> tuple[np.ndarray, np.ndarray]:
        """2-thinning. Devuelve (S+, S-) como índices globales en X."""
        X, y = self._validate(X, y)
        idx  = (np.arange(len(X)) if indices is None
                else np.asarray(indices, int))
        if len(idx) % 2:
            idx = idx[:-1]

        Xs  = X[idx]
        ys  = y[idx] if y is not None else None
        n   = len(idx)
        D   = _as_matrix(self.phi_rt, Xs[:1]).shape[1]
        Dy  = (_as_matrix(self.phi_rt_y, ys[:1]).shape[1]
               if self._joint and self.phi_rt_y is not None else 0)

        PHI = _as_matrix(self.phi_rt, Xs)
        PSI = (_as_matrix(self.phi_rt_y, ys)
               if self._joint and self.phi_rt_y is not None else None)

        cs_p = _Coreset(D, joint=(self._joint and PSI is not None), Dy=Dy)
        cs_m = _Coreset(D, joint=(self._joint and PSI is not None), Dy=Dy)
        psi_val = 0.0

        out_p, out_m = [], []

        for step in range(n // 2):
            li, lj = 2 * step, 2 * step + 1
            gi, gj = int(idx[li]), int(idx[lj])

            b2v = _b2(self.k_rt, Xs, li, lj, k_y=self.k_y, Y=ys)
            b   = math.sqrt(max(b2v, 0.0))
            a, psi_val = _swap_params(psi_val, b, 0.5)

            phi_i, phi_j = PHI[li], PHI[lj]

            if self._joint and PSI is not None:
                delta = _delta_joint(
                    phi_i, phi_j, PSI[li], PSI[lj],
                    cs_p.mat_x, cs_p.mat_y,
                    cs_m.mat_x, cs_m.mat_y,
                )
            else:
                delta = _delta_std(phi_i, phi_j, cs_p.acc, cs_m.acc)

            if self.rng.random() < _p_swap(delta, a):
                li, lj = lj, li
                gi, gj = gj, gi
                phi_i, phi_j = phi_j, phi_i

            out_p.append(gi);  out_m.append(gj)
            psi_i = PSI[li] if (self._joint and PSI is not None) else None
            psi_j = PSI[lj] if (self._joint and PSI is not None) else None
            cs_p.push(gi, PHI[li], psi_i)
            cs_m.push(gj, PHI[lj], psi_j)

        return np.array(out_p), np.array(out_m)

    # ──────────────────────────────────────────────────────────────────────────
    # kt-split (Algorithm 1a) — para thin() clásico
    # ──────────────────────────────────────────────────────────────────────────

    def _kt_split(self, X, y, PHI, PSI, m) -> list[np.ndarray]:
        """Produce 2^m listas de índices de tamaño ⌊n/2^m⌋."""
        n     = len(X)
        D     = PHI.shape[1]
        Dy    = PSI.shape[1] if PSI is not None else 0
        joint = self._joint and PSI is not None

        levels: list[list[_Coreset]] = [
            [_Coreset(D, joint=joint, Dy=Dy) for _ in range(2**j)]
            for j in range(m + 1)
        ]
        psi_arr = [[0.0] * max(1, 2 ** (j - 1)) for j in range(m + 1)]

        for step in range(n // 2):
            ia, ib = 2 * step, 2 * step + 1

            phi_a = PHI[ia]; phi_b = PHI[ib]
            psi_a = PSI[ia] if joint else None
            psi_b = PSI[ib] if joint else None

            levels[0][0].push(ia, phi_a, psi_a)
            levels[0][0].push(ib, phi_b, psi_b)

            for j in range(1, m + 1):
                period = 2 ** (j - 1)
                if (step + 1) % period != 0:
                    break

                for lp in range(2 ** (j - 1)):
                    parent  = levels[j - 1][lp]
                    lc_l    = 2 * lp
                    lc_r    = 2 * lp + 1
                    child_l = levels[j][lc_l]
                    child_r = levels[j][lc_r]

                    p_idx = parent.indices
                    if len(p_idx) < 2:
                        continue

                    gi, gj = p_idx[-2], p_idx[-1]

                    b2v = _b2(self.k_rt, X, gi, gj, k_y=self.k_y, Y=y)
                    b   = math.sqrt(max(b2v, 0.0))
                    a, psi_arr[j][lp] = _swap_params(psi_arr[j][lp], b, 0.5)

                    phi_i = PHI[gi]; phi_j = PHI[gj]

                    if joint:
                        delta = _delta_joint(
                            phi_i, phi_j,
                            PSI[gi], PSI[gj],
                            parent.mat_x,  parent.mat_y,
                            child_l.mat_x, child_l.mat_y,
                        )
                    else:
                        delta = _delta_std(
                            phi_i, phi_j,
                            parent.acc, child_l.acc
                        )

                    if self.rng.random() < _p_swap(delta, a):
                        gi, gj   = gj, gi
                        phi_i, phi_j = phi_j, phi_i

                    psi_i = PSI[gi] if joint else None
                    psi_j = PSI[gj] if joint else None

                    child_l.push(gi, phi_i, psi_i)
                    child_r.push(gj, phi_j, psi_j)

        return [np.array(levels[m][l].indices) for l in range(2**m)]

    # ──────────────────────────────────────────────────────────────────────────
    # Greedy refinement con k_star (Algorithm 1b) — para thin() clásico
    # ──────────────────────────────────────────────────────────────────────────

    def _greedy_select(self, candidates, X, y, PHI) -> np.ndarray:
        """Selecciona el candidato con menor MMD²(S, X)."""
        if self.phi_star is not None:
            Pstar   = _as_matrix(self.phi_star, X)
            mu_full = Pstar.mean(0)
            best_score, best = np.inf, candidates[0]
            for c in candidates:
                if not len(c):
                    continue
                diff = mu_full - Pstar[c].mean(0)
                s    = float(diff @ diff)
                if s < best_score:
                    best_score, best = s, c
        else:
            n = len(X)
            best_score, best = np.inf, candidates[0]
            for c in candidates:
                ns = len(c)
                if ns == 0:
                    continue
                kss = np.array([[_kval(self.k_star, X, int(a), int(b))
                                 for b in c] for a in c])
                ksx = np.array([[_kval(self.k_star, X, int(a), b)
                                 for b in range(n)] for a in c])
                s = float(kss.mean()) - 2.0 * float(ksx.mean())
                if s < best_score:
                    best_score, best = s, c
        return best

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _validate(self, X, y):
        X = np.asarray(X, np.float64)
        if X.ndim == 1:
            X = X[:, None]
        if self._joint:
            if y is None:
                raise ValueError("k_y != None requiere y.")
            y = np.asarray(y, np.float64)
            if y.ndim == 1:
                y = y[:, None]
        return X, y


# ═══════════════════════════════════════════════════════════════════════════════
# Constructores de RFF y kernels
# ═══════════════════════════════════════════════════════════════════════════════

def make_rff(bandwidth: float, D: int, d: int,
             kernel: str = "gaussian",
             rng: Union[int, np.random.Generator, None] = None) -> Callable:
    """
    phi: (n,d) -> (n,D)  con  phi(x)^T phi(y) ≈ k(x,y).
    """
    _rng = np.random.default_rng(rng)
    if kernel == "gaussian":
        Omega = _rng.normal(0.0, 1.0 / bandwidth, (D, d))
    elif kernel == "laplacian":
        Omega = _rng.standard_cauchy((D, d)) / bandwidth
    else:
        raise ValueError(f"kernel desconocido: '{kernel}'")
    b     = _rng.uniform(0.0, 2.0 * np.pi, D)
    scale = 1.0 / math.sqrt(D)

    def phi(X: np.ndarray) -> np.ndarray:
        return scale * np.cos(np.atleast_2d(X) @ Omega.T + b)

    return phi


def make_gaussian_kernel(bandwidth: float) -> Callable:
    """k(x,y) = exp(-‖x-y‖²/(2σ²))."""
    s2 = 2.0 * bandwidth ** 2
    def k(x, y):
        d = np.asarray(x) - np.asarray(y)
        return float(np.exp(-float(d @ d) / s2))
    return k


# ═══════════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time

    rng = np.random.default_rng(0)
    SEP = "─" * 64

    n, d, D = 1024, 30, 128
    X = rng.standard_normal((n, d))
    bw = 1.0

    k_rt     = make_gaussian_kernel(bw)
    phi_rt   = make_rff(bw, D, d, rng=1)
    k_star   = make_gaussian_kernel(bw * math.sqrt(2))
    phi_star = make_rff(bw * math.sqrt(2), D, d, rng=2)

    kt = KernelThinningRFF(k_rt=k_rt, phi_rt=phi_rt,
                           k_star=k_star, phi_star=phi_star, rng=42)

    # ──────────────────────────────────────────────────────────────
    # 1. thin() clásico
    # ──────────────────────────────────────────────────────────────
    print(SEP)
    print("1. thin() clásico — m=3")
    t0  = time.perf_counter()
    idx = kt.thin(X, m=3)
    print(f"   Tamaño: {len(idx)}  (esperado {n//8}={n//2**3})")
    print(f"   Tiempo: {time.perf_counter()-t0:.3f}s")

    # ──────────────────────────────────────────────────────────────
    # 2. thin_fraction() — varias fracciones
    # ──────────────────────────────────────────────────────────────
    print(SEP)
    print("2. thin_fraction() — fracciones arbitrarias")

    phi_eval = make_rff(bw * math.sqrt(2), 1024, d, rng=99)
    Phi_all  = phi_eval(X)
    mu_all   = Phi_all.mean(0)

    def mmd2_approx(subset_idx):
        if len(subset_idx) == 0:
            return float("inf")
        diff = mu_all - Phi_all[subset_idx].mean(0)
        return float(diff @ diff)

    for p_target in [0.5, 0.25, 0.75, 0.3, 0.1, 0.6]:
        print(f"\n   p = {p_target}")
        t0   = time.perf_counter()
        idx_f = kt.thin_fraction(X, p=p_target, tol=0.01)
        dt   = time.perf_counter() - t0

        rand_idx = rng.choice(n, len(idx_f), replace=False)
        mmd_kt   = mmd2_approx(idx_f)
        mmd_rnd  = mmd2_approx(rand_idx)
        mejora   = 100 * (mmd_rnd - mmd_kt) / mmd_rnd if mmd_rnd > 0 else 0

        print(f"   Tamaño real:   {len(idx_f)}  (esperado ≈ {int(p_target*n)})")
        print(f"   MMD²(KT,X):   {mmd_kt:.2e}   MMD²(rand,X): {mmd_rnd:.2e}"
              f"   mejora: {mejora:.1f}%")
        print(f"   Tiempo:        {dt:.3f}s")

    # ──────────────────────────────────────────────────────────────
    # 3. Representación binaria — verificación
    # ──────────────────────────────────────────────────────────────
    print(SEP)
    print("3. Verificación de _binary_levels")
    for p_test in [0.5, 0.25, 0.75, 0.125, 0.375, 0.6]:
        bits = _binary_levels(p_test, tol=0.001)
        p_rec = sum(b / 2**(i+1) for i, b in enumerate(bits))
        print(f"   p={p_test:.3f}  bits={bits}  p_reconstruido={p_rec:.4f}"
              f"  error={abs(p_test-p_rec):.4f}")