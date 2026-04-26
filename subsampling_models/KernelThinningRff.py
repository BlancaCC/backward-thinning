"""
Kernel Thinning — implementación completa y eficiente
======================================================

Sigue Dwivedi & Mackey (2024) con las sustituciones RFF selectivas
discutidas en el análisis teórico previo:

  Regla de uso de krt vs phi_rt
  ─────────────────────────────
  • b²   → krt EXACTO  (O(d), dos evaluaciones puntuales, sin suma;
                         phi_rt daría O(D) > O(d) si D > d y añade sesgo)
  • Δ    → phi_rt + ACUMULADORES (O(D) por paso vs O(|S|·d) exacto;
                                   única ganancia asintótica real)
  • Refinamiento greedy (Alg. 1b) → k_star EXACTO o phi_star si disponible

  Tensores de levantamiento (lifting tensors) — kernel supervisado
  ────────────────────────────────────────────────────────────────
  k_joint((x,u),(x',u')) = k_x(x,x') · k_y(u,u')
  phi_joint = phi_x ⊗ phi_y  ∈ R^{Dx·Dy}

  Nunca se materializa el producto tensorial. Se explotan dos identidades:

  (1) Norma:
      ‖phi_joint(p)‖² = ‖phi_x(x)‖² · ‖phi_y(u)‖²  → O(Dx+Dy)

  (2) Producto escalar en el espacio tensorial:
      phi_joint(z)·phi_joint(p)
        = (phi_x(z)·phi_x(x)) · (phi_y(z)·phi_y(u))  → O(Dx+Dy) por par

  (3) Δ joint vectorizado:
      Δ = Σ_{z∈S}[(phi_x(z)·phi_x(i))·(phi_y(z)·phi_y(i))
                  -(phi_x(z)·phi_x(j))·(phi_y(z)·phi_y(j))]
        - Σ_{z∈S'}[...]
      → guardamos PHI_S (|S|,Dx) y PSI_S (|S|,Dy) por coreset
      → Δ = sum( (PHI_S @ phi_i)*(PSI_S @ psi_i)
                -(PHI_S @ phi_j)*(PSI_S @ psi_j) )
      → O(|S|·(Dx+Dy))  en vez de O(|S|·Dx·Dy)

Argumentos del constructor
──────────────────────────
  k_rt     : callable(x,x')->float | np.ndarray (Gram n×n)
             Square-root kernel EXACTO. SOLO para b².
  phi_rt   : callable X(n,d)->Φ(n,D) o x(d,)->φ(D,)
             RFF de k_rt. SOLO para Δ con acumuladores.
  k_star   : callable(x,x')->float | np.ndarray
             Kernel objetivo. Para el refinamiento greedy.
  phi_star : callable | None
             RFF de k_star (opcional).
  k_y      : callable(u,u')->float | None  → activa modo joint
  phi_rt_y : callable U(n,p)->Ψ(n,Dy) | None
             RFF del square-root kernel de k_y.
  rng      : int | np.random.Generator | None
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Union

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Utilidades de bajo nivel
# ═══════════════════════════════════════════════════════════════════════════════

def _as_matrix(feat: Callable, Z: np.ndarray) -> np.ndarray:
    """Aplica feat a Z y devuelve (n, D).  Acepta callables vectorizados o por fila."""
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

    Kernel EXACTO en ambos casos: O(d), sin suma, sin ganancia RFF.
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
    """
    Δ estándar con acumuladores — O(D).

    Δ = ‖phi_j‖² - ‖phi_i‖²  +  (acc_S - acc_Sp) · (phi_i - phi_j)

    acc_S  = Σ_{z∈S}  phi_rt(z)    mantenido incrementalmente
    acc_Sp = Σ_{z∈S'} phi_rt(z)
    """
    d_phi = phi_i - phi_j
    norm  = float(phi_j @ phi_j) - float(phi_i @ phi_i)
    return norm + float((acc_S - acc_Sp) @ d_phi)


def _delta_joint(phi_i: np.ndarray, phi_j: np.ndarray,
                 psi_i: np.ndarray, psi_j: np.ndarray,
                 PHI_S:  np.ndarray, PSI_S:  np.ndarray,
                 PHI_Sp: np.ndarray, PSI_Sp: np.ndarray) -> float:
    """
    Δ joint sin materializar el tensor phi_x ⊗ phi_y.

    Identidad explotada:
      phi_joint(z) · phi_joint(p) = (phi_x(z)·phi_x(x)) · (phi_y(z)·phi_y(u))

    Término diagonal:
      norm_j - norm_i  con  norm_p = ‖phi_x(p)‖²·‖phi_y(p)‖²

    Suma sobre S:
      (PHI_S @ phi_i) ⊙ (PSI_S @ psi_i)  →  elemento a elemento  →  escalar
      coste O(|S|·(Dx+Dy))

    PHI_S : (|S|, Dx)    PSI_S : (|S|, Dy)
    """
    norm_i = float(phi_i @ phi_i) * float(psi_i @ psi_i)
    norm_j = float(phi_j @ phi_j) * float(psi_j @ psi_j)
    delta  = norm_j - norm_i

    if len(PHI_S):
        cxi = PHI_S @ phi_i;  cyi = PSI_S @ psi_i   # (|S|,)
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
# Clase principal
# ═══════════════════════════════════════════════════════════════════════════════

class KernelThinningRff:
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
    # API pública
    # ──────────────────────────────────────────────────────────────────────────

    def thin(self, X: np.ndarray,
             y: Optional[np.ndarray] = None,
             m: int = 1) -> np.ndarray:
        """
        Ejecuta Kernel Thinning completo.

        Parameters
        ----------
        X : (n, d)
        y : (n, p) | None   requerido si k_y is not None
        m : int              número de divisiones; coreset de tamaño ⌊n/2^m⌋

        Returns
        -------
        indices : (⌊n/2^m⌋,)
        """
        X, y = self._validate(X, y)
        n    = len(X)

        # Precómputo embeddings — O(nD)
        PHI = _as_matrix(self.phi_rt, X)
        PSI = (_as_matrix(self.phi_rt_y, y)
               if self._joint and self.phi_rt_y is not None else None)

        # kt-split → 2^m candidatos
        candidates = self._kt_split(X, y, PHI, PSI, m)

        # Greedy refinement con k_star
        return self._greedy_select(candidates, X, y, PHI)

    def kernel_halving(self, X: np.ndarray,
                       y: Optional[np.ndarray] = None,
                       indices: Optional[np.ndarray] = None,
                       ) -> tuple[np.ndarray, np.ndarray]:
        """
        2-thinning. Devuelve (S+, S-) como índices globales en X.
        """
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
    # kt-split  (Algorithm 1a)
    # ──────────────────────────────────────────────────────────────────────────

    def _kt_split(self, X, y, PHI, PSI, m) -> list[np.ndarray]:
        """Produce 2^m listas de índices de tamaño ⌊n/2^m⌋."""
        n     = len(X)
        D     = PHI.shape[1]
        Dy    = PSI.shape[1] if PSI is not None else 0
        joint = self._joint and PSI is not None

        # levels[j] = lista de 2^j _Coreset
        levels: list[list[_Coreset]] = [
            [_Coreset(D, joint=joint, Dy=Dy) for _ in range(2**j)]
            for j in range(m + 1)
        ]

        # psi_{j,l} indexado [j][l]  (nivel 0 tiene 1 entrada ficticia)
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

                for lp in range(2 ** (j - 1)):        # índice del padre
                    parent  = levels[j - 1][lp]
                    lc_l    = 2 * lp                   # hijo izquierdo
                    lc_r    = 2 * lp + 1               # hijo derecho
                    child_l = levels[j][lc_l]
                    child_r = levels[j][lc_r]

                    p_idx = parent.indices
                    if len(p_idx) < 2:
                        continue

                    gi, gj = p_idx[-2], p_idx[-1]

                    # ── b²: kernel EXACTO ──────────────────────────────────
                    b2v = _b2(self.k_rt, X, gi, gj, k_y=self.k_y, Y=y)
                    b   = math.sqrt(max(b2v, 0.0))
                    a, psi_arr[j][lp] = _swap_params(psi_arr[j][lp], b, 0.5)

                    # ── Δ: phi_rt + acumuladores / lifting ─────────────────
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

                    # ── Swap ───────────────────────────────────────────────
                    if self.rng.random() < _p_swap(delta, a):
                        gi, gj   = gj, gi
                        phi_i, phi_j = phi_j, phi_i

                    psi_i = PSI[gi] if joint else None
                    psi_j = PSI[gj] if joint else None

                    child_l.push(gi, phi_i, psi_i)
                    child_r.push(gj, phi_j, psi_j)

        return [np.array(levels[m][l].indices) for l in range(2**m)]

    # ──────────────────────────────────────────────────────────────────────────
    # Greedy refinement con k_star  (Algorithm 1b)
    # ──────────────────────────────────────────────────────────────────────────

    def _greedy_select(self, candidates, X, y, PHI) -> np.ndarray:
        """
        Selecciona el candidato con menor MMD²(S, X).

        Con phi_star: MMD² ≈ ‖mean_phi*(X) - mean_phi*(S)‖²  → O(n·D*)
        Sin phi_star: evaluación exacta de k_star             → O(n·|S|·d)
        """
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
                # MMD²(S,X) ∝ mean_S(k_star) - 2·mean_{S×X}(k_star)
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
# Constructores de RFF y kernels listos para usar
# ═══════════════════════════════════════════════════════════════════════════════

def make_rff(bandwidth: float, D: int, d: int,
             kernel: str = "gaussian",
             rng: Union[int, np.random.Generator, None] = None) -> Callable:
    """
    phi: (n,d) -> (n,D)  con  phi(x)^T phi(y) ≈ k(x,y).

    kernel="gaussian"  : Omega_j ~ N(0, 1/sigma²)
    kernel="laplacian" : Omega_j ~ Cauchy(0, 1/gamma)
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
# Demo y pruebas
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time

    rng = np.random.default_rng(0)
    SEP = "─" * 64

    # ──────────────────────────────────────────────────────────────
    # 1. Modo estándar
    # ──────────────────────────────────────────────────────────────
    print(SEP)
    print("1. Modo estándar")
    n, d, D = 1024, 30, 128
    X = rng.standard_normal((n, d))
    bw = 1.0

    k_rt     = make_gaussian_kernel(bw)
    phi_rt   = make_rff(bw, D, d, rng=1)
    k_star   = make_gaussian_kernel(bw * math.sqrt(2))
    phi_star = make_rff(bw * math.sqrt(2), D, d, rng=2)

    kt = KernelThinning(k_rt=k_rt, phi_rt=phi_rt,
                        k_star=k_star, phi_star=phi_star, rng=42)

    m = 3
    t0  = time.perf_counter()
    idx = kt.thin(X, m=m)
    dt  = time.perf_counter() - t0
    print(f"   n={n}, d={d}, D={D}, m={m}")
    print(f"   Tamaño coreset : {len(idx)}  (esperado {n//2**m})")
    print(f"   Tiempo         : {dt:.3f}s")

    # ──────────────────────────────────────────────────────────────
    # 2. Modo joint kernel (supervisado)
    # ──────────────────────────────────────────────────────────────
    print(SEP)
    print("2. Modo supervisado (joint kernel, lifting tensors)")
    p, Dy = 3, 64
    y = np.sin(X[:, :p]) + 0.1 * rng.standard_normal((n, p))

    bw_y     = 0.5
    k_y      = make_gaussian_kernel(bw_y)
    phi_rt_y = make_rff(bw_y, Dy, p, rng=3)

    kt_j = KernelThinning(k_rt=k_rt, phi_rt=phi_rt,
                           k_star=k_star, phi_star=phi_star,
                           k_y=k_y, phi_rt_y=phi_rt_y, rng=42)

    m = 2
    t0    = time.perf_counter()
    idx_j = kt_j.thin(X, y=y, m=m)
    dt    = time.perf_counter() - t0
    print(f"   n={n}, d={d}, p={p}, D={D}, Dy={Dy}, m={m}")
    print(f"   Tamaño coreset : {len(idx_j)}  (esperado {n//2**m})")
    print(f"   Tiempo         : {dt:.3f}s")

    # ──────────────────────────────────────────────────────────────
    # 3. Kernel Halving
    # ──────────────────────────────────────────────────────────────
    print(SEP)
    print("3. Kernel Halving")
    t0      = time.perf_counter()
    sp, sm  = kt.kernel_halving(X)
    print(f"   S+={len(sp)}, S-={len(sm)}, "
          f"intersección={len(set(sp)&set(sm))}, "
          f"tiempo={time.perf_counter()-t0:.3f}s")
    assert len(set(sp) & set(sm)) == 0
    print("   ✓ S+ ∩ S- = ∅")

    # ──────────────────────────────────────────────────────────────
    # 4. Calidad MMD² aproximado
    # ──────────────────────────────────────────────────────────────
    print(SEP)
    print("4. Calidad del coreset  (MMD² vía RFF independiente)")
    phi_eval = make_rff(bw * math.sqrt(2), 1024, d, rng=99)
    Phi_all  = phi_eval(X)
    mu_all   = Phi_all.mean(0)

    def mmd2_approx(subset_idx):
        diff = mu_all - Phi_all[subset_idx].mean(0)
        return float(diff @ diff)

    rand_idx = rng.choice(n, len(idx), replace=False)
    mmd_kt   = mmd2_approx(idx)
    mmd_rnd  = mmd2_approx(rand_idx)
    print(f"   MMD²(KT,  X) = {mmd_kt:.2e}")
    print(f"   MMD²(rand,X) = {mmd_rnd:.2e}")
    print(f"   Mejora       = {100*(mmd_rnd-mmd_kt)/mmd_rnd:.1f}%")