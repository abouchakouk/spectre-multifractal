"""Cascade d'ondelettes log-normale (« W-cascade », Arneodo–Bacry–Muzy).

Coefficients L¹ (convention maths, j fin = grand) :
    c_{j,k} = eps_{j,k} * prod_{l=1}^{j} W_l(ancêtre de (j,k) à l'échelle l),
    W = 2^{-X}, X ~ N(m, s^2) i.i.d. sur les noeuds, eps ~ N(0,1) i.i.d.
Le signal est obtenu par transformée inverse (PyWavelets, mode périodique) ; la
transformée directe redonne donc exactement les c_{j,k} (aucun défaut d'initialisation).

Théorie (moments) : E 2^{-j} sum_k |c_{j,k}|^p = E|eps|^p (E W^p)^j et
log2 E W^p = -p m + (ln2 / 2) s^2 p^2, d'où, avec b = s^2 ln 2 :
    zeta(p) = m p - (b/2) p^2,     D(h) = 1 - (h - m)^2 / (2b).
"""
import numpy as np
import pywt


def simuler_wcascade(n: int, m: float, b: float, rng: np.random.Generator,
                     ondelette: str = "db3", avec_coefficients: bool = False):
    """Renvoie le signal x (taille n) ; avec avec_coefficients=True, renvoie (x, {j: c_j})."""
    J = int(np.log2(n))
    assert 2**J == n, "n doit être une puissance de 2"
    w = pywt.Wavelet(ondelette)
    niveau = pywt.dwt_max_level(n, w.dec_len)
    s = np.sqrt(b / np.log(2))
    M = np.ones(1)                                   # multiplicateurs cumulés à l'échelle 0
    details, coeffs = {}, {}
    for j in range(1, J):
        W = 2.0 ** (-rng.normal(m, s, 2**j))
        M = np.repeat(M, 2) * W                      # chaque fils hérite du produit de ses ancêtres
        if j >= J - niveau:
            c = rng.standard_normal(2**j) * M        # coefficients L¹
            coeffs[j] = c
            details[j] = 2.0 ** ((J - j) / 2) * c    # vers PyWavelets : d = 2^{j_oct/2} c
    liste = [np.zeros(2 ** (J - niveau))] + [details[j] for j in range(J - niveau, J)]
    x = pywt.waverec(liste, w, mode="periodization")
    return (x, coeffs) if avec_coefficients else x


def zeta_theorique(p, m: float, b: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return m * p - 0.5 * b * p**2


def spectre_theorique(h, m: float, b: float) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    return 1 - (h - m) ** 2 / (2 * b)
