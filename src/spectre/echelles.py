"""Échelle la plus fine utile en présence de bruit.

Heuristique (étape 1) : à l'échelle j, les coefficients L² d'une fonction C^eps sont
au plus d'ordre 2^{-j(eps + 1/2)}, et le bruit d'ordre sigma / sqrt(n). Au-delà de
    2^{-j(eps + 1/2)} = sigma / sqrt(n),   soit   j = (J/2 - log2 sigma) / (eps + 1/2),
tous les coefficients sont noyés (constantes ignorées).
"""
import numpy as np


def j_max_bruit(sigma: float, n: int, eps: float, j_plus_fin: int) -> int:
    if sigma == 0:
        return j_plus_fin
    J = np.log2(n)
    j = int(np.floor((J / 2 - np.log2(sigma)) / (eps + 0.5)))
    return min(j, j_plus_fin)
