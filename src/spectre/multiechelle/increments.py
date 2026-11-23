"""Incréments d'ordre 1 aux échelles dyadiques.

Pour x[i-1] = f(i/n), n = 2^J : à l'échelle j (pas 2^{-j}, soit 2^{J-j} échantillons),
    Delta_{j,k} = f((k+1) 2^{-j}) - f(k 2^{-j}),   k = 1, ..., 2^j - 1.
Ordre 1 : ne voit que les régularités h < 1.
"""
import numpy as np

from .base import Multiechelle


def increments(x: np.ndarray) -> Multiechelle:
    n = x.size
    J = int(np.log2(n))
    assert 2**J == n, "n doit être une puissance de 2"
    valeurs, valide = {}, {}
    for j in range(1, J + 1):
        m = 2 ** (J - j)                  # pas en nombre d'échantillons
        points = x[m - 1 :: m]            # f(k 2^{-j}), k = 1, ..., 2^j
        valeurs[j] = np.diff(points)
        valide[j] = np.ones(valeurs[j].size, dtype=bool)
    return Multiechelle("increments", J, valeurs, valide)
