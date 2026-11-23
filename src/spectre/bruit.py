"""Modèle d'observation Y_i = f(i/n) + sigma * xi_i, xi_i i.i.d. N(0, 1).

sigma est le niveau de bruit par échantillon. Sur les coefficients d'ondelettes
L² de f, cela correspond à un bruit de niveau effectif sigma / sqrt(n).
"""
import numpy as np


def ajouter_bruit(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma == 0:
        return x.copy()
    return x + sigma * rng.standard_normal(x.size)
