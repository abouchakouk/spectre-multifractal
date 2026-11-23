"""Transformée de Legendre discrète :  D(h) = min_{p dans la grille} (1 + p h - zeta(p)).

Sur une grille finie [p_min, p_max], la formule donne l'enveloppe inférieure de droites
de pentes p : hors de [zeta'(p_max), zeta'(p_min)], D décroît avec la pente p_max ou p_min.
"""
import numpy as np


def legendre(p: np.ndarray, zeta: np.ndarray, h: np.ndarray) -> np.ndarray:
    p, zeta, h = (np.asarray(a, dtype=float) for a in (p, zeta, h))
    return np.min(1 + np.outer(h, p) - zeta[None, :], axis=1)


def legendre_parametrique(p: np.ndarray, zeta: np.ndarray):
    """Forme paramétrique : h(p) = zeta'(p), D(p) = 1 + p zeta'(p) - zeta(p)
    (dérivée par différences finies, grille non uniforme permise)."""
    p, zeta = np.asarray(p, dtype=float), np.asarray(zeta, dtype=float)
    dz = np.gradient(zeta, p)
    return dz, 1 + p * dz - zeta
