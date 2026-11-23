"""Mouvement brownien fractionnaire sur la grille t_i = i/n, i = 1, ..., n.

Simulation exacte par plongement circulant (Davies–Harte) du bruit gaussien
fractionnaire (fGn), puis somme cumulée.
"""
import numpy as np


def autocovariance_fgn(k: np.ndarray, H: float) -> np.ndarray:
    """gamma(k) = Cov(G_0, G_k) pour le fGn de variance 1."""
    k = np.abs(k).astype(float)
    return 0.5 * ((k + 1) ** (2 * H) - 2 * k ** (2 * H) + np.abs(k - 1) ** (2 * H))


def simuler_fbm(n: int, H: float, rng: np.random.Generator) -> np.ndarray:
    """Renvoie x de taille n avec x[i-1] = B_H(i/n), Var B_H(t) = t^{2H}."""
    m = 2 * n
    gamma = autocovariance_fgn(np.arange(n + 1), H)
    ligne = np.concatenate([gamma, gamma[-2:0:-1]])       # première ligne de la matrice circulante (taille m)
    lam = np.fft.fft(ligne).real                          # valeurs propres
    if lam.min() < -1e-10:
        raise ValueError("Plongement circulant non défini positif")
    lam = np.clip(lam, 0.0, None)
    Z = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    W = np.fft.fft(np.sqrt(lam / m) * Z)
    fgn = W.real[:n]                                      # fGn : Var = 1, Cov = gamma
    return n ** (-H) * np.cumsum(fgn)                     # changement d'échelle : pas 1/n


def zeta_theorique(p: np.ndarray, H: float) -> np.ndarray:
    """zeta(p) = pH (monofractal). Pour les incréments et les coefficients,
    valable seulement pour p > -1 (moments négatifs gaussiens)."""
    return np.asarray(p, dtype=float) * H
