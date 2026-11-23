"""Fonction de structure, sommée sur les termes valides et non nuls :
    S(j, p) = 2^{-j} * sum_{k : valide, |v_{j,k}| > 0} |v_{j,k}|^p.
Calcul en log2 (logsumexp) pour éviter les dépassements quand p < 0 et |v| petit.
"""
import numpy as np
from scipy.special import logsumexp

from .multiechelle.base import Multiechelle


def log2_structure(m: Multiechelle, p: np.ndarray, echelles: list[int] | None = None):
    """Renvoie (js, log2S, nb) : log2S[a, b] = log2 S(js[a], p[b]) ; nb[a] = nombre de termes."""
    p = np.asarray(p, dtype=float)
    js = m.echelles if echelles is None else list(echelles)
    log2S = np.full((len(js), p.size), np.nan)
    nb = np.zeros(len(js), dtype=int)
    for a, j in enumerate(js):
        v = np.abs(m.valeurs_valides(j))
        v = v[v > 0]
        nb[a] = v.size
        if v.size == 0:
            continue
        lnv = np.log(v)
        # ln sum_k exp(p ln|v_k|), pour chaque p
        log2S[a] = logsumexp(np.outer(p, lnv), axis=1) / np.log(2) - j
    return np.array(js), log2S, nb
