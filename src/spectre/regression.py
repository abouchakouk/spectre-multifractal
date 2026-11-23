"""Estimation de zeta(p) par régression sur une plage d'échelles [j1, j2] :
    log2 S(j, p) ~ C(p) - j zeta(p)   =>   zeta_hat(p) = - pente (moindres carrés ordinaires).
"""
import numpy as np


def estimer_zeta(js: np.ndarray, log2S: np.ndarray, j1: int, j2: int) -> np.ndarray:
    sel = (js >= j1) & (js <= j2)
    assert sel.sum() >= 2, "il faut au moins deux échelles dans [j1, j2]"
    x, Y = js[sel].astype(float), log2S[sel]
    assert np.all(np.isfinite(Y)), "log2 S non fini sur la plage (échelle sans terme ?)"
    xc = x - x.mean()
    pentes = xc @ (Y - Y.mean(axis=0)) / (xc @ xc)
    return -pentes
