import numpy as np
import pytest

from spectre.bruit import ajouter_bruit
from spectre.modeles.fbm import simuler_fbm
from spectre.multiechelle.increments import increments
from spectre.multiechelle.ondelettes import coefficients_ondelettes


def pente(j, y):
    return np.polyfit(j, y, 1)[0]


# ---------- fBm ----------
@pytest.mark.parametrize("H", [0.3, 0.7])
def test_fbm_variance(H):
    """Var B_H(1) = 1 et Var(incrément fin) = n^{-2H}."""
    rng = np.random.default_rng(0)
    n, R = 2**10, 400
    X = np.array([simuler_fbm(n, H, rng) for _ in range(R)])
    assert abs(X[:, -1].var() - 1) < 0.2
    assert abs(np.diff(X, axis=1).var() * n ** (2 * H) - 1) < 0.05


# ---------- incréments ----------
def test_increments_fbm_pente():
    """E Delta_{j,k}^2 = 2^{-2jH} exactement : pente -2H en log2."""
    rng = np.random.default_rng(1)
    H, n = 0.6, 2**12
    inc = [increments(simuler_fbm(n, H, rng)) for _ in range(30)]
    js = np.arange(4, 12)
    y = [np.log2(np.mean([np.mean(m.valeurs[j] ** 2) for m in inc])) for j in js]
    assert abs(pente(js, y) + 2 * H) < 0.05


def test_increments_bruit():
    """Bruit pur : Var Delta = 2 sigma^2 à toutes les échelles (pas de moyennage)."""
    rng = np.random.default_rng(2)
    m = increments(ajouter_bruit(np.zeros(2**14), 0.5, rng))
    for j in range(6, 15):
        assert abs(m.valeurs[j].var() / (2 * 0.5**2) - 1) < 0.3


# ---------- ondelettes ----------
def test_ondelettes_tailles():
    m = coefficients_ondelettes(np.zeros(2**12))
    for j in m.echelles:
        assert m.valeurs[j].size == 2**j


def test_ondelettes_polynome():
    """db3 a 3 moments nuls : un polynôme de degré 2 donne 0 hors bord."""
    t = np.arange(1, 2**12 + 1) / 2**12
    m = coefficients_ondelettes(1 + 2 * t - 3 * t**2, "db3")
    for j in m.echelles:
        assert np.all(np.abs(m.valeurs_valides(j)) < 1e-9)


def test_ondelettes_bruit():
    """Bruit pur : Var c_{j,k} = sigma^2 2^j / n (normalisation L¹)."""
    rng = np.random.default_rng(3)
    n, sigma = 2**12, 0.5
    ms = [coefficients_ondelettes(ajouter_bruit(np.zeros(n), sigma, rng)) for _ in range(20)]
    for j in range(6, 12):  # assez de coefficients pour une variance empirique fiable
        v = np.mean([np.mean(m.valeurs_valides(j) ** 2) for m in ms])
        assert abs(v / (sigma**2 * 2**j / n) - 1) < 0.25


def test_ondelettes_fbm_pente():
    """fBm : E c_{j,k}^2 ~ 2^{-2jH} en normalisation L¹."""
    rng = np.random.default_rng(4)
    H, n = 0.6, 2**14
    ms = [coefficients_ondelettes(simuler_fbm(n, H, rng)) for _ in range(20)]
    js = np.arange(5, 13)
    y = [np.log2(np.mean([np.mean(m.valeurs_valides(j) ** 2) for m in ms])) for j in js]
    assert abs(pente(js, y) + 2 * H) < 0.08
