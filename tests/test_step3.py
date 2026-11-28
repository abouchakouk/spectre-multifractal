import numpy as np
import pytest

from spectre.legendre import legendre, legendre_parametrique
from spectre.modeles.fbm import simuler_fbm
from spectre.multiechelle.base import Multiechelle
from spectre.multiechelle.leaders import leaders
from spectre.multiechelle.ondelettes import coefficients_ondelettes
from spectre.regression import estimer_zeta
from spectre.structure import log2_structure


def faux_coefficients(j0, j1, rng):
    """Coefficients aléatoires sur les échelles j0..j1, tous valides."""
    val = {j: rng.standard_normal(2**j) for j in range(j0, j1 + 1)}
    return Multiechelle("ondelettes", j1 + 1, val, {j: np.ones(2**j, bool) for j in val})


def leader_force_brute(c, j, k, j_max):
    """Définition directe : sup |c_{j',k'}| sur lambda' inclus dans 3 lambda_{j,k}, j <= j' <= j_max."""
    s = 0.0
    for jp in range(j, j_max + 1):
        kp = np.arange(2**jp)
        parent = kp // 2 ** (jp - j)          # ancêtre de (jp, kp) à l'échelle j
        dedans = np.abs(parent - k) <= 1
        s = max(s, np.abs(c.valeurs[jp][dedans]).max())
    return s


# ---------- leaders ----------
@pytest.mark.parametrize("j_max", [7, 5])
def test_leaders_force_brute(j_max):
    c = faux_coefficients(2, 7, np.random.default_rng(0))
    L = leaders(c, j_max)
    assert L.echelles == list(range(2, j_max + 1))
    for j in L.echelles:
        for k in range(1, 2**j - 1):          # extrémités exclues
            assert L.valeurs[j][k] == pytest.approx(leader_force_brute(c, j, k, j_max))
        assert not L.valide[j][0] and not L.valide[j][-1]


def test_leaders_proprietes():
    """L >= |c| et hiérarchie L_{j+1, 2k}, L_{j+1, 2k+1} <= L_{j,k}."""
    rng = np.random.default_rng(1)
    c = coefficients_ondelettes(simuler_fbm(2**12, 0.5, rng))
    L = leaders(c)
    for j in L.echelles:
        ok = L.valide[j] & c.valide[j]
        assert np.all(L.valeurs[j][ok] >= np.abs(c.valeurs[j][ok]))
        if j + 1 in L.echelles:
            for fils in (L.valeurs[j + 1][0::2], L.valeurs[j + 1][1::2]):
                assert np.all(fils <= L.valeurs[j] + 1e-15)


# ---------- fonction de structure ----------
def test_structure_constante_et_zeros():
    """v = a partout sauf des zéros : S(j,p) = 2^{-j} * (#non nuls) * a^p."""
    j, a = 6, 0.3
    v = np.full(2**j, a)
    v[::4] = 0.0
    m = Multiechelle("test", 8, {j: v}, {j: np.ones(2**j, bool)})
    p = np.array([-3.0, -1.0, 0.0, 2.0])
    js, log2S, nb = log2_structure(m, p)
    assert nb[0] == 48
    assert np.allclose(log2S[0], -j + np.log2(48) + p * np.log2(a))


# ---------- régression ----------
def test_regression_exacte():
    p = np.linspace(-3, 3, 13)
    zeta = 0.4 * p - 0.05 * p**2
    js = np.arange(3, 12)
    log2S = 2.0 - np.outer(js, zeta)
    assert np.allclose(estimer_zeta(js, log2S, 4, 10), zeta)


# ---------- Legendre ----------
def test_legendre_parabole():
    """zeta(p) = a p - b p^2 / 2  =>  D(h) = 1 - (h - a)^2 / (2b), minimiseur p* = (a - h)/b."""
    a, b = 0.5, 0.04
    p = np.linspace(-10, 10, 2001)
    h = np.linspace(a - 9 * b, a + 9 * b, 50)        # p* dans [-9, 9]
    D = legendre(p, a * p - b * p**2 / 2, h)
    assert np.allclose(D, 1 - (h - a) ** 2 / (2 * b), atol=1e-5)
    hp, Dp = legendre_parametrique(p, a * p - b * p**2 / 2)
    assert np.allclose(Dp[1:-1], 1 - (hp[1:-1] - a) ** 2 / (2 * b), atol=1e-8)


def test_legendre_lineaire_tente():
    """zeta(p) = pH sur [-P, P] : D(h) = 1 - P |h - H| (tente due à la grille finie)."""
    H, P = 0.6, 4.0
    p = np.linspace(-P, P, 81)
    h = np.linspace(0, 1.2, 25)
    assert np.allclose(legendre(p, H * p, h), 1 - P * np.abs(h - H))


# ---------- chaîne complète sur le fBm (contrôle grossier) ----------
def test_chaine_fbm_leaders():
    rng = np.random.default_rng(2)
    H = 0.6
    L = leaders(coefficients_ondelettes(simuler_fbm(2**14, H, rng)))
    p = np.array([-2.0, -1.0, 1.0, 2.0])
    js, log2S, _ = log2_structure(L, p)
    zeta = estimer_zeta(js, log2S, 5, 12) - p / 2        # L2 : zeta = zeta_d - p/2
    assert np.allclose(zeta, p * H, atol=0.15)
