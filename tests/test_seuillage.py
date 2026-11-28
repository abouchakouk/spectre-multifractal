"""Seuillage dur et plage d'échelles déterminée par les données."""
import numpy as np

from spectre.bruit import ajouter_bruit
from spectre.multiechelle.base import Multiechelle
from spectre.multiechelle.leaders import leaders
from spectre.multiechelle.ondelettes import coefficients_ondelettes
from spectre.seuillage import niveau_seuil, plage_donnees, seuiller


def test_niveaux():
    n, sigma = 2**15, 0.1
    assert np.isclose(niveau_seuil(sigma, n, "universel"), sigma * np.sqrt(2 * np.log(n)) / np.sqrt(n))
    assert np.isclose(niveau_seuil(sigma, n, "3sigma"), 3 * sigma / np.sqrt(n))
    assert niveau_seuil(0.0, n, "universel") == 0.0


def test_seuillage_dur():
    v = np.array([-3.0, -1.0, 0.5, 2.0, 1.0])
    d = Multiechelle("ondelettes", 3, {2: v}, {2: np.ones(5, bool)})
    assert np.array_equal(seuiller(d, 1.0).valeurs[2], [-3.0, 0.0, 0.0, 2.0, 0.0])   # |d| = lambda : tué


def test_bruit_pur_tue():
    """Bruit seul, seuil universel : (presque) tous les coefficients sont tués."""
    n, sigma = 2**15, 0.5
    rng = np.random.default_rng(0)
    tues = []
    for _ in range(20):
        d = coefficients_ondelettes(ajouter_bruit(np.zeros(n), sigma, rng))
        s = seuiller(d, niveau_seuil(sigma, n, "universel"))
        tues.append(sum(np.count_nonzero(v) for v in s.valeurs.values()))
    assert np.mean(tues) < 1.0          # en moyenne moins d'un coefficient survivant sur n


def test_plage_donnees():
    """Leaders non nuls jusqu'à j = 9, nuls au-delà : plage [6, 9] ; tout nul : None."""
    J, val, ok = 14, {}, {}
    for j in range(3, 15):
        val[j] = np.ones(2**j) if j <= 9 else np.zeros(2**j)
        ok[j] = np.ones(2**j, bool)
    L = Multiechelle("leaders", J, val, ok)
    assert plage_donnees(L) == (6, 9)
    val7 = {j: (np.ones(2**j) if j <= 7 else np.zeros(2**j)) for j in range(3, 15)}
    assert plage_donnees(Multiechelle("leaders", J, val7, ok)) == (4, 7)
    val4 = {j: (np.ones(2**j) if j <= 4 else np.zeros(2**j)) for j in range(3, 15)}
    assert plage_donnees(Multiechelle("leaders", J, val4, ok)) is None
    L0 = Multiechelle("leaders", J, {j: np.zeros(2**j) for j in range(3, 15)}, ok)
    assert plage_donnees(L0) is None


def test_sans_bruit_inchange():
    """sigma = 0 : seuil nul, leaders identiques, plage plafonnée à 13."""
    rng = np.random.default_rng(1)
    d = coefficients_ondelettes(rng.standard_normal(2**12).cumsum())
    L, Ls = leaders(d), leaders(seuiller(d, 0.0))
    for j in L.echelles:
        assert np.array_equal(L.valeurs[j], Ls.valeurs[j])
    assert plage_donnees(Ls, j2_plafond=11) == (6, 11)
