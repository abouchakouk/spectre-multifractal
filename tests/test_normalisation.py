"""Normalisation L2 des coefficients (décision du 25 nov. 2026)."""
import numpy as np

from spectre.bruit import ajouter_bruit
from spectre.modeles.fbm import simuler_fbm
from spectre.multiechelle.ondelettes import coefficients_ondelettes


def test_lien_L1_L2():
    """c_{j,k} = 2^{j/2} d_{j,k}, échelle par échelle."""
    x = simuler_fbm(2**12, 0.5, np.random.default_rng(0))
    d = coefficients_ondelettes(x, normalisation="L2")
    c = coefficients_ondelettes(x, normalisation="L1")
    for j in d.echelles:
        assert np.allclose(c.valeurs[j], 2.0 ** (j / 2) * d.valeurs[j])


def test_bruit_constant_en_j():
    """Bruit blanc seul : écart-type des d_{j,k} égal à sigma / sqrt(n) à toutes les échelles."""
    n, sigma = 2**15, 0.7
    y = ajouter_bruit(np.zeros(n), sigma, np.random.default_rng(1))
    d = coefficients_ondelettes(y)
    for j in d.echelles:
        if 2**j >= 256:                                  # assez de coefficients pour estimer
            assert abs(d.valeurs[j].std() / (sigma / np.sqrt(n)) - 1) < 0.15
