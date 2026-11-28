import numpy as np

from spectre.legendre import legendre
from spectre.modeles.wcascade import simuler_wcascade, spectre_theorique, zeta_theorique
from spectre.multiechelle.ondelettes import coefficients_ondelettes
from spectre.regression import estimer_zeta
from spectre.structure import log2_structure

m, b = 0.5, 0.025


def test_wcascade_coefficients_exacts():
    """La transformée directe redonne exactement les coefficients fabriqués (pas d'initialisation)."""
    x, c = simuler_wcascade(2**12, m, b, np.random.default_rng(0), avec_coefficients=True)
    mc = coefficients_ondelettes(x, normalisation="L1")   # la cascade fabrique des c (L1)
    assert mc.echelles == sorted(c)
    for j in c:
        assert np.allclose(mc.valeurs[j], c[j], atol=1e-10)


def test_wcascade_zeta_coefficients():
    """Coefficients, p > 0 : zeta_hat proche de m p - (b/2) p^2."""
    rng = np.random.default_rng(1)
    P = np.array([0.5, 1.0, 2.0])
    z = []
    for _ in range(30):
        mc = coefficients_ondelettes(simuler_wcascade(2**15, m, b, rng))
        js, l, _ = log2_structure(mc, P, range(3, 15))
        z.append(estimer_zeta(js, l, 6, 13) - P / 2)       # L2 : zeta = zeta_d - p/2
    assert np.allclose(np.median(z, axis=0), zeta_theorique(P, m, b), atol=0.05)


def test_wcascade_theorie():
    """D(m) = 1, D(m ± sqrt(2b)) = 0, et D = Legendre de zeta."""
    assert spectre_theorique(m, m, b) == 1
    assert np.allclose(spectre_theorique(m + np.array([-1, 1]) * np.sqrt(2 * b), m, b), 0)
    P = np.linspace(-10, 10, 2001)
    h = np.linspace(m - 0.2, m + 0.2, 21)                 # minimiseur p* = (m - h)/b dans [-8, 8]
    assert np.allclose(legendre(P, zeta_theorique(P, m, b), h), spectre_theorique(h, m, b), atol=1e-4)
