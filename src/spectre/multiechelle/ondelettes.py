"""Coefficients d'ondelettes en convention maths (j fin = grand).

PyWavelets (transformée orthonormale sur les échantillons, mode périodique) renvoie
les détails par octave j_oct = 1 (fin), ..., niveau (grossier), avec j = J - j_oct et
d^pywt_{j_oct,k} ≈ sqrt(n) <f, psi_{j,k}>   (psi_{j,k} = 2^{j/2} psi(2^j . - k)).

Deux normalisations :
  "L2" (défaut depuis le 26 nov. 2026) : d_{j,k} = d^pywt / sqrt(n) ≈ <f, psi_{j,k}>.
        Bruit : sigma / sqrt(n) à toutes les échelles. fBm : d ~ 2^{-j(H + 1/2)}.
        S_d(j,p) ~ 2^{-j zeta_d(p)} avec zeta_d(p) = zeta(p) + p/2 (coefficients ET leaders).
  "L1" (historique, E1–E6 du 25 nov.) : c_{j,k} = 2^{j/2} d_{j,k} = 2^{-j_oct/2} d^pywt.

Attention : pour les coefficients, passer de L1 à L2 ne fait que décaler les pentes de p/2.
Pour les leaders, ce n'est pas un simple changement d'unité : le sup sur les descendants
pondère l'échelle j' par 2^{-j'/2} en L2, donc L^d_{j,k} != 2^{-j/2} L^c_{j,k}.

Effets de bord : la périodisation recolle f(1) et f(0). Les coefficients dont le
support traverse ce recollement sont marqués non valides (masque calculé sur une
rampe, que les moments nuls annulent partout ailleurs).
"""
import numpy as np
import pywt

from .base import Multiechelle


def _details_pywt(x: np.ndarray, ondelette: pywt.Wavelet, niveau: int) -> list[np.ndarray]:
    """Liste [cD_niveau, ..., cD_1] (du plus grossier au plus fin)."""
    return pywt.wavedec(x, ondelette, mode="periodization", level=niveau)[1:]


def coefficients_ondelettes(x: np.ndarray, ondelette: str = "db3",
                            normalisation: str = "L2") -> Multiechelle:
    assert normalisation in ("L1", "L2"), "normalisation : 'L1' ou 'L2'"
    n = x.size
    J = int(np.log2(n))
    assert 2**J == n, "n doit être une puissance de 2"
    w = pywt.Wavelet(ondelette)
    niveau = pywt.dwt_max_level(n, w.dec_len)

    details = _details_pywt(x, w, niveau)
    rampe = _details_pywt(np.arange(n) / n, w, niveau)   # polynôme de degré 1

    valeurs, valide = {}, {}
    for i, (d, r) in enumerate(zip(details, rampe)):
        j_oct = niveau - i
        j = J - j_oct
        if normalisation == "L2":
            valeurs[j] = d / np.sqrt(n)
        else:
            valeurs[j] = 2.0 ** (-j_oct / 2) * d
        valide[j] = np.abs(r) < 1e-10
    return Multiechelle("ondelettes", J, valeurs, valide)
