"""Coefficients d'ondelettes en convention maths (j fin = grand), normalisation L¹.

PyWavelets (transformée orthonormale sur les échantillons, mode périodique) renvoie
les détails par octave j_oct = 1 (fin), ..., niveau (grossier). Conversion :
    j = J - j_oct,      c_{j,k} = 2^{-j_oct / 2} * d^{pywt}_{j_oct,k}.
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


def coefficients_ondelettes(x: np.ndarray, ondelette: str = "db3") -> Multiechelle:
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
        valeurs[j] = 2.0 ** (-j_oct / 2) * d
        valide[j] = np.abs(r) < 1e-10
    return Multiechelle("ondelettes", J, valeurs, valide)
