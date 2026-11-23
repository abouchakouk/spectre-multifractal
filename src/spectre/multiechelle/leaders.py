"""Leaders d'ondelettes, avec coupure possible aux échelles fines.

Pour lambda = (j, k) et j <= j_max :
    L_{j,k} = sup { |c_{j',k'}| : lambda' = (j',k') inclus dans 3 lambda, j <= j' <= j_max },
avec 3 lambda = lambda_{j,k-1} u lambda_{j,k} u lambda_{j,k+1}.
Calcul récursif (du fin vers le grossier) :
    l_{j,k} = max(|c_{j,k}|, l_{j+1,2k}, l_{j+1,2k+1}),   L_{j,k} = max(l_{j,k-1}, l_{j,k}, l_{j,k+1}).
Un leader est valide si tous les coefficients de son sup sont valides et si ses deux
voisins existent (pas de leader aux extrémités k = 0 et k = 2^j - 1).
"""
import numpy as np

from .base import Multiechelle


def leaders(c: Multiechelle, j_max: int | None = None) -> Multiechelle:
    assert c.nom == "ondelettes", "les leaders se calculent à partir des coefficients d'ondelettes"
    js = c.echelles
    if j_max is None:
        j_max = js[-1]
    assert js[0] <= j_max <= js[-1], "j_max hors des échelles disponibles"

    # sup local l_{j,k} sur les descendants de lambda_{j,k}, et validité associée
    l = np.where(c.valide[j_max], np.abs(c.valeurs[j_max]), 0.0)
    v = c.valide[j_max].copy()
    valeurs, valide = {}, {}
    for j in range(j_max, js[0] - 1, -1):
        if j < j_max:
            l = np.maximum.reduce([np.where(c.valide[j], np.abs(c.valeurs[j]), 0.0), l[0::2], l[1::2]])
            v = c.valide[j] & v[0::2] & v[1::2]
        # sup sur les trois voisins k-1, k, k+1 (complété par 0 / non valide aux extrémités)
        lp = np.concatenate([[0.0], l, [0.0]])
        vp = np.concatenate([[False], v, [False]])
        valeurs[j] = np.maximum.reduce([lp[:-2], lp[1:-1], lp[2:]])
        valide[j] = vp[:-2] & vp[1:-1] & vp[2:]
    return Multiechelle("leaders", c.J, valeurs, valide)
