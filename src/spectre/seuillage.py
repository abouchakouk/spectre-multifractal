"""Seuillage dur des coefficients d'ondelettes (normalisation L2) et plage d'échelles associée.

Sur les d_{j,k} (normalisation L2), le bruit a le niveau sigma / sqrt(n) à toutes les échelles :
un seul seuil lambda sert pour toutes les échelles.
  Seuil universel (Donoho–Johnstone) : lambda_U = sigma * sqrt(2 ln n) / sqrt(n).
  Seuil plus doux                     : lambda_3 = 3 * sigma / sqrt(n).
Seuillage dur : d~_{j,k} = d_{j,k} * 1{|d_{j,k}| > lambda}.

Après seuillage, des leaders peuvent être nuls (tout le cône est tué). Pour p > 0, un leader nul
contribue 0^p = 0 à S(j,p) = 2^{-j} sum_k L^p : log2_structure le traite correctement.
Pour p < 0, L^p = +inf : NE PAS utiliser ce module pour p < 0 tant que la règle n'est pas fixée.
"""
import numpy as np

from .multiechelle.base import Multiechelle

REGLES_SEUIL = ("universel", "3sigma")


def niveau_seuil(sigma: float, n: int, regle: str) -> float:
    """lambda pour des coefficients L2 : 'universel' -> sigma sqrt(2 ln n)/sqrt(n) ; '3sigma' -> 3 sigma/sqrt(n)."""
    assert regle in REGLES_SEUIL, f"règle inconnue : {regle}"
    facteur = np.sqrt(2 * np.log(n)) if regle == "universel" else 3.0
    return facteur * sigma / np.sqrt(n)


def seuiller(d: Multiechelle, lam: float) -> Multiechelle:
    """Seuillage dur : d * 1{|d| > lam}, échelle par échelle (masque de validité inchangé)."""
    assert d.nom == "ondelettes", "on seuille des coefficients d'ondelettes"
    valeurs = {j: np.where(np.abs(v) > lam, v, 0.0) for j, v in d.valeurs.items()}
    return Multiechelle(d.nom, d.J, valeurs, d.valide)


def fraction_non_nuls(L: Multiechelle, js) -> np.ndarray:
    """q_j = proportion de leaders valides non nuls à l'échelle j."""
    return np.array([np.mean(L.valeurs_valides(j) > 0) for j in js])


def plage_donnees(L: Multiechelle, j1_defaut: int = 6, j2_plafond: int = 13, j_min: int = 3,
                  largeur_min: int = 3, tolerance: float = 1.0):
    """Plage de régression déterminée par les données après seuillage.

    j2 = plus grande échelle j <= j2_plafond telle que q_i >= tolerance pour tout i dans [j_min, j]
         (tolerance = 1 : tous les leaders valides sont non nuls) ;
    j1 = min(j1_defaut, j2 - largeur_min), borné inférieurement par j_min.
    Renvoie (j1, j2), ou None si j2 - j1 < largeur_min - 1 (moins de `largeur_min` échelles) :
    configuration déclarée non estimable.
    """
    j2 = None
    for j in range(j_min, j2_plafond + 1):
        if j not in L.valeurs or np.mean(L.valeurs_valides(j) > 0) < tolerance:
            break
        j2 = j
    if j2 is None:
        return None
    j1 = max(j_min, min(j1_defaut, j2 - largeur_min))
    if j2 - j1 + 1 < largeur_min:
        return None
    return j1, j2
