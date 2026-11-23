"""Structure commune à toutes les quantités multi-échelles.

Convention (voir docs/conventions.md) : j = 0 est l'échelle la plus grossière,
j grand est fin ; l'échelle j correspond à la longueur 2^{-j} sur [0, 1].
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Multiechelle:
    nom: str                       # "increments", "ondelettes" ou "leaders"
    J: int                         # n = 2^J données
    valeurs: dict[int, np.ndarray]  # j -> valeurs à l'échelle j (indexées par k)
    valide: dict[int, np.ndarray]   # j -> masque booléen (False = effet de bord)

    @property
    def echelles(self) -> list[int]:
        return sorted(self.valeurs)

    def valeurs_valides(self, j: int) -> np.ndarray:
        return self.valeurs[j][self.valide[j]]
