# Questions ouvertes (23 nov. 2026)

1. Initialisation de la DWT : les échelles j = 13, 14 (n = 2^15) cassent la loi d'échelle
   (pente locale de log2 E c^2 : 0,58 puis -0,11 en unités de -2H, à H = 0,3).
   Les leaders héritent de ce défaut à toutes les échelles (sup jusqu'à j_max).
   Test : j_max = 12 divise le biais p > 0 des leaders par 4 à 7 (H = 0,3).
   -> Fixer une coupure de base j_max <= J - 3 même à sigma = 0 ? Référence à vérifier
      (Veitch, Taqqu, Abry, initialisation de la MRA pour séries discrètes, ~2000).
2. Valeurs extrêmes à j_max : le leader y est le max de 3 coefficients
   -> E L^p infini pour p <= -3 (heuristique). Règle : régresser jusqu'à j2 <= j_max - 1 (ou -2).
   E3 actuel régresse jusqu'à j_max : à corriger.
3. Troncature de l'arbre : leaders proches de j_max plus petits -> biais modéré pour p < 0.
4. Coupure due au bruit : la règle 2^{-j(eps+1/2)} = sigma/sqrt(n) est trop optimiste
   (constantes ignorées, facteur sqrt(log) du max du bruit). « Petite manip » de Marc ?
5. Exprimer sigma relativement au signal (sigma = 0,01 n'est pas « petit » pour H = 0,7).
