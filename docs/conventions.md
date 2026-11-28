# Conventions (contrat du dépôt)

- Données : x[i-1] = f(i/n), i = 1..n, n = 2^J. Bruit : Y = f + sigma * xi, xi ~ N(0,1) i.i.d.
- Échelles : convention maths. j = 0 grossier, j grand fin ; échelle j <-> longueur 2^{-j}.
  Octaves (PyWavelets) : j_oct = J - j, utilisées uniquement dans ondelettes.py.
- Coefficients (depuis le 26 nov. 2026) : normalisation L², d_{j,k} = d^pywt / sqrt(n) ≈ <f, psi_{j,k}>.
  Bruit : sigma / sqrt(n) à toutes les échelles. fBm : d_{j,k} ~ 2^{-j(H + 1/2)}.
  Historique (E1–E6 du 25 nov.) : normalisation L¹, c_{j,k} = 2^{j/2} d_{j,k} (option normalisation="L1").
- Fonction de structure : S(j,p) = 2^{-j} * somme sur les termes non nuls ; S ~ 2^{-j zeta_d(p)}.
- Lien avec zeta : zeta(p) = zeta_d(p) - p/2 (on corrige AVANT Legendre et avant toute comparaison à pH).
- Spectre : D(h) = inf_p (1 + p h - zeta(p)) = inf_p (1 + p (h + 1/2) - zeta_d(p)).
- Leaders : calculés sur les d_{j,k} ; ce ne sont PAS 2^{-j/2} fois les leaders L¹ (le sup pondère les échelles).
- Bord : coefficients dont le support traverse le recollement périodique exclus (masque `valide`).
