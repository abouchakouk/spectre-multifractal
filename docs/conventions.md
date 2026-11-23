# Conventions (contrat du dépôt)

- Données : x[i-1] = f(i/n), i = 1..n, n = 2^J. Bruit : Y = f + sigma * xi, xi ~ N(0,1) i.i.d.
  Niveau effectif sur les coefficients L² : sigma / sqrt(n).
- Échelles : convention maths. j = 0 grossier, j grand fin ; échelle j <-> longueur 2^{-j}.
  Octaves (PyWavelets) : j_oct = J - j, utilisées uniquement dans ondelettes.py.
- Coefficients : normalisation L¹, c_{j,k} = 2^j ∫ f(t) psi(2^j t - k) dt = 2^{-j_oct/2} d^pywt.
- Fonction de structure : S(j,p) = 2^{-j} * somme sur les termes non nuls ; S ~ 2^{-j zeta(p)}.
- Spectre : D(h) = inf_p (1 + p h - zeta(p)).
- Bord : coefficients dont le support traverse le recollement périodique exclus (masque `valide`).
