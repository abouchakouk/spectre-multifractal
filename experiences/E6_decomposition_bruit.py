"""E6 — Décomposition signal / bruit des fonctions de structure des leaders.

But : expliquer les courbes « sans coupure » et « avec coupure » d'E2–E3 (et E5) en traçant,
sur les mêmes réalisations, log2 S(j, p) des leaders de
    (1) X seul (signal),  (2) sigma * xi seul (bruit),  (3) Y = X + sigma * xi (observation).
Lemme (sup 1-lipschitzien) : |L^X - L^bruit| <= L^Y <= L^X + L^bruit, donc (3) suit le max de (1) et (2).

Courbes théoriques :
  bruit  : log2 S ~ p log2( sigma 2^{-(J - j_max)/2} sqrt(2 ln(3 * 2^{j_max - j})) )
           (écart-type du bruit à l'échelle j_max en normalisation L¹, max de 3 * 2^{j_max - j} gaussiennes)
  signal : droite de pente -zeta(p), recalée sur la médiane du signal seul en j = 8
  croisement prédit j_x : intersection des deux.
Coupures : sans (j_max = J - 1 = 14) ; avec, règle d'E3 : j_max = floor((J/2 - log2 sigma)/(eps + 1/2)).

Usage :  python experiences/E6_decomposition_bruit.py [--refaire]
Sorties : resultats/E6_*.npz, figures/E6_<modèle>_<coupure>.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spectre.echelles import j_max_bruit
from spectre.modeles.fbm import simuler_fbm
from spectre.modeles.wcascade import simuler_wcascade, zeta_theorique
from spectre.multiechelle.leaders import leaders
from spectre.multiechelle.ondelettes import coefficients_ondelettes
from spectre.structure import log2_structure

# ---------------- paramètres ----------------
n = 2**15
J = 15
R = 50
ONDELETTE = "db3"
SIGMAS = (1e-3, 1e-2, 1e-1, 1.0)
P = np.array([2.0, -2.0])
ECHELLES = list(range(3, 15))
J_FIN = 14                                       # échelle la plus fine disponible (J - 1)
J_REF = 8                                        # échelle de recalage de la droite du signal
GRAINE = 20261124
M_C, B_C = 0.5, 0.025                            # W-cascade
MODELES = {
    "fBm_H0.3": dict(simuler=lambda rng: simuler_fbm(n, 0.3, rng), zeta=lambda p: 0.3 * p, eps=0.3,
                     titre=r"fBm $H=0{,}3$"),
    "fBm_H0.7": dict(simuler=lambda rng: simuler_fbm(n, 0.7, rng), zeta=lambda p: 0.7 * p, eps=0.7,
                     titre=r"fBm $H=0{,}7$"),
    "cascade": dict(simuler=lambda rng: simuler_wcascade(n, M_C, B_C, rng, ONDELETTE),
                    zeta=lambda p: zeta_theorique(p, M_C, B_C), eps=M_C - np.sqrt(2 * B_C),
                    titre=rf"W-cascade $m={M_C}$, $b={B_C}$"),
}
COMPOSANTES = ("signal", "bruit", "observation")
COULEURS = {"signal": "#2a78d6", "bruit": "#eb6834", "observation": "#4a4a4a"}
NOMS = {"signal": "signal seul", "bruit": r"bruit seul $\sigma\xi$", "observation": r"observation $X+\sigma\xi$"}

RACINE = Path(__file__).resolve().parents[1]
DOSSIER_RES, DOSSIER_FIG = RACINE / "resultats", RACINE / "figures"


def coupures(eps, sigma):
    """(nom, j_max) : sans coupure, puis règle d'E3."""
    return (("sans", J_FIN), ("avec", j_max_bruit(sigma, n, eps, J_FIN)))


def log2S_leaders(c, j_max):
    """log2 S(j, p) des leaders coupés à j_max, sur ECHELLES (NaN pour j > j_max)."""
    L = leaders(c, j_max)
    out = np.full((len(ECHELLES), P.size), np.nan)
    pres = [j for j in ECHELLES if j <= j_max]
    _, l, _ = log2_structure(L, P, pres)
    out[: len(pres)] = l
    return out


# ---------------- simulation ----------------
def simuler(nom):
    mod = MODELES[nom]
    rng = np.random.default_rng([GRAINE, list(MODELES).index(nom)])
    # res[sigma, r, coupure, composante, j, p]
    res = np.full((len(SIGMAS), R, 2, 3, len(ECHELLES), P.size), np.nan)
    jmax = np.array([[jm for _, jm in coupures(mod["eps"], s)] for s in SIGMAS])
    for r in range(R):
        x = mod["simuler"](rng)
        xi = rng.standard_normal(n)
        cx, cxi = coefficients_ondelettes(x, ONDELETTE, normalisation="L1"), coefficients_ondelettes(xi, ONDELETTE)
        for i, s in enumerate(SIGMAS):
            cb = coefficients_ondelettes(s * xi, ONDELETTE)
            cy = coefficients_ondelettes(x + s * xi, ONDELETTE)
            for k, jm in enumerate(jmax[i]):
                for q, c in enumerate((cx, cb, cy)):
                    res[i, r, k, q] = log2S_leaders(c, jm)
    return dict(res=res, jmax=jmax)


def charger_ou_simuler(nom, refaire):
    f = DOSSIER_RES / f"E6_{nom}.npz"
    if f.exists() and not refaire:
        return dict(np.load(f))
    print(f"Simulation {nom} ({R} répétitions)...")
    d = simuler(nom)
    DOSSIER_RES.mkdir(exist_ok=True)
    np.savez(f, **d)
    return d


# ---------------- théorie ----------------
def theorie_bruit(j, p, sigma, j_max):
    ecart_type = sigma * 2.0 ** (-(J - j_max) / 2)
    return p * np.log2(ecart_type * np.sqrt(2 * np.log(3 * 2.0 ** (j_max - j))))


def theorie_signal(j, p, zeta, med_signal_ref):
    return med_signal_ref - zeta(p) * (j - J_REF)


def croisement(p, sigma, j_max, zeta, med_signal_ref):
    """Première échelle (continue) où la droite du signal passe du « bon côté » du bruit."""
    jj = np.linspace(ECHELLES[0], j_max, 2001)
    d = theorie_signal(jj, p, zeta, med_signal_ref) - theorie_bruit(jj, p, sigma, j_max)
    d = d * np.sign(p)                         # p > 0 : signal > bruit ; p < 0 : signal < bruit (S en L^p)
    bas = np.where(d < 0)[0]
    return jj[bas[0]] if bas.size else np.nan


# ---------------- figures ----------------
def figure(nom, d, k):
    mod = MODELES[nom]
    res, jmax = d["res"], d["jmax"]
    js = np.array(ECHELLES)
    fig, axes = plt.subplots(len(SIGMAS), P.size, figsize=(14, 3.4 * len(SIGMAS)), sharex=True)
    for i, s in enumerate(SIGMAS):
        jm = int(jmax[i, k])
        for b, p in enumerate(P):
            ax = axes[i, b]
            ax.axvspan(6, 13, color="0.94", zorder=0)
            for q, (comp, dx) in enumerate(zip(COMPOSANTES, (-0.27, 0.0, 0.27))):
                vals = res[i, :, k, q, :, b]
                ok = [a for a in range(len(js)) if np.all(np.isfinite(vals[:, a]))]
                bp = ax.boxplot([vals[:, a] for a in ok], positions=js[ok] + dx, widths=0.22,
                                patch_artist=True, showfliers=True, manage_ticks=False,
                                flierprops=dict(markersize=1.5, alpha=0.4),
                                medianprops=dict(color="k", lw=1))
                for box in bp["boxes"]:
                    box.set_facecolor(COULEURS[comp]); box.set_alpha(0.75)
            ref = np.median(res[i, :, k, 0, ECHELLES.index(J_REF), b])
            jj = np.linspace(js[0], jm, 200)
            ax.plot(jj, theorie_bruit(jj, p, s, jm), "--", color=COULEURS["bruit"], lw=1.3)
            ax.plot(jj, theorie_signal(jj, p, mod["zeta"], ref), "--", color=COULEURS["signal"], lw=1.3)
            jx = croisement(p, s, jm, mod["zeta"], ref)
            if np.isfinite(jx):
                ax.axvline(jx, color="k", ls=":", lw=1.2)
            ax.axvline(jm, color="0.4", ls="--", lw=0.8)
            ax.set_xticks(js)
            ax.grid(alpha=0.3)
            ax.set_title(rf"$\sigma={s:g}$, $p={p:g}$, $j_{{\max}}={jm}$" +
                         (rf", $j_\times\approx{jx:.1f}$" if np.isfinite(jx) else ""), fontsize=10)
            if b == 0: ax.set_ylabel(r"$\log_2 S(j,p)$")
            if i == len(SIGMAS) - 1: ax.set_xlabel(r"$j$")
    poignees = [plt.Rectangle((0, 0), 1, 1, fc=COULEURS[c], alpha=0.75) for c in COMPOSANTES]
    poignees += [plt.Line2D([], [], color=COULEURS["bruit"], ls="--"), plt.Line2D([], [], color=COULEURS["signal"], ls="--"),
                 plt.Line2D([], [], color="k", ls=":"), plt.Line2D([], [], color="0.4", ls="--")]
    etiquettes = [NOMS[c] for c in COMPOSANTES] + ["théorie bruit", r"théorie signal (pente $-\zeta(p)$)",
                                                    r"croisement prédit $j_\times$", r"$j_{\max}$"]
    fig.legend(poignees, etiquettes, loc="lower center", ncol=7, fontsize=8, bbox_to_anchor=(0.5, -0.005))
    libelle = "sans coupure" if k == 0 else "coupure règle E3"
    fig.suptitle(f"E6 — {mod['titre']}, leaders {libelle} : décomposition signal / bruit / observation "
                 rf"($n=2^{{15}}$, $R={R}$, gris : $[6,13]$)")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def resume(nom, d):
    """Croisement prédit et observé (première échelle où la médiane de l'observation
    s'écarte de celle du signal de plus de |p|/2, i.e. un facteur sqrt(2) sur les leaders)."""
    mod = MODELES[nom]
    res, jmax = d["res"], d["jmax"]
    print(f"\n=== {nom} : croisements (prédit / observé) ===")
    for k, lib in enumerate(("sans", "avec")):
        for i, s in enumerate(SIGMAS):
            jm = int(jmax[i, k])
            lignes = []
            for b, p in enumerate(P):
                ref = np.median(res[i, :, k, 0, ECHELLES.index(J_REF), b])
                jx = croisement(p, s, jm, mod["zeta"], ref)
                ecart = np.abs(np.median(res[i, :, k, 2, :, b], 0) - np.median(res[i, :, k, 0, :, b], 0))
                obs = [j for j, e in zip(ECHELLES, ecart) if j <= jm and e > abs(p) / 2]
                lignes.append(f"p={p:+g} : prédit {jx:5.1f}, observé {obs[0] if obs else '—'}")
            print(f"  {lib:5s} sigma={s:<6g} j_max={jm:<3d} " + " | ".join(lignes))


if __name__ == "__main__":
    refaire = "--refaire" in sys.argv
    DOSSIER_FIG.mkdir(exist_ok=True)
    for nom in MODELES:
        d = charger_ou_simuler(nom, refaire)
        resume(nom, d)
        for k, lib in enumerate(("sans", "avec")):
            fig = figure(nom, d, k)
            fig.savefig(DOSSIER_FIG / f"E6_{nom}_{lib}_coupure.png", dpi=110)
            plt.close(fig)
    print(f"\nFigures écrites dans {DOSSIER_FIG}")
