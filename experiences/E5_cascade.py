"""E5 — Cascade d'ondelettes log-normale (spectre parabolique connu), sans puis avec bruit.

Modèle : X = W-cascade (m, b), Y_i = X_i + sigma * xi_i.
Théorie : zeta(p) = m p - (b/2) p^2,  D(h) = 1 - (h - m)^2 / (2b),  p critiques ± sqrt(2/b).
Quantités : incréments, coefficients, leaders sans coupure (régression [6, 13]),
            leaders coupés à j_max(sigma, n, eps = h_min) avec régression [j1', j2'],
            j2' = min(13, j_max - 1), j1' = min(6, j2' - 3)   (règle j2 <= j_max - 1).
Pas de défaut d'initialisation (coefficients fabriqués dans la base) : test de l'explication d'E1.

Usage :  python experiences/E5_cascade.py [--refaire]
Sorties : resultats/E5_s*.npz, figures/E5_*.png
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spectre.bruit import ajouter_bruit
from spectre.echelles import j_max_bruit
from spectre.legendre import legendre, legendre_parametrique
from spectre.modeles.wcascade import simuler_wcascade, spectre_theorique, zeta_theorique
from spectre.multiechelle.increments import increments
from spectre.multiechelle.leaders import leaders
from spectre.multiechelle.ondelettes import coefficients_ondelettes
from spectre.regression import estimer_zeta
from spectre.structure import log2_structure

# ---------------- paramètres ----------------
M, B = 0.5, 0.025                    # zeta(p) = M p - (B/2) p^2
H_MIN = M - np.sqrt(2 * B)           # plus petit exposant (D = 0) : sert à la coupure
SIGMAS = (0.0, 0.01, 0.1, 1.0)
n = 2**15
R = 200
ONDELETTE = "db3"
J1, J2 = 6, 13
ECHELLES = list(range(3, 15))
P = np.round(np.arange(-8, 8.0001, 0.25), 2)
GRAINE = 20261125
QUANTITES = ("increments", "ondelettes", "leaders", "leaders_coupes")
NOMS = {"increments": "Incréments", "ondelettes": "Coefficients",
        "leaders": "Leaders (sans coupure)", "leaders_coupes": "Leaders (coupure)"}
COULEURS = {"increments": "#2a78d6", "ondelettes": "#eb6834", "leaders": "#1baf7a", "leaders_coupes": "#e87ba4"}

warnings.filterwarnings("ignore", message="Mean of empty slice")

RACINE = Path(__file__).resolve().parents[1]
DOSSIER_RES, DOSSIER_FIG = RACINE / "resultats", RACINE / "figures"


def plage_coupure(j_max: int) -> tuple[int, int]:
    j2 = min(J2, j_max - 1)
    return min(J1, j2 - 3), j2


def log2S_sur_echelles(m, p):
    out = np.full((len(ECHELLES), p.size), np.nan)
    pres = [j for j in ECHELLES if j in m.valeurs]
    _, l, _ = log2_structure(m, p, pres)
    out[[ECHELLES.index(j) for j in pres]] = l
    return out


# ---------------- simulation ----------------
def simuler(sigma: float) -> dict:
    rng = np.random.default_rng([GRAINE, int(round(1e4 * sigma))])
    j_max = j_max_bruit(sigma, n, eps=H_MIN, j_plus_fin=ECHELLES[-1])
    a1, a2 = plage_coupure(j_max)
    js = np.array(ECHELLES)
    log2S = np.empty((R, len(QUANTITES), len(ECHELLES), P.size))
    for r in range(R):
        y = ajouter_bruit(simuler_wcascade(n, M, B, rng, ONDELETTE), sigma, rng)
        c = coefficients_ondelettes(y, ONDELETTE, normalisation="L1")
        for q, m in enumerate((increments(y), c, leaders(c), leaders(c, j_max))):
            log2S[r, q] = log2S_sur_echelles(m, P)
    zeta = np.empty((R, len(QUANTITES), P.size))
    for r in range(R):
        for q in range(3):
            zeta[r, q] = estimer_zeta(js, log2S[r, q], J1, J2)
        zeta[r, 3] = estimer_zeta(js, log2S[r, 3], a1, a2)
    return dict(sigma=sigma, j_max=j_max, plage_coupure=(a1, a2), js=js, p=P, log2S=log2S, zeta=zeta)


def charger_ou_simuler(sigma, refaire):
    f = DOSSIER_RES / f"E5_s{sigma}.npz"
    if f.exists() and not refaire:
        return dict(np.load(f))
    print(f"Simulation sigma = {sigma} ({R} répétitions)...")
    res = simuler(sigma)
    DOSSIER_RES.mkdir(exist_ok=True)
    np.savez(f, **res)
    return res


def titre_ligne(res):
    a1, a2 = res["plage_coupure"]
    return rf"$\sigma={float(res['sigma']):g}$" + "\n" + rf"$j_{{\max}}={int(res['j_max'])}$, coupure : $[{a1},{a2}]$"


# ---------------- figures ----------------
def fig_zeta(tous):
    zt = zeta_theorique(P, M, B)
    fig, axes = plt.subplots(len(tous), 4, figsize=(15, 3.1 * len(tous)), sharex=True, sharey=True)
    for i, res in enumerate(tous):
        for q, nom in enumerate(QUANTITES):
            ax = axes[i, q]
            lo, med, hi = np.percentile(res["zeta"][:, q], [5, 50, 95], axis=0)
            ax.fill_between(P, lo, hi, color=COULEURS[nom], alpha=0.25, lw=0)
            ax.plot(P, med, color=COULEURS[nom], lw=2)
            ax.plot(P, zt, "k--", lw=1)
            ax.axvline(-1, color="0.6", lw=0.8, ls=":")
            ax.grid(alpha=0.3)
            if i == 0: ax.set_title(NOMS[nom])
            if q == 0: ax.set_ylabel(titre_ligne(res), fontsize=9)
    for ax in axes[-1]: ax.set_xlabel(r"$p$")
    axes[0, 0].set_ylim(-7, 5)
    fig.suptitle(rf"E5 — W-cascade $m={M}$, $b={B}$ : $\hat\zeta(p)$ (médiane, bande 5–95 %), théorie $mp-\frac{{b}}{{2}}p^2$ (tirets), $R={R}$")
    return fig


def fig_spectre(tous, p_lim=5.0):
    """Forme min_p sur p in [-p_lim, p_lim] (même grille que E1–E3) ; théorie en tirets."""
    sel = np.abs(P) <= p_lim
    h = np.linspace(M - 0.6, M + 0.6, 241)
    Dt = spectre_theorique(h, M, B)
    Dg = legendre(P[sel], zeta_theorique(P[sel], M, B), h)       # oracle : vrai zeta, même grille
    fig, axes = plt.subplots(len(tous), 4, figsize=(15, 3.1 * len(tous)), sharex=True, sharey=True)
    for i, res in enumerate(tous):
        for q, nom in enumerate(QUANTITES):
            ax = axes[i, q]
            D = np.array([legendre(P[sel], zr[sel], h) for zr in res["zeta"][:, q]])
            lo, med, hi = np.percentile(D, [5, 50, 95], axis=0)
            ax.fill_between(h, lo, hi, color=COULEURS[nom], alpha=0.25, lw=0)
            ax.plot(h, med, color=COULEURS[nom], lw=2)
            ax.plot(h, Dt, "k--", lw=1)
            ax.plot(h, Dg, ":", color="0.4", lw=1)
            ax.grid(alpha=0.3)
            if i == 0: ax.set_title(NOMS[nom])
            if q == 0: ax.set_ylabel(titre_ligne(res), fontsize=9)
    for ax in axes[-1]: ax.set_xlabel(r"$h$")
    axes[0, 0].set_ylim(-1, 1.1); axes[0, 0].set_xlim(h[0], h[-1])
    fig.suptitle(rf"E5 — $\hat D(h)=\min_p$, $p\in[-{p_lim:g},{p_lim:g}]$ ; tirets : $D$ théorique ; pointillés : Legendre du vrai $\zeta$ sur la grille")
    return fig


def fig_parametrique(tous, p_lim=5.0):
    sel = np.abs(P) <= p_lim
    h = np.linspace(M - 0.6, M + 0.6, 241)
    fig, axes = plt.subplots(len(tous), 4, figsize=(15, 3.1 * len(tous)), sharex=True, sharey=True)
    for i, res in enumerate(tous):
        for q, nom in enumerate(QUANTITES):
            ax = axes[i, q]
            for zr in res["zeta"][::4, q]:
                hp, Dp = legendre_parametrique(P[sel], zr[sel])
                ax.plot(hp, Dp, ".", color=COULEURS[nom], ms=2, alpha=0.1)
            hp, Dp = legendre_parametrique(P[sel], np.median(res["zeta"][:, q], axis=0)[sel])
            ax.plot(hp, Dp, "k.", ms=3)
            ax.plot(h, spectre_theorique(h, M, B), "k--", lw=1)
            ax.grid(alpha=0.3)
            if i == 0: ax.set_title(NOMS[nom])
            if q == 0: ax.set_ylabel(titre_ligne(res), fontsize=9)
    for ax in axes[-1]: ax.set_xlabel(r"$h$")
    axes[0, 0].set_ylim(-1, 1.1); axes[0, 0].set_xlim(h[0], h[-1])
    fig.suptitle(rf"E5 — forme paramétrique $(\hat\zeta'(p),\,1+p\hat\zeta'(p)-\hat\zeta(p))$, $p\in[-{p_lim:g},{p_lim:g}]$ (points noirs : médiane de $\hat\zeta$)")
    return fig


def fig_boites(tous):
    p_choisis = (-4, -2, -1, 1, 2, 4)
    fig, axes = plt.subplots(1, len(p_choisis), figsize=(17, 4), sharey=True)
    x = np.arange(len(tous))
    for a, v in enumerate(p_choisis):
        ax = axes[a]
        b = int(np.argmin(np.abs(P - v)))
        for d, q in zip((-0.27, -0.09, 0.09, 0.27), range(4)):
            err = [res["zeta"][:, q, b] - zeta_theorique(v, M, B) for res in tous]
            bp = ax.boxplot(err, positions=x + d, widths=0.16, patch_artist=True,
                            flierprops=dict(markersize=1.5, alpha=0.4))
            for box in bp["boxes"]:
                box.set_facecolor(COULEURS[QUANTITES[q]]); box.set_alpha(0.7)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x, [f"{float(r['sigma']):g}" for r in tous])
        ax.set_xlabel(r"$\sigma$"); ax.set_title(rf"$p={v}$"); ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel(r"$\hat\zeta(p)-\zeta(p)$"); axes[0].set_ylim(-1.5, 1.5)
    axes[0].legend([plt.Rectangle((0, 0), 1, 1, fc=COULEURS[k], alpha=0.7) for k in QUANTITES],
                   [NOMS[k] for k in QUANTITES], fontsize=7)
    fig.suptitle(rf"E5 — erreur sur $\zeta$ selon $\sigma$ (échelle tronquée à $\pm1{{,}}5$), $R={R}$")
    return fig


def fig_log2S(tous):
    p_choisis = (-2, 2)
    js = np.array(ECHELLES)
    fig, axes = plt.subplots(2, len(tous), figsize=(4 * len(tous), 6.5), sharex=True)
    for i, res in enumerate(tous):
        for a, v in enumerate(p_choisis):
            ax = axes[a, i]
            b = int(np.argmin(np.abs(P - v)))
            ax.axvspan(J1, J2, color="0.92")
            for q in range(4):
                ax.plot(js, np.nanmean(res["log2S"][:, q, :, b], axis=0), "o-", ms=3,
                        color=COULEURS[QUANTITES[q]], label=NOMS[QUANTITES[q]])
            ax.axvline(int(res["j_max"]), color="0.4", lw=0.8, ls="--")
            ax.grid(alpha=0.3)
            if a == 0: ax.set_title(rf"$\sigma={float(res['sigma']):g}$, $j_{{\max}}={int(res['j_max'])}$")
            if i == 0: ax.set_ylabel(rf"$\log_2 S(j,{v})$ moyen")
            if a == 1: ax.set_xlabel(r"$j$")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(r"E5 — fonctions de structure (gris : $[6,13]$ ; tirets : $j_{\max}$)")
    return fig


def resume(tous):
    p_choisis = (-4, -2, -1, 1, 2, 4)
    print(f"\n=== E5 : médiane [écart interquartile] de zeta_hat(p) - zeta(p) ===")
    print("sigma  j_max  quantité".ljust(38) + "".join(f"{v:>16g}" for v in p_choisis))
    for res in tous:
        for q, nom in enumerate(QUANTITES):
            ligne = f"{float(res['sigma']):<6g} {int(res['j_max']):<6d} {NOMS[nom]}".ljust(38)
            for v in p_choisis:
                b = int(np.argmin(np.abs(P - v)))
                q1, m, q3 = np.percentile(res["zeta"][:, q, b] - zeta_theorique(v, M, B), [25, 50, 75])
                ligne += f"{m:>+8.3f} [{q3 - q1:.3f}]".rjust(16)
            print(ligne)


if __name__ == "__main__":
    refaire = "--refaire" in sys.argv
    DOSSIER_FIG.mkdir(exist_ok=True)
    tous = [charger_ou_simuler(s, refaire) for s in SIGMAS]
    resume(tous)
    for nom, f in [("zeta", fig_zeta), ("spectre", fig_spectre), ("parametrique", fig_parametrique),
                   ("boites", fig_boites), ("log2S", fig_log2S)]:
        fig = f(tous)
        fig.tight_layout()
        fig.savefig(DOSSIER_FIG / f"E5_{nom}.png", dpi=120)
        plt.close(fig)
    print(f"\nFigures écrites dans {DOSSIER_FIG}")
