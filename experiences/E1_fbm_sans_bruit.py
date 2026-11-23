"""E1 — fBm sans bruit : incréments, coefficients d'ondelettes et leaders, pour p > 0 et p < 0.

Question : à sigma = 0, quelle quantité multi-échelle reconstruit zeta(p) = pH et D = delta_(H, 1) ?

Usage (depuis la racine du dépôt) :
    python experiences/E1_fbm_sans_bruit.py            # simule si nécessaire, puis trace
    python experiences/E1_fbm_sans_bruit.py --refaire  # force la simulation
Sorties : resultats/E1_H*.npz (données brutes), figures/E1_H*_*.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spectre.legendre import legendre, legendre_parametrique
from spectre.modeles.fbm import simuler_fbm
from spectre.multiechelle.increments import increments
from spectre.multiechelle.leaders import leaders
from spectre.multiechelle.ondelettes import coefficients_ondelettes
from spectre.regression import estimer_zeta
from spectre.structure import log2_structure

# ---------------- paramètres (tout ce qui est fixé est ici) ----------------
H_VALEURS = (0.3, 0.7)
n = 2**15
sigma = 0.0
R = 200
ONDELETTE = "db3"
j1, j2 = 6, 13                       # plage de régression
ECHELLES = list(range(3, 15))        # échelles calculées et sauvegardées
P = np.round(np.arange(-5, 5.0001, 0.25), 2)
GRAINE = 20261123
QUANTITES = ("increments", "ondelettes", "leaders")
NOMS = {"increments": "Incréments", "ondelettes": "Coefficients d'ondelettes", "leaders": "Leaders"}
COULEURS = {"increments": "#2a78d6", "ondelettes": "#eb6834", "leaders": "#1baf7a"}

RACINE = Path(__file__).resolve().parents[1]
DOSSIER_RES, DOSSIER_FIG = RACINE / "resultats", RACINE / "figures"


# ---------------- simulation ----------------
def simuler(H: float) -> dict:
    rng = np.random.default_rng([GRAINE, int(round(1000 * H))])
    log2S = np.empty((R, len(QUANTITES), len(ECHELLES), P.size))
    nb = np.empty((R, len(QUANTITES), len(ECHELLES)), dtype=int)
    for r in range(R):
        x = simuler_fbm(n, H, rng)            # sigma = 0 : pas de bruit ajouté
        c = coefficients_ondelettes(x, ONDELETTE)
        for q, m in enumerate((increments(x), c, leaders(c))):
            js, log2S[r, q], nb[r, q] = log2_structure(m, P, ECHELLES)
    zeta = np.array([[estimer_zeta(js, log2S[r, q], j1, j2) for q in range(len(QUANTITES))]
                     for r in range(R)])     # forme (R, quantités, p)
    return dict(H=H, js=js, p=P, log2S=log2S, nb=nb, zeta=zeta)


def charger_ou_simuler(H: float, refaire: bool) -> dict:
    fichier = DOSSIER_RES / f"E1_H{H}.npz"
    if fichier.exists() and not refaire:
        return dict(np.load(fichier))
    print(f"Simulation H = {H} ({R} répétitions)...")
    res = simuler(H)
    DOSSIER_RES.mkdir(exist_ok=True)
    np.savez(fichier, **res)
    return res


# ---------------- figures ----------------
def fig_zeta(res, H):
    """Figure 1 : zeta_hat(p) (médiane, bande 5-95 %) contre pH."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for q, (ax, nom) in enumerate(zip(axes, QUANTITES)):
        z = res["zeta"][:, q]
        lo, med, hi = np.percentile(z, [5, 50, 95], axis=0)
        ax.fill_between(P, lo, hi, color=COULEURS[nom], alpha=0.25, lw=0, label="bande 5–95 %")
        ax.plot(P, med, color=COULEURS[nom], lw=2, label="médiane")
        ax.plot(P, P * H, "k--", lw=1, label=r"$pH$")
        ax.axvline(-1, color="0.6", lw=0.8, ls=":")
        ax.set_title(NOMS[nom]); ax.set_xlabel(r"$p$"); ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$\hat\zeta(p)$"); axes[0].legend(loc="upper left")
    axes[0].set_ylim(-5 * H - 1.5, 5 * H + 1.5)
    fig.suptitle(rf"E1 — fBm $H={H}$, $n=2^{{15}}$, $\sigma=0$, $R={R}$, $j\in[{j1},{j2}]$")
    return fig


def fig_boites(res, H):
    """Figure 2 : boîtes à moustaches de zeta_hat(p) - pH sur un sous-ensemble de p."""
    p_choisis = [-5, -3, -2, -1.5, -1, -0.5, 0.5, 1, 2, 3, 5]
    idx = [int(np.argmin(np.abs(P - v))) for v in p_choisis]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for q, (ax, nom) in enumerate(zip(axes, QUANTITES)):
        err = res["zeta"][:, q][:, idx] - P[idx] * H
        ax.boxplot(err, tick_labels=[f"{v:g}" for v in p_choisis], showfliers=True,
                   flierprops=dict(markersize=2, alpha=0.4))
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(NOMS[nom]); ax.set_xlabel(r"$p$"); ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel(r"$\hat\zeta(p) - pH$"); axes[0].set_ylim(-1, 1)
    fig.suptitle(rf"E1 — erreur sur $\zeta$, fBm $H={H}$ (échelle tronquée à $\pm1$)")
    return fig


def fig_log2S(res, H):
    """Figure 3 : log2 S(j, p) moyen contre j, plage de régression grisée."""
    p_choisis = [-3, -1, 1, 3]
    js = res["js"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for q, (ax, nom) in enumerate(zip(axes, QUANTITES)):
        ax.axvspan(j1, j2, color="0.9")
        for v, style in zip(p_choisis, ["o-", "s-", "^-", "d-"]):
            b = int(np.argmin(np.abs(P - v)))
            y = res["log2S"][:, q, :, b]
            ax.plot(js, np.nanmean(y, axis=0), style, color=COULEURS[nom], ms=4, lw=1.5,
                    alpha=0.4 + 0.15 * p_choisis.index(v), label=rf"$p={v}$")
        ax.set_title(NOMS[nom]); ax.set_xlabel(r"$j$"); ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"moyenne de $\log_2 S(j,p)$"); axes[0].legend()
    fig.suptitle(rf"E1 — fonctions de structure, fBm $H={H}$ (gris : plage $[{j1},{j2}]$)")
    return fig


def fig_spectre(res, H):
    """Figure 4 : D_hat(h), forme min_p (haut) et forme paramétrique (bas)."""
    h = np.linspace(H - 0.7, H + 0.7, 281)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True, sharey="row")
    for q, nom in enumerate(QUANTITES):
        z = res["zeta"][:, q]
        D = np.array([legendre(P, zr, h) for zr in z])
        lo, med, hi = np.percentile(D, [5, 50, 95], axis=0)
        ax = axes[0, q]
        ax.fill_between(h, lo, hi, color=COULEURS[nom], alpha=0.25, lw=0)
        ax.plot(h, med, color=COULEURS[nom], lw=2, label="médiane")
        ax.plot(h, 1 - P.max() * np.abs(h - H), "k--", lw=1, label=r"tente $1-5|h-H|$")
        ax.plot([H], [1], "k*", ms=10, label=r"$(H,1)$")
        ax.set_title(NOMS[nom] + r" — $\min_p$"); ax.grid(alpha=0.3)
        ax = axes[1, q]
        for zr in z:
            hp, Dp = legendre_parametrique(P, zr)
            ax.plot(hp, Dp, ".", color=COULEURS[nom], ms=2, alpha=0.05)
        hp, Dp = legendre_parametrique(P, np.median(z, axis=0))
        ax.plot(hp[P < 0], Dp[P < 0], "v", color="k", ms=4, label=r"médiane, $p<0$")
        ax.plot(hp[P > 0], Dp[P > 0], "^", color="0.5", ms=4, label=r"médiane, $p>0$")
        ax.plot([H], [1], "k*", ms=10)
        ax.set_title(NOMS[nom] + " — paramétrique"); ax.set_xlabel(r"$h$"); ax.grid(alpha=0.3)
    axes[0, 0].set_ylim(-1, 1.1); axes[1, 0].set_ylim(-1, 1.1); axes[0, 0].set_xlim(h[0], h[-1])
    axes[0, 0].set_ylabel(r"$\hat D(h)$"); axes[1, 0].set_ylabel(r"$\hat D$")
    axes[0, 0].legend(loc="lower center", fontsize=8); axes[1, 0].legend(loc="lower center", fontsize=8)
    fig.suptitle(rf"E1 — spectre estimé, fBm $H={H}$, $p\in[-5,5]$")
    return fig


def resume(res, H):
    """Tableau texte : biais médian et écart interquartile de zeta_hat(p) - pH."""
    print(f"\n=== H = {H} : médiane [écart interquartile] de zeta_hat(p) - pH ===")
    p_choisis = [-5, -2, -1, -0.5, 0.5, 1, 2, 5]
    print("p".ljust(12) + "".join(f"{v:>16g}" for v in p_choisis))
    for q, nom in enumerate(QUANTITES):
        ligne = nom.ljust(12)
        for v in p_choisis:
            b = int(np.argmin(np.abs(P - v)))
            e = res["zeta"][:, q, b] - v * H
            q1, m, q3 = np.percentile(e, [25, 50, 75])
            ligne += f"{m:>+8.3f} [{q3 - q1:.3f}]".rjust(16)
        print(ligne)


if __name__ == "__main__":
    refaire = "--refaire" in sys.argv
    DOSSIER_FIG.mkdir(exist_ok=True)
    for H in H_VALEURS:
        res = charger_ou_simuler(H, refaire)
        resume(res, H)
        for nom, f in [("zeta", fig_zeta), ("boites", fig_boites), ("log2S", fig_log2S), ("spectre", fig_spectre)]:
            fig = f(res, H)
            fig.tight_layout()
            fig.savefig(DOSSIER_FIG / f"E1_H{H}_{nom}.png", dpi=130)
            plt.close(fig)
    print(f"\nFigures écrites dans {DOSSIER_FIG}")
