"""E2–E3 — fBm bruité : effet du bruit, sans puis avec coupure des échelles fines.

Modèle : Y_i = B_H(i/n) + sigma * xi_i.
  E2 : chaîne inchangée (leaders calculés jusqu'à l'échelle la plus fine, régression sur [6, 13]).
  E3 : coupure à j_max(sigma, n, eps = H) : leaders tronqués à j_max, régression sur [j1', j2']
       avec j2' = min(13, j_max), j1' = min(6, j2' - 3). Coupure « oracle » : elle utilise H.
E2 et E3 sont calculés sur les mêmes réalisations (comparaison appariée).

Usage :  python experiences/E2_E3_fbm_bruit.py [--refaire]
Sorties : resultats/E23_H*_s*.npz, figures/E23_H*_*.png
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
from spectre.legendre import legendre
from spectre.modeles.fbm import simuler_fbm
from spectre.multiechelle.increments import increments
from spectre.multiechelle.leaders import leaders
from spectre.multiechelle.ondelettes import coefficients_ondelettes
from spectre.regression import estimer_zeta
from spectre.structure import log2_structure

# ---------------- paramètres ----------------
H_VALEURS = (0.3, 0.7)
SIGMAS = (0.0, 0.01, 0.1, 1.0)
n = 2**15
R = 200
ONDELETTE = "db3"
J1, J2 = 6, 13                       # plage de E2 (identique à E1)
ECHELLES = list(range(3, 15))
P = np.round(np.arange(-5, 5.0001, 0.25), 2)
GRAINE = 20261124
# quantités calculées : 0 incréments, 1 coefficients, 2 leaders sans coupure, 3 leaders coupés à j_max
QUANTITES = ("increments", "ondelettes", "leaders", "leaders_coupes")
NOMS = {"increments": "Incréments", "ondelettes": "Coefficients",
        "leaders": "Leaders (E2, sans coupure)", "leaders_coupes": "Leaders (E3, coupure)"}
COULEURS = {"increments": "#2a78d6", "ondelettes": "#eb6834", "leaders": "#1baf7a", "leaders_coupes": "#e87ba4"}

warnings.filterwarnings("ignore", message="Mean of empty slice")  # leaders coupés : j > j_max absent

RACINE = Path(__file__).resolve().parents[1]
DOSSIER_RES, DOSSIER_FIG = RACINE / "resultats", RACINE / "figures"


def plage_E3(j_max: int) -> tuple[int, int]:
    j2 = min(J2, j_max)
    return min(J1, j2 - 3), j2


def log2S_sur_echelles(m, p):
    """log2 S sur ECHELLES ; NaN aux échelles absentes (leaders coupés : j > j_max)."""
    out = np.full((len(ECHELLES), p.size), np.nan)
    pres = [j for j in ECHELLES if j in m.valeurs]
    _, l, _ = log2_structure(m, p, pres)
    out[[ECHELLES.index(j) for j in pres]] = l
    return out


# ---------------- simulation ----------------
def simuler(H: float, sigma: float) -> dict:
    rng = np.random.default_rng([GRAINE, int(round(1000 * H)), int(round(1e4 * sigma))])
    j_max = j_max_bruit(sigma, n, eps=H, j_plus_fin=ECHELLES[-1])
    a1, a2 = plage_E3(j_max)
    js = np.array(ECHELLES)
    log2S = np.empty((R, len(QUANTITES), len(ECHELLES), P.size))
    for r in range(R):
        y = ajouter_bruit(simuler_fbm(n, H, rng), sigma, rng)
        c = coefficients_ondelettes(y, ONDELETTE)
        for q, m in enumerate((increments(y), c, leaders(c), leaders(c, j_max))):
            log2S[r, q] = log2S_sur_echelles(m, P)
    zeta_E2 = np.array([[estimer_zeta(js, log2S[r, q], J1, J2) for q in range(3)] for r in range(R)])
    zeta_E3 = np.array([[estimer_zeta(js, log2S[r, q], a1, a2) for q in (0, 1, 3)] for r in range(R)])
    return dict(H=H, sigma=sigma, j_max=j_max, plage_E3=(a1, a2), js=js, p=P,
                log2S=log2S, zeta_E2=zeta_E2, zeta_E3=zeta_E3)


def charger_ou_simuler(H, sigma, refaire):
    f = DOSSIER_RES / f"E23_H{H}_s{sigma}.npz"
    if f.exists() and not refaire:
        return dict(np.load(f))
    print(f"Simulation H = {H}, sigma = {sigma} ({R} répétitions)...")
    res = simuler(H, sigma)
    DOSSIER_RES.mkdir(exist_ok=True)
    np.savez(f, **res)
    return res


# Colonnes affichées : (nom de la quantité, zeta_hat de forme (R, nombre de p))
def colonnes(res):
    return [("increments", res["zeta_E2"][:, 0]), ("ondelettes", res["zeta_E2"][:, 1]),
            ("leaders", res["zeta_E2"][:, 2]), ("leaders_coupes", res["zeta_E3"][:, 2])]


def titre_ligne(res):
    a1, a2 = res["plage_E3"]
    return rf"$\sigma={float(res['sigma']):g}$" + "\n" + rf"$j_{{\max}}={int(res['j_max'])}$, E3 : $[{a1},{a2}]$"


# ---------------- figures ----------------
def fig_zeta(tous, H):
    fig, axes = plt.subplots(len(tous), 4, figsize=(15, 3.1 * len(tous)), sharex=True, sharey=True)
    for i, res in enumerate(tous):
        for k, (nom, z) in enumerate(colonnes(res)):
            ax = axes[i, k]
            lo, med, hi = np.percentile(z, [5, 50, 95], axis=0)
            ax.fill_between(P, lo, hi, color=COULEURS[nom], alpha=0.25, lw=0)
            ax.plot(P, med, color=COULEURS[nom], lw=2)
            ax.plot(P, P * H, "k--", lw=1)
            ax.grid(alpha=0.3)
            if i == 0: ax.set_title(NOMS[nom])
            if k == 0: ax.set_ylabel(titre_ligne(res), fontsize=9)
    for ax in axes[-1]: ax.set_xlabel(r"$p$")
    axes[0, 0].set_ylim(-5 * H - 1.5, 5 * H + 1.5)
    fig.suptitle(rf"E2–E3 — $\hat\zeta(p)$ (médiane, bande 5–95 %) et $pH$ (tirets), fBm $H={H}$, $n=2^{{15}}$, $R={R}$")
    return fig


def fig_spectre(tous, H):
    h = np.linspace(H - 0.7, H + 0.7, 281)
    fig, axes = plt.subplots(len(tous), 4, figsize=(15, 3.1 * len(tous)), sharex=True, sharey=True)
    for i, res in enumerate(tous):
        for k, (nom, z) in enumerate(colonnes(res)):
            ax = axes[i, k]
            D = np.array([legendre(P, zr, h) for zr in z])
            lo, med, hi = np.percentile(D, [5, 50, 95], axis=0)
            ax.fill_between(h, lo, hi, color=COULEURS[nom], alpha=0.25, lw=0)
            ax.plot(h, med, color=COULEURS[nom], lw=2)
            ax.plot(h, 1 - P.max() * np.abs(h - H), "k--", lw=0.8)
            ax.plot([H], [1], "k*", ms=8)
            ax.grid(alpha=0.3)
            if i == 0: ax.set_title(NOMS[nom])
            if k == 0: ax.set_ylabel(titre_ligne(res), fontsize=9)
    for ax in axes[-1]: ax.set_xlabel(r"$h$")
    axes[0, 0].set_ylim(-1, 1.1); axes[0, 0].set_xlim(h[0], h[-1])
    fig.suptitle(rf"E2–E3 — $\hat D(h)=\min_p(1+ph-\hat\zeta(p))$, fBm $H={H}$ (tirets : tente $1-5|h-H|$)")
    return fig


def fig_log2S_leaders(tous, H):
    """Mécanisme : log2 S des leaders, sans et avec coupure, pour p = -2 et p = 2."""
    fig, axes = plt.subplots(2, len(tous), figsize=(4 * len(tous), 6.5), sharex=True)
    js = np.array(ECHELLES)
    for i, res in enumerate(tous):
        for a, v in enumerate((-2, 2)):
            ax = axes[a, i]
            b = int(np.argmin(np.abs(P - v)))
            ax.axvspan(J1, J2, color="0.92")
            for q, nom in ((2, "leaders"), (3, "leaders_coupes")):
                y = np.nanmean(res["log2S"][:, q, :, b], axis=0)
                ax.plot(js, y, "o-", color=COULEURS[nom], ms=3, label=NOMS[nom])
            y0 = np.nanmean(tous[0]["log2S"][:, 2, :, b], axis=0)      # sigma = 0 : référence
            ax.plot(js, y0, "k:", lw=1, label=r"leaders, $\sigma=0$")
            ax.axvline(int(res["j_max"]), color="0.4", lw=0.8, ls="--")
            ax.grid(alpha=0.3)
            if a == 0: ax.set_title(rf"$\sigma={float(res['sigma']):g}$, $j_{{\max}}={int(res['j_max'])}$")
            if i == 0: ax.set_ylabel(rf"$\log_2 S(j,{v})$ moyen")
            if a == 1: ax.set_xlabel(r"$j$")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(rf"E2–E3 — fonctions de structure des leaders, fBm $H={H}$ (gris : $[{J1},{J2}]$ ; tirets : $j_{{\max}}$)")
    return fig


def fig_boites(tous, H):
    """Boîtes à moustaches de zeta_hat(p) - pH, leaders E2 contre E3, selon sigma."""
    p_choisis = (-2, -1, 1, 2)
    fig, axes = plt.subplots(1, len(p_choisis), figsize=(15, 4), sharey=True)
    x = np.arange(len(tous))
    for a, v in enumerate(p_choisis):
        ax = axes[a]
        b = int(np.argmin(np.abs(P - v)))
        for d, (nom, cle, q) in zip((-0.18, 0.18), (("leaders", "zeta_E2", 2), ("leaders_coupes", "zeta_E3", 2))):
            err = [res[cle][:, q, b] - v * H for res in tous]
            bp = ax.boxplot(err, positions=x + d, widths=0.3, patch_artist=True, showfliers=True,
                            flierprops=dict(markersize=2, alpha=0.4))
            for box in bp["boxes"]:
                box.set_facecolor(COULEURS[nom]); box.set_alpha(0.6)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x, [f"{float(r['sigma']):g}" for r in tous])
        ax.set_xlabel(r"$\sigma$"); ax.set_title(rf"$p={v}$"); ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel(r"$\hat\zeta(p)-pH$")
    axes[0].legend([plt.Rectangle((0, 0), 1, 1, fc=COULEURS[k], alpha=0.6) for k in ("leaders", "leaders_coupes")],
                   [NOMS["leaders"], NOMS["leaders_coupes"]], fontsize=8)
    fig.suptitle(rf"E2–E3 — erreur des leaders sans / avec coupure, fBm $H={H}$, $R={R}$")
    return fig


def resume(tous, H):
    p_choisis = (-2, -1, 1, 2)
    print(f"\n=== H = {H} : médiane [écart interquartile] de zeta_hat(p) - pH ===")
    print("sigma  j_max  quantité".ljust(40) + "".join(f"{v:>16g}" for v in p_choisis))
    for res in tous:
        for nom, z in colonnes(res):
            ligne = f"{float(res['sigma']):<6g} {int(res['j_max']):<6d} {NOMS[nom]}".ljust(40)
            for v in p_choisis:
                b = int(np.argmin(np.abs(P - v)))
                q1, m, q3 = np.percentile(z[:, b] - v * H, [25, 50, 75])
                ligne += f"{m:>+8.3f} [{q3 - q1:.3f}]".rjust(16)
            print(ligne)


if __name__ == "__main__":
    refaire = "--refaire" in sys.argv
    DOSSIER_FIG.mkdir(exist_ok=True)
    for H in H_VALEURS:
        tous = [charger_ou_simuler(H, s, refaire) for s in SIGMAS]
        resume(tous, H)
        for nom, f in [("zeta", fig_zeta), ("spectre", fig_spectre),
                       ("log2S_leaders", fig_log2S_leaders), ("boites", fig_boites)]:
            fig = f(tous, H)
            fig.tight_layout()
            fig.savefig(DOSSIER_FIG / f"E23_H{H}_{nom}.png", dpi=120)
            plt.close(fig)
    print(f"\nFigures écrites dans {DOSSIER_FIG}")
