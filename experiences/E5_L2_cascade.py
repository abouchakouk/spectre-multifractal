"""E5 (L2) — W-cascade log-normale (spectre parabolique connu) : leaders des d_{j,k}, SANS coupure.

Question : sur un signal réellement multifractal, les leaders des d_{j,k} reconstruisent-ils
zeta(p) = m p - (b/2) p^2 et D(h) = 1 - (h - m)^2 / (2b) sans bruit, et comment le bruit les
déforme-t-il quand les leaders vont jusqu'à l'échelle la plus fine (j_max = 14) ?
Que change le passage L1 -> L2 par rapport à E5 (leaders des c_{j,k}, mêmes réalisations) ?

Modèle : X = W-cascade (m, b) (coefficients fabriqués dans la base : pas de défaut
d'initialisation), Y_i = X_i + sigma * xi_i.
Sur les d_{j,k} : bruit de niveau sigma / sqrt(n) à toutes les échelles.
Estimateur (voir docs/conventions.md) :
  L_{j,k} = sup |d| sur 3 lambda_{j,k} et ses descendants (j' <= 14),
  S_d(j,p) = 2^{-j} sum_k L_{j,k}^p,  zeta_d_hat = - pente de log2 S_d sur [6, 13],
  zeta_hat(p) = zeta_d_hat(p) - p/2,  D_hat(h) = min_p (1 + p h - zeta_hat(p)).
Cible : log2 S_d(j,p)/p = -(m - (b/2) p + 1/2) j + C(p)/p : la pente dépend de p (multifractalité).

Mêmes graine et mêmes réalisations que E5 (L1) : la comparaison L1 / L2 est appariée.

Usage :  python experiences/E5_L2_cascade.py [--refaire]
Sorties : resultats/E5_L2_s*.npz, figures/E5_L2_*.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spectre.bruit import ajouter_bruit
from spectre.legendre import legendre, legendre_parametrique
from spectre.modeles.wcascade import simuler_wcascade, spectre_theorique, zeta_theorique
from spectre.multiechelle.leaders import leaders
from spectre.multiechelle.ondelettes import coefficients_ondelettes
from spectre.regression import estimer_zeta
from spectre.structure import log2_structure

# ---------------- paramètres ----------------
M, B = 0.5, 0.025                    # zeta(p) = M p - (B/2) p^2 ; p critiques ± sqrt(2/B) ≈ ± 8.9
SIGMAS = (0.0, 0.01, 0.1, 1.0)
n = 2**15
R = 200
ONDELETTE = "db3"
J1, J2 = 6, 13                       # plage de régression (identique à E1, E2 et E5)
ECHELLES = list(range(3, 15))        # leaders jusqu'à j_max = 14 : pas de coupure
P = np.round(np.arange(-8, 8.0001, 0.25), 2)
GRAINE = 20261125                    # même graine que E5 (L1) : mêmes cascades et mêmes bruits
COUL_L2, COUL_L1 = "#1baf7a", "#8a8a8a"

RACINE = Path(__file__).resolve().parents[1]
DOSSIER_RES, DOSSIER_FIG = RACINE / "resultats", RACINE / "figures"


# ---------------- simulation ----------------
def simuler(sigma: float) -> dict:
    rng = np.random.default_rng([GRAINE, int(round(1e4 * sigma))])
    log2S = np.empty((R, len(ECHELLES), P.size))
    for r in range(R):
        y = ajouter_bruit(simuler_wcascade(n, M, B, rng, ONDELETTE), sigma, rng)
        d = coefficients_ondelettes(y, ONDELETTE, normalisation="L2")
        js, log2S[r], _ = log2_structure(leaders(d), P, ECHELLES)
    zeta_d = np.array([estimer_zeta(js, log2S[r], J1, J2) for r in range(R)])
    return dict(sigma=sigma, js=js, p=P, log2S=log2S, zeta_d=zeta_d, zeta=zeta_d - P / 2)


def charger_ou_simuler(sigma, refaire):
    f = DOSSIER_RES / f"E5_L2_s{sigma}.npz"
    if f.exists() and not refaire:
        return dict(np.load(f))
    print(f"Simulation sigma = {sigma} ({R} répétitions)...")
    res = simuler(sigma)
    DOSSIER_RES.mkdir(exist_ok=True)
    np.savez(f, **res)
    return res


def charger_L1(sigma):
    """zeta_hat des leaders L1 sans coupure d'E5 (mêmes réalisations), ou None si absent."""
    f = DOSSIER_RES / f"E5_s{sigma}.npz"
    if not f.exists():
        return None
    old = np.load(f)
    assert np.allclose(old["p"], P), "grille en p différente de celle d'E5"
    return old["zeta"][:, 2]                    # quantité 2 = leaders sans coupure


def niveau_bruit(sigma):
    """log2 du niveau du bruit sur les d_{j,k} : log2(sigma / sqrt(n)) (None si sigma = 0)."""
    return None if sigma == 0 else np.log2(sigma / np.sqrt(n))


# ---------------- figures ----------------
def fig_log2S_sur_p(tous):
    """Lignes : sigma ; colonnes : p. Moyenne de log2 S_d(j,p)/p, bande 5–95 %,
    cible de pente -(zeta(p)/p + 1/2) = -(M - (B/2) p + 1/2) (ordonnée ajustée sur la courbe
    sigma = 0, la même pour tous les sigma), niveau du bruit log2(sigma/sqrt(n)) en rouge."""
    p_choisis = [-6, -2, 2, 6]
    js = tous[0]["js"]
    sel = (js >= J1) & (js <= J2)
    fig, axes = plt.subplots(len(tous), len(p_choisis), figsize=(15, 3.1 * len(tous)), sharex=True)
    for k, v in enumerate(p_choisis):
        b = int(np.argmin(np.abs(P - v)))
        pente = zeta_theorique(v, M, B) / v + 0.5
        moy0 = (tous[0]["log2S"][:, :, b] / v).mean(axis=0)
        c = np.mean(moy0[sel] + pente * js[sel])
        for i, res in enumerate(tous):
            ax = axes[i, k]
            y = res["log2S"][:, :, b] / v
            ax.axvspan(J1, J2, color="0.92")
            ax.fill_between(js, np.percentile(y, 5, 0), np.percentile(y, 95, 0), color=COUL_L2, alpha=0.25, lw=0)
            ax.plot(js, y.mean(axis=0), "o-", color=COUL_L2, ms=3, lw=1.5, label="moyenne")
            ax.plot(js, c - pente * js, "k--", lw=1, label=rf"cible, pente $-{pente:.3f}$")
            nb = niveau_bruit(float(res["sigma"]))
            if nb is not None:
                ax.axhline(nb, color="#d9534f", lw=1, ls=":", label=r"$\log_2(\sigma/\sqrt{n})$")
            ax.grid(alpha=0.3)
            if i == 0: ax.set_title(rf"$p={v}$ (pente cible $-{pente:.3f}$)")
            if k == 0: ax.set_ylabel(rf"$\sigma={float(res['sigma']):g}$" + "\n" + r"$\log_2 S_d(j,p)/p$", fontsize=9)
            if i == len(tous) - 1: ax.set_xlabel(r"$j$")
    axes[1, 0].legend(fontsize=7)
    fig.suptitle(rf"E5 (L2) — leaders des $d_{{j,k}}$ sans coupure, W-cascade $m={M}$, $b={B}$, $n=2^{{15}}$, $R={R}$, db3"
                 rf" (gris : $[{J1},{J2}]$ ; rouge : niveau du bruit)")
    return fig


def fig_zeta(tous):
    zt = zeta_theorique(P, M, B)
    fig, axes = plt.subplots(1, len(tous), figsize=(4 * len(tous), 3.8), sharey=True)
    for ax, res in zip(axes, tous):
        lo, med, hi = np.percentile(res["zeta"], [5, 50, 95], axis=0)
        ax.fill_between(P, lo, hi, color=COUL_L2, alpha=0.25, lw=0, label="bande 5–95 %")
        ax.plot(P, med, color=COUL_L2, lw=2, label="médiane")
        ax.plot(P, zt, "k--", lw=1, label=r"$mp-\frac{b}{2}p^2$")
        ax.set_title(rf"$\sigma={float(res['sigma']):g}$"); ax.set_xlabel(r"$p$"); ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$\hat\zeta(p)=\hat\zeta_d(p)-p/2$"); axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(rf"E5 (L2) — $\hat\zeta(p)$, W-cascade $m={M}$, $b={B}$, $R={R}$, $j\in[{J1},{J2}]$")
    return fig


def fig_boites(tous, L1):
    """zeta_hat(p) - zeta(p) selon sigma : leaders L2 (vert) et L1 d'E5 (gris), mêmes réalisations."""
    p_choisis = (-6, -2, -1, 1, 2, 6)
    avec_L1 = all(z is not None for z in L1)
    fig, axes = plt.subplots(1, len(p_choisis), figsize=(18, 4), sharey=True)
    x = np.arange(len(tous))
    for a, v in enumerate(p_choisis):
        ax = axes[a]
        b = int(np.argmin(np.abs(P - v)))
        series = [([r["zeta"] for r in tous], COUL_L2, 0.18 if avec_L1 else 0.0)]
        if avec_L1:
            series.append((L1, COUL_L1, -0.18))
        for zs, coul, dx in series:
            err = [z[:, b] - zeta_theorique(v, M, B) for z in zs]
            bp = ax.boxplot(err, positions=x + dx, widths=0.3, patch_artist=True,
                            flierprops=dict(markersize=2, alpha=0.4))
            for box in bp["boxes"]:
                box.set_facecolor(coul); box.set_alpha(0.6)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x, [f"{float(r['sigma']):g}" for r in tous])
        ax.set_xlabel(r"$\sigma$"); ax.set_title(rf"$p={v}$"); ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel(r"$\hat\zeta(p)-\zeta(p)$")
    noms = ["leaders L2 ($d_{j,k}$)"] + (["leaders L1 ($c_{j,k}$, E5)"] if avec_L1 else [])
    axes[0].legend([plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.6) for c in (COUL_L2, COUL_L1)][:len(noms)], noms, fontsize=8)
    fig.suptitle(rf"E5 — erreur sur $\zeta$ selon $\sigma$, W-cascade, $R={R}$ (sans coupure, mêmes réalisations L1/L2)")
    return fig


def fig_spectre(tous):
    """Ligne du haut : min_p sur toute la grille p in [-8, 8] ; ligne du bas : paramétrique.
    Tirets : D théorique ; pointillés gris : Legendre du vrai zeta sur la même grille (oracle)."""
    h = np.linspace(M - 1.2, M + 0.6, 361)
    Dt = spectre_theorique(h, M, B)
    Dg = legendre(P, zeta_theorique(P, M, B), h)
    fig, axes = plt.subplots(2, len(tous), figsize=(4 * len(tous), 7), sharex=True, sharey="row")
    for i, res in enumerate(tous):
        D = np.array([legendre(P, zr, h) for zr in res["zeta"]])
        lo, med, hi = np.percentile(D, [5, 50, 95], axis=0)
        ax = axes[0, i]
        ax.fill_between(h, lo, hi, color=COUL_L2, alpha=0.25, lw=0)
        ax.plot(h, med, color=COUL_L2, lw=2, label="médiane")
        ax.plot(h, Dt, "k--", lw=1, label=r"$D$ théorique")
        ax.plot(h, Dg, ":", color="0.4", lw=1, label="Legendre du vrai $\\zeta$ (grille)")
        ax.set_title(rf"$\sigma={float(res['sigma']):g}$ — $\min_p$"); ax.grid(alpha=0.3)
        ax = axes[1, i]
        for zr in res["zeta"][::4]:
            hp, Dp = legendre_parametrique(P, zr)
            ax.plot(hp, Dp, ".", color=COUL_L2, ms=2, alpha=0.1)
        hp, Dp = legendre_parametrique(P, np.median(res["zeta"], axis=0))
        ax.plot(hp[P < 0], Dp[P < 0], "v", color="k", ms=3, label=r"médiane, $p<0$")
        ax.plot(hp[P > 0], Dp[P > 0], "^", color="0.5", ms=3, label=r"médiane, $p>0$")
        ax.plot(h, Dt, "k--", lw=1)
        ax.set_title("paramétrique"); ax.set_xlabel(r"$h$"); ax.grid(alpha=0.3)
    axes[0, 0].set_ylim(-1, 1.1); axes[1, 0].set_ylim(-0.2, 1.1); axes[0, 0].set_xlim(h[0], h[-1])
    axes[0, 0].set_ylabel(r"$\hat D(h)$"); axes[1, 0].set_ylabel(r"$\hat D$")
    axes[0, 0].legend(loc="lower left", fontsize=7); axes[1, 0].legend(loc="lower left", fontsize=7)
    fig.suptitle(rf"E5 (L2) — spectre estimé, W-cascade $m={M}$, $b={B}$, $p\in[-8,8]$, $R={R}$")
    return fig


def resume(tous, L1):
    p_choisis = (-6, -2, -1, 1, 2, 6)
    print(f"\n=== E5 (L2) : médiane [écart interquartile] de zeta_hat(p) - zeta(p) ===")
    print("sigma  leaders".ljust(16) + "".join(f"{v:>16g}" for v in p_choisis))
    for res, z1 in zip(tous, L1):
        for nom, z in [("L2", res["zeta"])] + ([("L1", z1)] if z1 is not None else []):
            ligne = f"{float(res['sigma']):<6g} {nom}".ljust(16)
            for v in p_choisis:
                b = int(np.argmin(np.abs(P - v)))
                q1, med, q3 = np.percentile(z[:, b] - zeta_theorique(v, M, B), [25, 50, 75])
                ligne += f"{med:>+8.3f} [{q3 - q1:.3f}]".rjust(16)
            print(ligne)


if __name__ == "__main__":
    refaire = "--refaire" in sys.argv
    DOSSIER_FIG.mkdir(exist_ok=True)
    tous = [charger_ou_simuler(s, refaire) for s in SIGMAS]
    L1 = [charger_L1(s) for s in SIGMAS]
    resume(tous, L1)
    figures = [("log2S_sur_p", fig_log2S_sur_p(tous)), ("zeta", fig_zeta(tous)),
               ("boites", fig_boites(tous, L1)), ("spectre", fig_spectre(tous))]
    for nom, fig in figures:
        fig.tight_layout()
        fig.savefig(DOSSIER_FIG / f"E5_L2_{nom}.png", dpi=120)
        plt.close(fig)
    print(f"\nFigures écrites dans {DOSSIER_FIG}")
