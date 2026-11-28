"""E1 (L2) — fBm sans bruit : leaders calculés sur les coefficients orthonormés d_{j,k}.

Question : à sigma = 0, les leaders des d_{j,k} redonnent-ils zeta(p) = pH et D = delta_(H, 1) ?
Que change le passage L1 -> L2 par rapport à E1 (leaders des c_{j,k} = 2^{j/2} d_{j,k}) ?

Définitions (voir docs/conventions.md) :
  d_{j,k} = <f, psi_{j,k}>,  L_{j,k} = sup |d| sur 3 lambda_{j,k} et ses descendants (j' <= 14),
  S_d(j,p) = 2^{-j} sum_k L_{j,k}^p ~ 2^{-j zeta_d(p)},  zeta_d(p) = zeta(p) + p/2,
  zeta_hat(p) = zeta_d_hat(p) - p/2   (correction faite AVANT Legendre et avant comparaison à pH),
  D_hat(h) = min_p (1 + p h - zeta_hat(p)).
Cible : log2 S_d(j,p) / p = -(H + 1/2) j + C(p)/p, même pente pour tout p.

Mêmes graine et mêmes réalisations que E1 (L1) : la comparaison L1 / L2 est appariée.

Usage (depuis la racine du dépôt) :
    python experiences/E1_L2_fbm_sans_bruit.py            # simule si nécessaire, puis trace
    python experiences/E1_L2_fbm_sans_bruit.py --refaire  # force la simulation
Sorties : resultats/E1_L2_H*.npz, figures/E1_L2_H*_*.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spectre.legendre import legendre, legendre_parametrique
from spectre.modeles.fbm import simuler_fbm
from spectre.multiechelle.leaders import leaders
from spectre.multiechelle.ondelettes import coefficients_ondelettes
from spectre.regression import estimer_zeta
from spectre.structure import log2_structure

# ---------------- paramètres (tout ce qui est fixé est ici) ----------------
H_VALEURS = (0.3, 0.7)
n = 2**15
R = 200
ONDELETTE = "db3"
J1, J2 = 6, 13                       # plage de régression (identique à E1)
ECHELLES = list(range(3, 15))        # échelles calculées ; leaders jusqu'à j_max = 14 (pas de coupure)
P = np.round(np.arange(-5, 5.0001, 0.25), 2)
GRAINE = 20261123                    # même graine que E1 (L1) : mêmes trajectoires
COUL_L2, COUL_L1 = "#1baf7a", "#8a8a8a"

RACINE = Path(__file__).resolve().parents[1]
DOSSIER_RES, DOSSIER_FIG = RACINE / "resultats", RACINE / "figures"


# ---------------- simulation ----------------
def simuler(H: float) -> dict:
    rng = np.random.default_rng([GRAINE, int(round(1000 * H))])
    log2S = np.empty((R, len(ECHELLES), P.size))
    nb = np.empty((R, len(ECHELLES)), dtype=int)
    for r in range(R):
        x = simuler_fbm(n, H, rng)                                   # sigma = 0
        d = coefficients_ondelettes(x, ONDELETTE, normalisation="L2")
        js, log2S[r], nb[r] = log2_structure(leaders(d), P, ECHELLES)
    zeta_d = np.array([estimer_zeta(js, log2S[r], J1, J2) for r in range(R)])
    return dict(H=H, js=js, p=P, log2S=log2S, nb=nb, zeta_d=zeta_d, zeta=zeta_d - P / 2)


def charger_ou_simuler(H: float, refaire: bool) -> dict:
    fichier = DOSSIER_RES / f"E1_L2_H{H}.npz"
    if fichier.exists() and not refaire:
        return dict(np.load(fichier))
    print(f"Simulation H = {H} ({R} répétitions)...")
    res = simuler(H)
    DOSSIER_RES.mkdir(exist_ok=True)
    np.savez(fichier, **res)
    return res


def charger_L1(H: float):
    """zeta_hat des leaders L1 d'E1 (mêmes réalisations), ou None si E1 n'a pas été lancé."""
    f = DOSSIER_RES / f"E1_H{H}.npz"
    if not f.exists():
        return None
    old = np.load(f)
    assert np.allclose(old["p"], P), "grille en p différente de celle d'E1"
    return old["zeta"][:, 2]                    # quantité 2 = leaders


# ---------------- figures ----------------
def fig_log2S_sur_p(res, H):
    """log2 S_d(j,p)/p : moyenne, bande 5–95 %, et droite cible de pente -(H+1/2).
    La cible n'a pas de constante connue : on prend l'ordonnée qui ajuste au mieux la
    courbe moyenne sur [J1, J2] à pente imposée."""
    p_choisis = [-5, -3, -1, 1, 3, 5]
    js = res["js"]
    sel = (js >= J1) & (js <= J2)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, v in zip(axes.ravel(), p_choisis):
        b = int(np.argmin(np.abs(P - v)))
        y = res["log2S"][:, :, b] / v
        lo, moy, hi = np.percentile(y, 5, axis=0), y.mean(axis=0), np.percentile(y, 95, axis=0)
        ax.axvspan(J1, J2, color="0.92")
        ax.fill_between(js, lo, hi, color=COUL_L2, alpha=0.25, lw=0, label="bande 5–95 %")
        ax.plot(js, moy, "o-", color=COUL_L2, ms=4, lw=1.5, label="moyenne")
        c = np.mean(moy[sel] + (H + 0.5) * js[sel])
        ax.plot(js, c - (H + 0.5) * js, "k--", lw=1, label=rf"cible, pente $-(H+1/2)$")
        ax.set_title(rf"$p={v}$"); ax.grid(alpha=0.3)
    for ax in axes[1]: ax.set_xlabel(r"$j$")
    for ax in axes[:, 0]: ax.set_ylabel(r"$\log_2 S_d(j,p)\,/\,p$")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(rf"E1 (L2) — leaders des $d_{{j,k}}$, fBm $H={H}$, $n=2^{{15}}$, $\sigma=0$, $R={R}$, db3"
                 rf" (gris : régression $[{J1},{J2}]$)")
    return fig


def fig_zeta(res, H):
    """zeta_hat(p) = zeta_d_hat(p) - p/2 (médiane, bande 5–95 %) contre pH."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    lo, med, hi = np.percentile(res["zeta"], [5, 50, 95], axis=0)
    ax.fill_between(P, lo, hi, color=COUL_L2, alpha=0.25, lw=0, label="bande 5–95 %")
    ax.plot(P, med, color=COUL_L2, lw=2, label="médiane")
    ax.plot(P, P * H, "k--", lw=1, label=r"$pH$")
    ax.set_xlabel(r"$p$"); ax.set_ylabel(r"$\hat\zeta(p)=\hat\zeta_d(p)-p/2$"); ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_title(rf"E1 (L2) — fBm $H={H}$, $R={R}$, $j\in[{J1},{J2}]$")
    return fig


def fig_boites(res, H, zeta_L1):
    """Boîtes à moustaches de zeta_hat(p) - pH : leaders L2 (vert) et, si disponibles, L1 d'E1 (gris)."""
    p_choisis = [-5, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 5]
    idx = [int(np.argmin(np.abs(P - v))) for v in p_choisis]
    x = np.arange(len(p_choisis))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    series = [(res["zeta"], COUL_L2, "leaders L2 ($d_{j,k}$)", 0.18 if zeta_L1 is not None else 0.0)]
    if zeta_L1 is not None:
        series.append((zeta_L1, COUL_L1, "leaders L1 ($c_{j,k}$, E1)", -0.18))
    for z, coul, nom, dx in series:
        err = z[:, idx] - P[idx] * H
        bp = ax.boxplot(err, positions=x + dx, widths=0.3, patch_artist=True,
                        flierprops=dict(markersize=2, alpha=0.4))
        for box in bp["boxes"]:
            box.set_facecolor(coul); box.set_alpha(0.6)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [f"{v:g}" for v in p_choisis]); ax.set_xlabel(r"$p$")
    ax.set_ylabel(r"$\hat\zeta(p)-pH$"); ax.grid(alpha=0.3, axis="y")
    ax.legend([plt.Rectangle((0, 0), 1, 1, fc=s[1], alpha=0.6) for s in series], [s[2] for s in series])
    ax.set_title(rf"E1 — erreur sur $\zeta$, fBm $H={H}$, $R={R}$ (mêmes réalisations pour L1 et L2)")
    return fig


def fig_spectre(res, H):
    """Spectre : forme min_p (gauche) et forme paramétrique (droite)."""
    h = np.linspace(H - 0.7, H + 0.7, 281)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    D = np.array([legendre(P, zr, h) for zr in res["zeta"]])
    lo, med, hi = np.percentile(D, [5, 50, 95], axis=0)
    ax = axes[0]
    ax.fill_between(h, lo, hi, color=COUL_L2, alpha=0.25, lw=0, label="bande 5–95 %")
    ax.plot(h, med, color=COUL_L2, lw=2, label="médiane")
    ax.plot(h, 1 - P.max() * np.abs(h - H), "k--", lw=1, label=r"tente $1-5|h-H|$ (grille finie)")
    ax.plot([H], [1], "k*", ms=10, label=r"$(H,1)$")
    ax.set_title(r"$\min_p\,(1+ph-\hat\zeta(p))$"); ax.set_xlabel(r"$h$"); ax.set_ylabel(r"$\hat D(h)$")
    ax.legend(loc="lower center", fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1]
    for zr in res["zeta"]:
        hp, Dp = legendre_parametrique(P, zr)
        ax.plot(hp, Dp, ".", color=COUL_L2, ms=2, alpha=0.05)
    hp, Dp = legendre_parametrique(P, np.median(res["zeta"], axis=0))
    ax.plot(hp[P < 0], Dp[P < 0], "v", color="k", ms=4, label=r"médiane, $p<0$")
    ax.plot(hp[P > 0], Dp[P > 0], "^", color="0.5", ms=4, label=r"médiane, $p>0$")
    ax.plot([H], [1], "k*", ms=10)
    ax.set_title(r"paramétrique $(\hat\zeta'(p),\,1+p\hat\zeta'(p)-\hat\zeta(p))$"); ax.set_xlabel(r"$h$")
    ax.legend(loc="lower center", fontsize=8); ax.grid(alpha=0.3)
    axes[0].set_ylim(-1, 1.1); axes[0].set_xlim(h[0], h[-1])
    axes[1].set_xlim(H - 0.15, H + 0.15); axes[1].set_ylim(0.8, 1.05)    # zoom : le nuage est très resserré
    fig.suptitle(rf"E1 (L2) — spectre estimé, fBm $H={H}$, $p\in[-5,5]$, $R={R}$")
    return fig


def resume(res, H, zeta_L1):
    """Médiane [écart interquartile] de zeta_hat(p) - pH, L2 puis L1."""
    p_choisis = [-5, -2, -1, -0.5, 0.5, 1, 2, 5]
    print(f"\n=== H = {H} : médiane [écart interquartile] de zeta_hat(p) - pH ===")
    print("p".ljust(12) + "".join(f"{v:>16g}" for v in p_choisis))
    lignes = [("leaders L2", res["zeta"])] + ([("leaders L1", zeta_L1)] if zeta_L1 is not None else [])
    for nom, z in lignes:
        ligne = nom.ljust(12)
        for v in p_choisis:
            b = int(np.argmin(np.abs(P - v)))
            q1, m, q3 = np.percentile(z[:, b] - v * H, [25, 50, 75])
            ligne += f"{m:>+8.3f} [{q3 - q1:.3f}]".rjust(16)
        print(ligne)


if __name__ == "__main__":
    refaire = "--refaire" in sys.argv
    DOSSIER_FIG.mkdir(exist_ok=True)
    for H in H_VALEURS:
        res = charger_ou_simuler(H, refaire)
        zeta_L1 = charger_L1(H)
        resume(res, H, zeta_L1)
        figures = [("log2S_sur_p", fig_log2S_sur_p(res, H)), ("zeta", fig_zeta(res, H)),
                   ("boites", fig_boites(res, H, zeta_L1)), ("spectre", fig_spectre(res, H))]
        for nom, fig in figures:
            fig.tight_layout()
            fig.savefig(DOSSIER_FIG / f"E1_L2_H{H}_{nom}.png", dpi=130)
            plt.close(fig)
    print(f"\nFigures écrites dans {DOSSIER_FIG}")
