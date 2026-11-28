"""E4 — Choix de la grille en p (question de Marc), sur la W-cascade log-normale.

Leaders uniquement (seule quantité valable pour p < 0), sigma in {0, 0.01}, régression [6, 13].
Grilles comparées (toutes contiennent p = 0) :
  G1 [-5,5] linéaire, 41 pts (référence E1–E3)   G2 [-5,5] linéaire, 11 pts
  G3 [-2,2] linéaire, 41 pts                       G4 [-8,8] linéaire, 41 pts
  G5 [-5,5] resserrée près de 0 : p = 5 sgn(u) u^2, u in [-1,1], 41 pts
Mesures (par réalisation), formes min_p et paramétrique :
  err_sup  = sup |D_hat - D| (min_p : sur la zone D >= 1/2 ; paramétrique : sur les points de la grille)
  sommet   = h du maximum de D_hat, moins m
  largeur  = longueur de {D_hat >= 1/2} moins 2 sqrt(b)  (min_p seulement)
Oracle : mêmes mesures avec le vrai zeta sur la même grille (erreur due à la grille seule).

Usage :  python experiences/E4_grilles_p.py [--refaire]
Sorties : resultats/E4_s*.npz, figures/E4_*.png
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
M, B = 0.5, 0.025
SIGMAS = (0.0, 0.01)
n = 2**15
R = 200
ONDELETTE = "db3"
J1, J2 = 6, 13
GRAINE = 20261126
u = np.linspace(-1, 1, 41)
GRILLES = {
    "G1 [-5,5] ×41": np.linspace(-5, 5, 41),
    "G2 [-5,5] ×11": np.linspace(-5, 5, 11),
    "G3 [-2,2] ×41": np.linspace(-2, 2, 41),
    "G4 [-8,8] ×41": np.linspace(-8, 8, 41),
    "G5 [-5,5] ×41 resserrée": 5 * np.sign(u) * u**2,
}
P_TOUS = np.unique(np.round(np.concatenate(list(GRILLES.values())), 10))
H_FIN = np.linspace(M - 0.6, M + 0.6, 1201)
ZONE = np.abs(H_FIN - M) <= np.sqrt(B)                 # D(h) >= 1/2
MESURES = ("err_sup_min", "err_sup_par", "sommet_min", "sommet_par", "largeur_min")
TITRES = {"err_sup_min": r"$\sup|\hat D-D|$, forme $\min_p$", "err_sup_par": r"$\sup|\hat D-D|$, paramétrique",
          "sommet_min": r"sommet $-\,m$, forme $\min_p$", "sommet_par": r"sommet $-\,m$, paramétrique",
          "largeur_min": r"largeur à mi-hauteur $-\,2\sqrt{b}$"}

RACINE = Path(__file__).resolve().parents[1]
DOSSIER_RES, DOSSIER_FIG = RACINE / "resultats", RACINE / "figures"


def indices(grille):
    return np.array([int(np.argmin(np.abs(P_TOUS - v))) for v in grille])


def mesurer(grille, zeta):
    """Les cinq mesures pour un zeta donné sur une grille donnée."""
    D = legendre(grille, zeta, H_FIN)
    Dt = spectre_theorique(H_FIN, M, B)
    sommet = H_FIN[D >= D.max() - 1e-9].mean()            # milieu du plateau éventuel
    hp, Dp = legendre_parametrique(grille, zeta)
    return dict(err_sup_min=np.max(np.abs(D - Dt)[ZONE]),
                err_sup_par=np.max(np.abs(Dp - spectre_theorique(hp, M, B))),
                sommet_min=sommet - M,
                sommet_par=hp[np.argmax(Dp)] - M,
                largeur_min=(D >= 0.5).sum() * (H_FIN[1] - H_FIN[0]) - 2 * np.sqrt(B))


# ---------------- simulation ----------------
def simuler(sigma):
    rng = np.random.default_rng([GRAINE, int(round(1e4 * sigma))])
    zeta = np.empty((R, P_TOUS.size))
    for r in range(R):
        y = ajouter_bruit(simuler_wcascade(n, M, B, rng, ONDELETTE), sigma, rng)
        js, l, _ = log2_structure(leaders(coefficients_ondelettes(y, ONDELETTE, normalisation="L1")), P_TOUS, range(3, 15))
        zeta[r] = estimer_zeta(js, l, J1, J2)
    return dict(sigma=sigma, p=P_TOUS, zeta=zeta)


def charger_ou_simuler(sigma, refaire):
    f = DOSSIER_RES / f"E4_s{sigma}.npz"
    if f.exists() and not refaire:
        return dict(np.load(f))
    print(f"Simulation sigma = {sigma} ({R} répétitions)...")
    res = simuler(sigma)
    DOSSIER_RES.mkdir(exist_ok=True)
    np.savez(f, **res)
    return res


def tableau(res):
    """mes[nom_grille][mesure] : tableau (R,) ; oracle[nom_grille][mesure] : nombre."""
    mes, oracle = {}, {}
    for nom, g in GRILLES.items():
        idx = indices(g)
        lignes = [mesurer(P_TOUS[idx], z[idx]) for z in res["zeta"]]
        mes[nom] = {k: np.array([l[k] for l in lignes]) for k in MESURES}
        oracle[nom] = mesurer(P_TOUS[idx], zeta_theorique(P_TOUS[idx], M, B))
    return mes, oracle


# ---------------- figures ----------------
def fig_mesures(resultats):
    noms = list(GRILLES)
    fig, axes = plt.subplots(len(resultats), len(MESURES), figsize=(19, 4.2 * len(resultats)), squeeze=False)
    for i, (sigma, (mes, oracle)) in enumerate(resultats.items()):
        for k, cle in enumerate(MESURES):
            ax = axes[i, k]
            ax.boxplot([mes[g][cle] for g in noms], flierprops=dict(markersize=1.5, alpha=0.4))
            ax.plot(range(1, len(noms) + 1), [oracle[g][cle] for g in noms], "rx", ms=8, mew=2, label="oracle (vrai $\\zeta$)")
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xticks(range(1, len(noms) + 1), [g.split()[0] for g in noms])
            ax.grid(alpha=0.3, axis="y")
            if i == 0: ax.set_title(TITRES[cle], fontsize=10)
            if k == 0: ax.set_ylabel(rf"$\sigma={sigma:g}$")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("E4 — grilles en $p$ : " + " ; ".join(noms) + rf" — W-cascade $m={M}$, $b={B}$, leaders, $R={R}$", fontsize=10)
    return fig


def fig_spectres(res, sigma):
    noms = list(GRILLES)
    fig, axes = plt.subplots(1, len(noms), figsize=(19, 4), sharey=True)
    Dt = spectre_theorique(H_FIN, M, B)
    for ax, nom in zip(axes, noms):
        idx = indices(GRILLES[nom])
        g = P_TOUS[idx]
        D = np.array([legendre(g, z[idx], H_FIN) for z in res["zeta"]])
        lo, med, hi = np.percentile(D, [5, 50, 95], axis=0)
        ax.fill_between(H_FIN, lo, hi, color="#1baf7a", alpha=0.25, lw=0)
        ax.plot(H_FIN, med, color="#1baf7a", lw=2, label=r"$\min_p$ (médiane)")
        hp, Dp = legendre_parametrique(g, np.median(res["zeta"][:, idx], axis=0))
        ax.plot(hp, Dp, "k.", ms=4, label="paramétrique (médiane)")
        ax.plot(H_FIN, Dt, "k--", lw=1, label="$D$ théorique")
        ax.plot(H_FIN, legendre(g, zeta_theorique(g, M, B), H_FIN), ":", color="0.4", label="oracle")
        ax.axhline(0.5, color="0.7", lw=0.6)
        ax.set_title(nom, fontsize=10); ax.set_xlabel("$h$"); ax.grid(alpha=0.3)
    axes[0].set_ylim(-0.5, 1.1); axes[0].set_ylabel(r"$\hat D(h)$"); axes[0].legend(fontsize=7)
    fig.suptitle(rf"E4 — spectres estimés selon la grille, $\sigma={sigma:g}$ (bande 5–95 %)")
    return fig


def resume(resultats):
    for sigma, (mes, oracle) in resultats.items():
        print(f"\n=== E4, sigma = {sigma:g} : médiane [écart interquartile] (oracle) ===")
        print("grille".ljust(26) + "".join(f"{c:>24}" for c in MESURES))
        for g in GRILLES:
            ligne = g.ljust(26)
            for c in MESURES:
                q1, m, q3 = np.percentile(mes[g][c], [25, 50, 75])
                ligne += f"{m:+.3f} [{q3 - q1:.3f}] ({oracle[g][c]:+.3f})".rjust(24)
            print(ligne)


if __name__ == "__main__":
    refaire = "--refaire" in sys.argv
    DOSSIER_FIG.mkdir(exist_ok=True)
    donnees = {s: charger_ou_simuler(s, refaire) for s in SIGMAS}
    resultats = {s: tableau(d) for s, d in donnees.items()}
    resume(resultats)
    fig = fig_mesures(resultats); fig.tight_layout(); fig.savefig(DOSSIER_FIG / "E4_mesures.png", dpi=120); plt.close(fig)
    for s, d in donnees.items():
        fig = fig_spectres(d, s); fig.tight_layout(); fig.savefig(DOSSIER_FIG / f"E4_spectres_s{s}.png", dpi=120); plt.close(fig)
    print(f"\nFigures écrites dans {DOSSIER_FIG}")
