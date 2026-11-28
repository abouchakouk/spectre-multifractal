"""E7 — Seuillage dur des d_{j,k} avant les leaders, branche p > 0 (sigma connu).

Question : le seuillage (sans connaître H) reconstruit-il la branche p > 0 de zeta et la branche
croissante du spectre, sur le fBm et sur la cascade ? À quel prix en nombre d'échelles ?
Comparaison de deux seuils : universel (Marc) et plus doux (3 sigma).

Modèle : Y_i = X(i/n) + sigma * xi_i, X = fBm (H = 0.3, 0.7) ou W-cascade (m = 0.5, b = 0.025).
Estimateur, pour chaque réalisation et chaque seuil :
  d_{j,k} (L2) ;  d~ = d * 1{|d| > lambda}, lambda_U = sigma sqrt(2 ln n)/sqrt(n), lambda_3 = 3 sigma/sqrt(n) ;
  L~_{j,k} = leaders de d~ (jusqu'à j = 14) ;  S~_d(j,p) = 2^{-j} sum_k L~^p  (p > 0 : leaders nuls -> 0) ;
  plage : j2 = plus grande échelle <= 13 où TOUS les leaders valides sont non nuls,
          j1 = min(6, j2 - 3) (>= 3) ; moins de 3 échelles -> non estimable (NaN, compté) ;
  zeta_hat(p) = - pente sur [j1, j2] - p/2 ;  spectre : forme paramétrique (branche croissante).
Référence « sans seuil » : leaders sur [6, 13] (E2 / E5 en L2), recalculés ici sur les mêmes réalisations.

Mêmes graines que E2 (fBm) et E5 (cascade) : mêmes trajectoires et mêmes bruits.

Usage :  python experiences/E7_seuillage_p_positif.py [--refaire]
Sorties : resultats/E7_*.npz, figures/E7_*.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spectre.bruit import ajouter_bruit
from spectre.legendre import legendre_parametrique
from spectre.modeles import fbm, wcascade
from spectre.multiechelle.leaders import leaders
from spectre.multiechelle.ondelettes import coefficients_ondelettes
from spectre.regression import estimer_zeta
from spectre.seuillage import niveau_seuil, plage_donnees, seuiller
from spectre.structure import log2_structure

# ---------------- paramètres ----------------
n = 2**15
R = 200
ONDELETTE = "db3"
SIGMAS = (0.0, 0.01, 0.1, 1.0)
ECHELLES = list(range(3, 15))
J1, J2 = 6, 13                       # plage de la référence sans seuil
METHODES = ("sans seuil", "universel", "3sigma")
COULEURS = {"sans seuil": "#8a8a8a", "universel": "#1baf7a", "3sigma": "#2a78d6"}
M_CASC, B_CASC = 0.5, 0.025
# modèle -> (graine identique à E2 / E5, grille p > 0, fonction zeta théorique)
MODELES = {
    "fBm_H0.3": dict(graine=20261124, H=0.3, P=np.round(np.arange(0.25, 5.0001, 0.25), 2)),
    "fBm_H0.7": dict(graine=20261124, H=0.7, P=np.round(np.arange(0.25, 5.0001, 0.25), 2)),
    "cascade": dict(graine=20261125, H=None, P=np.round(np.arange(0.25, 8.0001, 0.25), 2)),
}

RACINE = Path(__file__).resolve().parents[1]
DOSSIER_RES, DOSSIER_FIG = RACINE / "resultats", RACINE / "figures"


def zeta_vrai(nom, p):
    cfg = MODELES[nom]
    return p * cfg["H"] if cfg["H"] is not None else wcascade.zeta_theorique(p, M_CASC, B_CASC)


def generateur(nom, sigma):
    """Même suite d'appels aléatoires que E2 (fBm) ou E5 (cascade) : mêmes réalisations."""
    cfg = MODELES[nom]
    if cfg["H"] is not None:
        rng = np.random.default_rng([cfg["graine"], int(round(1000 * cfg["H"])), int(round(1e4 * sigma))])
        return lambda: ajouter_bruit(fbm.simuler_fbm(n, cfg["H"], rng), sigma, rng)
    rng = np.random.default_rng([cfg["graine"], int(round(1e4 * sigma))])
    return lambda: ajouter_bruit(wcascade.simuler_wcascade(n, M_CASC, B_CASC, rng, ONDELETTE), sigma, rng)


# ---------------- simulation ----------------
def simuler(nom, sigma):
    P = MODELES[nom]["P"]
    tirer = generateur(nom, sigma)
    nm = len(METHODES)
    log2S = np.full((R, nm, len(ECHELLES), P.size), np.nan)
    zeta = np.full((R, nm, P.size), np.nan)
    plage = np.full((R, nm, 2), -1, dtype=int)           # (j1, j2) ; -1 : non estimable
    for r in range(R):
        d = coefficients_ondelettes(tirer(), ONDELETTE, normalisation="L2")
        for q, meth in enumerate(METHODES):
            dd = d if meth == "sans seuil" else seuiller(d, niveau_seuil(sigma, n, meth))
            L = leaders(dd)
            js, l, _ = log2_structure(L, P, ECHELLES)
            log2S[r, q] = l
            pl = (J1, J2) if meth == "sans seuil" else plage_donnees(L)
            if pl is None:
                continue
            plage[r, q] = pl
            zeta[r, q] = estimer_zeta(js, l, *pl) - P / 2
    return dict(sigma=sigma, js=np.array(ECHELLES), p=P, log2S=log2S, zeta=zeta, plage=plage)


def charger_ou_simuler(nom, sigma, refaire):
    f = DOSSIER_RES / f"E7_{nom}_s{sigma}.npz"
    if f.exists() and not refaire:
        return dict(np.load(f))
    print(f"Simulation {nom}, sigma = {sigma} ({R} répétitions)...")
    res = simuler(nom, sigma)
    DOSSIER_RES.mkdir(exist_ok=True)
    np.savez(f, **res)
    return res


# ---------------- figures ----------------
def fig_log2S_sur_p(tous, nom):
    """Lignes : sigma ; colonnes : (seuil, p). Moyenne de log2 S~_d(j,p)/p sur les échelles
    où elle est finie, cible de pente -(zeta(p)/p + 1/2) (placée sur la courbe sigma = 0), et
    médiane de j2 (trait vertical)."""
    P = MODELES[nom]["P"]
    p_choisis = (1, 3)
    cols = [(q, v) for q in (1, 2) for v in p_choisis]
    js = tous[0]["js"]
    sel = (js >= J1) & (js <= J2)
    fig, axes = plt.subplots(len(tous), len(cols), figsize=(15, 3.1 * len(tous)), sharex=True)
    for k, (q, v) in enumerate(cols):
        b = int(np.argmin(np.abs(P - v)))
        pente = zeta_vrai(nom, v) / v + 0.5
        moy0 = (tous[0]["log2S"][:, 0, :, b] / v).mean(axis=0)
        c = np.mean(moy0[sel] + pente * js[sel])
        for i, res in enumerate(tous):
            ax = axes[i, k]
            y = res["log2S"][:, q, :, b] / v
            fini = np.all(np.isfinite(y), axis=0)
            ax.plot(js[fini], y[:, fini].mean(axis=0), "o-", color=COULEURS[METHODES[q]], ms=3, lw=1.5, label="moyenne")
            ax.plot(js, c - pente * js, "k--", lw=1, label="cible")
            j2 = res["plage"][:, q, 1]
            if np.any(j2 > 0):
                ax.axvline(np.median(j2[j2 > 0]), color="0.3", lw=1, ls=":", label=r"médiane de $j_2$")
            ax.grid(alpha=0.3)
            if i == 0: ax.set_title(rf"{METHODES[q]}, $p={v}$")
            if k == 0: ax.set_ylabel(rf"$\sigma={float(res['sigma']):g}$" + "\n" + r"$\log_2 \tilde S_d(j,p)/p$", fontsize=9)
            if i == len(tous) - 1: ax.set_xlabel(r"$j$")
    axes[1, 0].legend(fontsize=7)
    fig.suptitle(rf"E7 — {nom} : fonctions de structure des leaders seuillés ($\sigma$ connu, $n=2^{{15}}$, $R={R}$)")
    return fig


def fig_boites(tous, nom):
    """zeta_hat(p) - zeta(p) selon sigma, pour les trois méthodes (estimations non NaN)."""
    P = MODELES[nom]["P"]
    p_choisis = (1, 2, 3, 5) if MODELES[nom]["H"] is not None else (1, 2, 4, 6)
    fig, axes = plt.subplots(1, len(p_choisis), figsize=(16, 4.2), sharey=True)
    x = np.arange(len(tous))
    for a, v in enumerate(p_choisis):
        ax = axes[a]
        b = int(np.argmin(np.abs(P - v)))
        for q, dx in zip(range(3), (-0.25, 0.0, 0.25)):
            err = [res["zeta"][:, q, b] - zeta_vrai(nom, v) for res in tous]
            err = [e[np.isfinite(e)] if np.any(np.isfinite(e)) else np.array([np.nan]) for e in err]
            bp = ax.boxplot(err, positions=x + dx, widths=0.22, patch_artist=True,
                            flierprops=dict(markersize=2, alpha=0.4))
            for box in bp["boxes"]:
                box.set_facecolor(COULEURS[METHODES[q]]); box.set_alpha(0.6)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x, [f"{float(r['sigma']):g}" for r in tous])
        ax.set_xlabel(r"$\sigma$"); ax.set_title(rf"$p={v}$"); ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel(r"$\hat\zeta(p)-\zeta(p)$")
    axes[0].legend([plt.Rectangle((0, 0), 1, 1, fc=COULEURS[m], alpha=0.6) for m in METHODES], METHODES, fontsize=8)
    fig.suptitle(rf"E7 — {nom} : erreur sur $\zeta$, $p>0$ (estimations non estimables exclues ; voir figure des plages)")
    return fig


def fig_plages(tous, nom):
    """Distribution de j2 et proportion de réalisations non estimables, par sigma et par seuil."""
    fig, axes = plt.subplots(1, len(tous), figsize=(4 * len(tous), 3.6), sharey=True)
    bins = np.arange(2.5, 14.5, 1)
    for ax, res in zip(axes, tous):
        for q in (1, 2):
            j2 = res["plage"][:, q, 1]
            ne = np.mean(j2 < 0)
            ax.hist(j2[j2 > 0], bins=bins, color=COULEURS[METHODES[q]], alpha=0.6,
                    label=f"{METHODES[q]} (non estimable : {100 * ne:.0f} %)")
        ax.set_title(rf"$\sigma={float(res['sigma']):g}$"); ax.set_xlabel(r"$j_2$"); ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("nombre de réalisations")
    fig.suptitle(rf"E7 — {nom} : plage choisie par les données ($j_2$ = dernière échelle sans leader nul), $R={R}$")
    return fig


def fig_spectre(tous, nom):
    """Branche croissante, forme paramétrique (zeta_hat'(p), 1 + p zeta_hat'(p) - zeta_hat(p)), p > 0.
    Lignes : méthodes ; colonnes : sigma. Points noirs : médiane de zeta_hat ; tirets : théorie."""
    P = MODELES[nom]["P"]
    H = MODELES[nom]["H"]
    fig, axes = plt.subplots(3, len(tous), figsize=(4 * len(tous), 9), sharex=True, sharey=True)
    for i, res in enumerate(tous):
        for q in range(3):
            ax = axes[q, i]
            Z = res["zeta"][:, q]
            ok = np.all(np.isfinite(Z), axis=1)
            for zr in Z[ok][::4]:
                hp, Dp = legendre_parametrique(P, zr)
                ax.plot(hp, Dp, ".", color=COULEURS[METHODES[q]], ms=2, alpha=0.15)
            if ok.any():
                hp, Dp = legendre_parametrique(P, np.median(Z[ok], axis=0))
                ax.plot(hp, Dp, "k.", ms=3)
            if H is None:
                hh = np.linspace(M_CASC - np.sqrt(2 * B_CASC), M_CASC, 100)
                ax.plot(hh, wcascade.spectre_theorique(hh, M_CASC, B_CASC), "k--", lw=1)
            else:
                ax.plot([H], [1], "k*", ms=8)
            ax.grid(alpha=0.3)
            if q == 0: ax.set_title(rf"$\sigma={float(res['sigma']):g}$")
            if i == 0: ax.set_ylabel(f"{METHODES[q]}\n" + r"$\hat D$")
            if q == 2: ax.set_xlabel(r"$h$")
    ref = H if H is not None else M_CASC
    axes[0, 0].set_xlim(ref - 0.9, ref + 0.3); axes[0, 0].set_ylim(-0.2, 1.1)
    fig.suptitle(rf"E7 — {nom} : branche croissante du spectre ($p>0$), forme paramétrique, $R={R}$")
    return fig


def resume(tous, nom):
    P = MODELES[nom]["P"]
    p_choisis = (1, 2, 5) if MODELES[nom]["H"] is not None else (1, 2, 6)
    print(f"\n=== E7 {nom} : médiane [écart interquartile] de zeta_hat(p) - zeta(p) ; j2 médian ; % non estimable ===")
    print("sigma  méthode".ljust(20) + "".join(f"{'p=' + str(v):>16}" for v in p_choisis) + "     j2   non est.")
    for res in tous:
        for q, meth in enumerate(METHODES):
            ligne = f"{float(res['sigma']):<6g} {meth}".ljust(20)
            for v in p_choisis:
                b = int(np.argmin(np.abs(P - v)))
                e = res["zeta"][:, q, b] - zeta_vrai(nom, v)
                e = e[np.isfinite(e)]
                if e.size:
                    q1, med, q3 = np.percentile(e, [25, 50, 75])
                    ligne += f"{med:>+8.3f} [{q3 - q1:.3f}]".rjust(16)
                else:
                    ligne += "—".rjust(16)
            j2 = res["plage"][:, q, 1]
            ligne += f"{np.median(j2[j2 > 0]) if np.any(j2 > 0) else float('nan'):>7.1f}{100 * np.mean(j2 < 0):>9.0f} %"
            print(ligne)


if __name__ == "__main__":
    refaire = "--refaire" in sys.argv
    DOSSIER_FIG.mkdir(exist_ok=True)
    for nom in MODELES:
        tous = [charger_ou_simuler(nom, s, refaire) for s in SIGMAS]
        resume(tous, nom)
        for fnom, f in [("log2S_sur_p", fig_log2S_sur_p), ("boites", fig_boites),
                        ("plages", fig_plages), ("spectre", fig_spectre)]:
            fig = f(tous, nom)
            fig.tight_layout()
            fig.savefig(DOSSIER_FIG / f"E7_{nom}_{fnom}.png", dpi=120)
            plt.close(fig)
    print(f"\nFigures écrites dans {DOSSIER_FIG}")
