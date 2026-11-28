"""E8 — Seuillage dur des d_{j,k} avant les leaders : spectre complet (p < 0 et p > 0), sigma connu.

Question : sur la plage où TOUS les leaders seuillés sont non nuls, l'estimateur reconstruit-il
les deux branches de zeta et du spectre, sur le fBm et sur la cascade, sans connaître H ?

Fait utilisé (lemme, voir tests/test_seuillage.py) : leaders(d * 1{|d| > lambda}) = L * 1{L > lambda}.
Sur une échelle où tous les leaders sont non nuls, L~ = L > lambda : L~^p < lambda^p pour p < 0,
tous les moments négatifs sont finis.

Modèle : Y_i = X(i/n) + sigma * xi_i, X = fBm (H = 0.3, 0.7) ou W-cascade (m = 0.5, b = 0.025).
Estimateur, pour chaque réalisation et chaque seuil (universel : sigma sqrt(2 ln n)/sqrt(n) ;
3sigma : 3 sigma/sqrt(n)) :
  L~ = leaders des d seuillés ; plage : j2 = plus grande échelle <= 13 où tous les leaders valides
  sont non nuls, j1 = min(6, j2 - 3) (>= 3) ; moins de 3 échelles -> non estimable (NaN, compté) ;
  zeta_hat(p) = - pente de log2 S~_d sur [j1, j2] - p/2, pour p dans toute la grille ;
  D_hat(h) = min_p (1 + p h - zeta_hat(p)) et forme paramétrique.
Référence « sans seuil » : leaders sur [6, 13] (E2 / E5 en L2), mêmes réalisations.
Plages : identiques à E7 (mêmes réalisations, même règle) ; voir figures/E7_*_plages.png.

Usage :  python experiences/E8_seuillage_spectre_complet.py [--refaire]
Sorties : resultats/E8_*.npz, figures/E8_*.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spectre.bruit import ajouter_bruit
from spectre.legendre import legendre, legendre_parametrique
from spectre.modeles import fbm, wcascade
from spectre.multiechelle.leaders import leaders
from spectre.multiechelle.ondelettes import coefficients_ondelettes
from spectre.regression import estimer_zeta
from spectre.seuillage import fraction_non_nuls, niveau_seuil, plage_donnees, seuiller
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
MODELES = {
    "fBm_H0.3": dict(graine=20261124, H=0.3, P=np.round(np.arange(-5, 5.0001, 0.25), 2)),
    "fBm_H0.7": dict(graine=20261124, H=0.7, P=np.round(np.arange(-5, 5.0001, 0.25), 2)),
    "cascade": dict(graine=20261125, H=None, P=np.round(np.arange(-8, 8.0001, 0.25), 2)),
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
    q = np.zeros((R, nm, len(ECHELLES)))                  # proportion de leaders non nuls par échelle
    zeta = np.full((R, nm, P.size), np.nan)
    plage = np.full((R, nm, 2), -1, dtype=int)            # (j1, j2) ; -1 : non estimable
    for r in range(R):
        d = coefficients_ondelettes(tirer(), ONDELETTE, normalisation="L2")
        for m, meth in enumerate(METHODES):
            dd = d if meth == "sans seuil" else seuiller(d, niveau_seuil(sigma, n, meth))
            L = leaders(dd)
            js, l, _ = log2_structure(L, P, ECHELLES)
            log2S[r, m], q[r, m] = l, fraction_non_nuls(L, ECHELLES)
            pl = (J1, J2) if meth == "sans seuil" else plage_donnees(L)
            if pl is None:
                continue
            plage[r, m] = pl
            zeta[r, m] = estimer_zeta(js, l, *pl) - P / 2
    return dict(sigma=sigma, js=np.array(ECHELLES), p=P, log2S=log2S, q=q, zeta=zeta, plage=plage)


def charger_ou_simuler(nom, sigma, refaire):
    f = DOSSIER_RES / f"E8_{nom}_s{sigma}.npz"
    if f.exists() and not refaire:
        return dict(np.load(f))
    print(f"Simulation {nom}, sigma = {sigma} ({R} répétitions)...")
    res = simuler(nom, sigma)
    DOSSIER_RES.mkdir(exist_ok=True)
    np.savez(f, **res)
    return res


# ---------------- figures ----------------
def fig_log2S_sur_p(tous, nom):
    """Lignes : sigma ; colonnes : (seuil, p). Moyenne de log2 S~_d(j,p)/p, calculée à chaque échelle
    sur les réalisations où TOUS les leaders y sont non nuls (échelle tracée si c'est le cas pour au
    moins la moitié des réalisations) ; cible de pente -(zeta(p)/p + 1/2) placée sur sigma = 0."""
    P = MODELES[nom]["P"]
    p_choisis = (-3, 3) if MODELES[nom]["H"] is not None else (-6, 6)
    cols = [(m, v) for m in (1, 2) for v in p_choisis]
    js = tous[0]["js"]
    sel = (js >= J1) & (js <= J2)
    fig, axes = plt.subplots(len(tous), len(cols), figsize=(15, 3.1 * len(tous)), sharex=True)
    for k, (m, v) in enumerate(cols):
        b = int(np.argmin(np.abs(P - v)))
        pente = zeta_vrai(nom, v) / v + 0.5
        moy0 = (tous[0]["log2S"][:, 0, :, b] / v).mean(axis=0)
        c = np.mean(moy0[sel] + pente * js[sel])
        for i, res in enumerate(tous):
            ax = axes[i, k]
            y = res["log2S"][:, m, :, b] / v
            plein = res["q"][:, m, :] == 1.0
            garde = plein.mean(axis=0) >= 0.5
            moy = np.array([y[plein[:, a], a].mean() if garde[a] else np.nan for a in range(len(js))])
            ax.plot(js, moy, "o-", color=COULEURS[METHODES[m]], ms=3, lw=1.5, label="moyenne (leaders tous non nuls)")
            ax.plot(js, c - pente * js, "k--", lw=1, label="cible")
            j2 = res["plage"][:, m, 1]
            if np.any(j2 > 0):
                ax.axvline(np.median(j2[j2 > 0]), color="0.3", lw=1, ls=":", label=r"médiane de $j_2$")
            ax.grid(alpha=0.3)
            if i == 0: ax.set_title(rf"{METHODES[m]}, $p={v}$")
            if k == 0: ax.set_ylabel(rf"$\sigma={float(res['sigma']):g}$" + "\n" + r"$\log_2 \tilde S_d(j,p)/p$", fontsize=9)
            if i == len(tous) - 1: ax.set_xlabel(r"$j$")
    axes[1, 0].legend(fontsize=7)
    fig.suptitle(rf"E8 — {nom} : fonctions de structure des leaders seuillés, $p<0$ et $p>0$ ($\sigma$ connu, $n=2^{{15}}$, $R={R}$)")
    return fig


def fig_boites(tous, nom):
    """zeta_hat(p) - zeta(p) selon sigma, trois méthodes (réalisations non estimables exclues)."""
    P = MODELES[nom]["P"]
    p_choisis = (-3, -1, 1, 3) if MODELES[nom]["H"] is not None else (-6, -2, 2, 6)
    fig, axes = plt.subplots(1, len(p_choisis), figsize=(16, 4.2), sharey=True)
    x = np.arange(len(tous))
    for a, v in enumerate(p_choisis):
        ax = axes[a]
        b = int(np.argmin(np.abs(P - v)))
        for m, dx in zip(range(3), (-0.25, 0.0, 0.25)):
            err = [res["zeta"][:, m, b] - zeta_vrai(nom, v) for res in tous]
            err = [e[np.isfinite(e)] if np.any(np.isfinite(e)) else np.array([np.nan]) for e in err]
            bp = ax.boxplot(err, positions=x + dx, widths=0.22, patch_artist=True,
                            flierprops=dict(markersize=2, alpha=0.4))
            for box in bp["boxes"]:
                box.set_facecolor(COULEURS[METHODES[m]]); box.set_alpha(0.6)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x, [f"{float(r['sigma']):g}" for r in tous])
        ax.set_xlabel(r"$\sigma$"); ax.set_title(rf"$p={v}$"); ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel(r"$\hat\zeta(p)-\zeta(p)$")
    axes[0].legend([plt.Rectangle((0, 0), 1, 1, fc=COULEURS[k], alpha=0.6) for k in METHODES], METHODES, fontsize=8)
    fig.suptitle(rf"E8 — {nom} : erreur sur $\zeta$, $p<0$ et $p>0$ (non estimables exclues ; plages : figures E7)")
    return fig


def _theorie(nom, h):
    H = MODELES[nom]["H"]
    if H is None:
        return wcascade.spectre_theorique(h, M_CASC, B_CASC)
    P = MODELES[nom]["P"]
    return 1 - P.max() * np.abs(h - H)          # monofractal vu sur la grille finie : tente


def fig_spectre_min(tous, nom):
    """D_hat(h) = min_p (1 + p h - zeta_hat(p)) sur toute la grille. Lignes : méthodes ; colonnes : sigma."""
    P = MODELES[nom]["P"]
    ref = MODELES[nom]["H"] if MODELES[nom]["H"] is not None else M_CASC
    h = np.linspace(ref - 1.2, ref + 0.6, 361)
    fig, axes = plt.subplots(3, len(tous), figsize=(4 * len(tous), 9), sharex=True, sharey=True)
    for i, res in enumerate(tous):
        for m in range(3):
            ax = axes[m, i]
            Z = res["zeta"][:, m]
            ok = np.all(np.isfinite(Z), axis=1)
            if ok.any():
                D = np.array([legendre(P, zr, h) for zr in Z[ok]])
                lo, med, hi = np.percentile(D, [5, 50, 95], axis=0)
                ax.fill_between(h, lo, hi, color=COULEURS[METHODES[m]], alpha=0.25, lw=0)
                ax.plot(h, med, color=COULEURS[METHODES[m]], lw=2)
            ax.plot(h, _theorie(nom, h), "k--", lw=1)
            ax.grid(alpha=0.3)
            ax.text(0.02, 0.04, f"estimables : {100 * ok.mean():.0f} %", transform=ax.transAxes, fontsize=7)
            if m == 0: ax.set_title(rf"$\sigma={float(res['sigma']):g}$")
            if i == 0: ax.set_ylabel(f"{METHODES[m]}\n" + r"$\hat D(h)$")
            if m == 2: ax.set_xlabel(r"$h$")
    axes[0, 0].set_ylim(-1, 1.1); axes[0, 0].set_xlim(h[0], h[-1])
    fig.suptitle(rf"E8 — {nom} : spectre $\min_p$, $p\in[{P.min():g},{P.max():g}]$, médiane et bande 5–95 % (tirets : théorie sur la grille)")
    return fig


def fig_spectre_param(tous, nom):
    """Forme paramétrique (zeta_hat'(p), 1 + p zeta_hat'(p) - zeta_hat(p)). Lignes : méthodes ; colonnes : sigma."""
    P = MODELES[nom]["P"]
    H = MODELES[nom]["H"]
    ref = H if H is not None else M_CASC
    fig, axes = plt.subplots(3, len(tous), figsize=(4 * len(tous), 9), sharex=True, sharey=True)
    for i, res in enumerate(tous):
        for m in range(3):
            ax = axes[m, i]
            Z = res["zeta"][:, m]
            ok = np.all(np.isfinite(Z), axis=1)
            for zr in Z[ok][::4]:
                hp, Dp = legendre_parametrique(P, zr)
                ax.plot(hp, Dp, ".", color=COULEURS[METHODES[m]], ms=2, alpha=0.12)
            if ok.any():
                hp, Dp = legendre_parametrique(P, np.median(Z[ok], axis=0))
                ax.plot(hp[P < 0], Dp[P < 0], "v", color="k", ms=3, label=r"médiane, $p<0$")
                ax.plot(hp[P > 0], Dp[P > 0], "^", color="0.5", ms=3, label=r"médiane, $p>0$")
            if H is None:
                hh = np.linspace(M_CASC - np.sqrt(2 * B_CASC), M_CASC + np.sqrt(2 * B_CASC), 200)
                ax.plot(hh, wcascade.spectre_theorique(hh, M_CASC, B_CASC), "k--", lw=1)
            else:
                ax.plot([H], [1], "k*", ms=8)
            ax.grid(alpha=0.3)
            if m == 0: ax.set_title(rf"$\sigma={float(res['sigma']):g}$")
            if i == 0: ax.set_ylabel(f"{METHODES[m]}\n" + r"$\hat D$")
            if m == 2: ax.set_xlabel(r"$h$")
    axes[0, 0].set_xlim(ref - 0.9, ref + 0.6); axes[0, 0].set_ylim(-0.2, 1.1)
    axes[2, 0].legend(loc="lower left", fontsize=7)
    fig.suptitle(rf"E8 — {nom} : spectre, forme paramétrique, $p\in[{P.min():g},{P.max():g}]$, $R={R}$")
    return fig


def resume(tous, nom):
    P = MODELES[nom]["P"]
    p_choisis = (-5, -2, -1, 1, 2, 5) if MODELES[nom]["H"] is not None else (-6, -2, -1, 1, 2, 6)
    print(f"\n=== E8 {nom} : médiane [écart interquartile] de zeta_hat(p) - zeta(p) ; % non estimable ===")
    print("sigma  méthode".ljust(20) + "".join(f"{'p=' + str(v):>16}" for v in p_choisis) + "   non est.")
    for res in tous:
        for m, meth in enumerate(METHODES):
            ligne = f"{float(res['sigma']):<6g} {meth}".ljust(20)
            for v in p_choisis:
                b = int(np.argmin(np.abs(P - v)))
                e = res["zeta"][:, m, b] - zeta_vrai(nom, v)
                e = e[np.isfinite(e)]
                if e.size:
                    q1, med, q3 = np.percentile(e, [25, 50, 75])
                    ligne += f"{med:>+8.3f} [{q3 - q1:.3f}]".rjust(16)
                else:
                    ligne += "—".rjust(16)
            ligne += f"{100 * np.mean(res['plage'][:, m, 1] < 0):>9.0f} %"
            print(ligne)


if __name__ == "__main__":
    refaire = "--refaire" in sys.argv
    DOSSIER_FIG.mkdir(exist_ok=True)
    for nom in MODELES:
        tous = [charger_ou_simuler(nom, s, refaire) for s in SIGMAS]
        resume(tous, nom)
        for fnom, f in [("log2S_sur_p", fig_log2S_sur_p), ("boites", fig_boites),
                        ("spectre_min", fig_spectre_min), ("spectre_param", fig_spectre_param)]:
            fig = f(tous, nom)
            fig.tight_layout()
            fig.savefig(DOSSIER_FIG / f"E8_{nom}_{fnom}.png", dpi=120)
            plt.close(fig)
    print(f"\nFigures écrites dans {DOSSIER_FIG}")
