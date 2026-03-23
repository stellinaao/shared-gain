import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
from pathlib import Path

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from scipy.stats import spearmanr
from sg.utils import spearmanr_vec

from sg.fitlvm_utils import eval_model

from copy import deepcopy

from sg import data


def plot_summary(
    family,
    model,
    potato=None,
    mode="offset",
    metric="spearman",
    save_fig=False,
    fname="",
):
    do_gain = mode == "gain" or mode == "affine"
    do_offset = mode == "offset" or mode == "affine"
    n_latents = (
        model.offset_mu.get_weights().shape[1]
        if mode == "offset"
        else model.gain_mu.get_weights().shape[1]
    )

    fig, axes = plt.subplots(
        ncols=1, nrows=n_latents, figsize=(3, 1.5 * n_latents), squeeze=False
    )

    for latent_idx, ax in enumerate(axes.flat):
        ax.plot(np.array(potato), "#9C9C9C", linewidth=2, label="Block")
        if do_gain:
            zgain = model.gain_mu.get_weights()[:, latent_idx]
            zweight = (
                model.readout_gain.get_weights()[latent_idx]
                if n_latents > 1
                else model.readout_gain.get_weights()
            )
            if np.mean(np.sign(zweight)) < 0:  # flip sign if both are negative
                zgain *= -1
                zweight *= -1

            ax.plot(zgain, "#9E2D2D", label="Gain Weights")

        if do_offset:
            zoffset = model.offset_mu.get_weights()[:, latent_idx]
            zoffweight = (
                model.readout_offset.get_weights()[latent_idx]
                if n_latents > 1
                else model.readout_offset.get_weights()
            )
            if np.mean(np.sign(zoffweight)) < 0:  # flip sign if both are negative
                zoffset *= -1
                zoffweight *= -1

            ax.plot(zoffset, "#463C8B", label="Offset Weights")

        ax.set_xlim((0, family.num_trials))
        ax.set_xlabel("Trials")
        ax.set_ylabel("Weights")
        ax.legend()

        title = ""
        if do_gain:
            if metric == "spearman":
                corr_g = spearmanr(potato, zgain).statistic
                title += rf"Gain $r$ = {corr_g:.3f}"
            elif metric == "pearsonr":
                corr_g = spearmanr(potato, zgain).statistic
                title += rf"Gain $\rho$ = {corr_g:.3f}"
        if do_offset:
            if metric == "spearman":
                corr_o = spearmanr(potato, zoffset).statistic
                title += (
                    rf" Offset $r$ = {corr_o:.3f}"
                    if do_gain
                    else rf"Offset $r$ = {corr_o:.3f}"
                )
            elif metric == "pearsonr":
                corr_g = spearmanr(potato, zoffset).statistic
                title += (
                    rf" Offset $\rho$ = {corr_o:.3f}"
                    if do_gain
                    else rf"Offset $rbo$ = {corr_o:.3f}"
                )

        ax.set_title(title)

    fig.tight_layout()
    if save_fig:
        from utils.paths import FIGURES_DIR

        save_dir = FIGURES_DIR / "latents"
        fpath = save_dir / fname
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fpath, dpi=300, bbox_inches="tight")


def get_num_latents(das, subj_idx, sess_idx, is_msess=True, ae=True, do_plot=False):
    das_ = das[subj_idx][sess_idx] if is_msess else das
    model_str = "affineae" if ae else "affine"

    r2s_tv = [
        np.mean(
            [
                torch.mean(das_[latent_idx]["tv"]["r2test"])
                for latent_idx in range(len(das_))
            ]
        )
    ]
    r2s_affine = [
        torch.mean(das_[latent_idx][model_str]["r2test"])
        for latent_idx in range(len(das_))
    ]

    d = np.diff(np.concatenate((r2s_tv, r2s_affine)), n=1)

    if do_plot:
        plot_r2_latents_diff(das, subj_idx, sess_idx, is_msess, ae)

    return np.where(d < 0.01)[0][0]


# latent decoding
def latent_decoding(family, y=None, model="affine", mode="both", cv=20):

    if model == "affine":
        model = family.mod_affine
    elif model == "single":
        if mode == "gain":
            model = family.mod_gain
        elif mode == "offset":
            model = family.mod_offset
        elif mode == "both":
            raise ValueError("model cannot be single yet request both gain and offset")
    else:
        raise ValueError("accepted values for model are 'affine' and 'single.'")

    do_gain = mode == "gain" or mode == "both"
    do_offset = mode == "offset" or mode == "both"

    X = []
    if do_gain:
        X.append(model.gain_mu.get_weights())
    if do_offset:
        X.append(model.offset_mu.get_weights())
    X = np.hstack(X)

    X_taskvar = np.array([family.response, family.rewarded, family.block_side]).T

    if y is None:
        y = family.strategy

    # check that dimensions match
    assert X.shape[0] == y.shape[0] == family.num_trials, (
        "First dimension of X or y is not equal to the number of trials."
    )
    assert X_taskvar.shape[0] == y.shape[0] == family.num_trials, (
        "First dimension of X_taskvar is not equal to the number of trials."
    )

    scores = {
        "latent": cross_val_score(
            SVC(random_state=1234), X, y, cv=cv, scoring="balanced_accuracy"
        ),
        "latent_shuffle": cross_val_score(
            SVC(random_state=1234),
            X,
            y.sample(frac=1),
            cv=cv,
            scoring="balanced_accuracy",
        ),
        "taskvar": cross_val_score(
            SVC(random_state=1234), X_taskvar, y, cv=cv, scoring="balanced_accuracy"
        ),
        "taskvar_shuffle": cross_val_score(
            SVC(random_state=1234),
            X_taskvar,
            y.sample(frac=1),
            cv=cv,
            scoring="balanced_accuracy",
        ),
    }

    return scores


# Unit R2s
def plot_r2_comp(
    results_a,
    results_b,
    label_a="",
    label_b="",
    title="",
    mode="unity",
    save=False,
    fpath=None,
):
    fig, ax = plt.subplots(figsize=(4, 4))

    if mode == "overlay":
        ax.plot(results_a["r2test"], "o", color="#666666", label=label_a)
        ax.plot(results_b["r2test"], "o", color="#E5A400", label=label_b)
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax.set_ylabel("$r^2$")
        ax.set_xlabel("Unit ID")
        # ax.set_title(f"R2, {label_a}: {torch.mean(np.delete(results_a['r2test'], np.where(results_a['r2test']==float('-inf')))):.3f},  {label_b}: {torch.mean(np.delete(results_b['r2test'], np.where(results_b['r2test']==float('-inf')))):.3f}")
        ax.set_ylim([-0.25, 1])
    elif mode == "unity":
        mn = torch.min(results_a["r2test"].min(), results_b["r2test"].min())
        mx = torch.max(results_a["r2test"].max(), results_b["r2test"].max())
        ax.scatter(results_a["r2test"], results_b["r2test"], color="#E5A400")
        ax.plot((mn, mx), (mn, mx), linestyle="--", color="#444444", label="unity")
        ax.set_xlabel(label_a)
        ax.set_ylabel(label_b)
        ax.set_xlim([mn - 0.05, mx + 0.05]), ax.set_ylim([mn - 0.05, mx + 0.05])
    else:
        raise ValueError("valid arguments for mode are 'unity' and 'overlay.'")

    if save:
        plt.savefig(fpath)

    ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_r2(results_a, label_a="", title="", save=True, fpath=None):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(results_a["r2test"], "o", color="#666666", label=label_a)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_ylabel("$r^2$")
    ax.set_xlabel("Unit ID")
    ax.set_title(
        f"R2, {label_a}: {torch.mean(np.delete(results_a['r2test'], np.where(results_a['r2test'] == float('-inf')))):.3f}"
    )
    ax.set_ylim([-1, 1])
    ax.legend()

    if save:
        plt.savefig(fpath)

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


# Session R2s
def plot_r2_latents_summary(das, subj_idx, ae=True):
    model_str = "affineae" if ae else "affine"

    r2s_tv = [
        [
            np.mean(
                [
                    torch.mean(das_sess[latent_idx]["tv"]["r2test"])
                    for latent_idx in range(len(das_sess))
                ]
            )
        ]
        for das_sess in das
    ]
    r2s_affine = [
        [
            torch.mean(das_sess[latent_idx][model_str]["r2test"])
            for latent_idx in range(len(das_sess))
        ]
        for das_sess in das
    ]

    r2s = np.concatenate((r2s_tv, r2s_affine), axis=1)
    r2s_avg = np.mean(r2s, axis=0)
    r2s_std = np.std(r2s, axis=0)

    fig, ax = plt.subplots()
    ax.plot(range(0, 9), r2s_avg)
    ax.fill_between(range(0, 9), r2s_avg - r2s_std, r2s_avg + r2s_std, alpha=0.4)
    ax.set_xlabel("Number of Latents")
    ax.set_ylabel("R2 of Affine Model")
    fig.suptitle(f"R2 Across Latents, {data.subject_ids[subj_idx]}")
    fig.tight_layout()
    fig.show()


def plot_r2_latents(das, subj_idx, sess_idx, is_msess=True, ae=True):
    das_ = das[subj_idx][sess_idx] if is_msess else das
    model_str = "affineae" if ae else "affine"

    r2s_tv = [
        np.mean(
            [
                torch.mean(das_[latent_idx]["tv"]["r2test"])
                for latent_idx in range(len(das_))
            ]
        )
    ]
    r2s_affine = [
        torch.mean(das_[latent_idx][model_str]["r2test"])
        for latent_idx in range(len(das_))
    ]

    fig, ax = plt.subplots()
    ax.plot(range(0, 9), np.concatenate((r2s_tv, r2s_affine)))
    ax.set_xlabel("Number of Latents")
    ax.set_ylabel("R2 of Affine Model")
    fig.suptitle(
        f"{data.subject_ids[subj_idx]}, {data.session_ids[subj_idx][sess_idx]}"
    )
    fig.tight_layout()
    fig.show()


def plot_r2_latents_diff(das, subj_idx, sess_idx, is_msess=True, ae=True, thresh=0.01):
    das_ = das[subj_idx][sess_idx] if is_msess else das
    model_str = "affineae" if ae else "affine"

    r2s_tv = [
        np.mean(
            [
                torch.mean(das_[latent_idx]["tv"]["r2test"])
                for latent_idx in range(len(das_))
            ]
        )
    ]
    r2s_affine = [
        torch.mean(das_[latent_idx][model_str]["r2test"])
        for latent_idx in range(len(das_))
    ]
    d = np.diff(np.concatenate((r2s_tv, r2s_affine)), n=1)
    first_min = np.where(d < thresh)[0][0]

    fig, ax = plt.subplots()
    ax.plot(range(0, 8), d)
    ax.scatter(first_min, d[first_min], marker="*", color="#FFD343", zorder=2)
    ax.axhline(y=0, color="#333333", linestyle="-")
    ax.axhline(y=0.01, color="#777777", linestyle="--")
    ax.set_xlabel("Number of Latents")
    ax.set_ylabel("R2 of Affine Model")
    fig.suptitle(
        f"{data.subject_ids[subj_idx]}, {data.session_ids[subj_idx][sess_idx]}"
    )
    fig.tight_layout()
    fig.show()


def plot_latents_all(das, num_latents=8):
    for latents in range(1, num_latents + 1):
        plot_latents(das[latents - 1], num_latents=latents, ae=True)


def plot_latents(das, num_latents, ae=True, mult=True):
    plt.figure()
    model = das["affineae"] if ae else das["affine"]
    weights = (
        model["model"].gain_mu.get_weights()
        if mult
        else model["model"].offset_mu.get_weights()
    )
    for ax in range(num_latents):
        plt.plot(weights[:, ax], alpha=0.5, label=f"Latent {ax + 1}")
    plt.title(f"Total # Latents: {num_latents}")
    plt.legend()


def plot_cweights_regions_hist(
    family, model, n_latents, mode="offset", do_save=False, fname=""
):
    """
    histogram, i don't really like this vis often
    """

    coupling = (
        model.readout_gain.weight.data[:].T
        if mode == "gain"
        else model.readout_offset.weight.data[:].T
    )

    regs = family.regions
    reg_keys = family.sample["reg_keys"]

    fig, axes = plt.subplots(
        ncols=n_latents, nrows=1, figsize=(2.5 * n_latents, 2.5), squeeze=False
    )

    for latent_idx, ax in enumerate(axes.flat):
        for region_idx, reg in enumerate(regs):
            idxs = np.where(reg_keys == region_idx)[0]
            # print(len(idxs))
            coupling_reg = coupling[idxs, latent_idx]
            # print(len(coupling_reg))
            ax.hist(
                coupling_reg,
                color=data.colors_region[reg],
                bins=np.linspace(-12, 12, 25),
                alpha=0.5,
                label=reg,
            )
        ax.legend()
        ax.set_xlabel("Coupling Weight")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Latent {latent_idx + 1}")
        fig.tight_layout()
        if do_save:
            from utils.paths import FIGURES_DIR

            save_dir = FIGURES_DIR / "cweights"
            fpath = save_dir / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(fpath, dpi=300, bbox_inches="tight")


def plot_cweight_regions(
    family,
    model,
    ax0,
    ax1,
    num_latents,
    mode="gain",
    is_msess=False,
    use_das=False,
    ae=True,
    abort=True,
    do_save=False,
    fname="",
    do_show=True,
):
    coupling = (
        model.readout_gain.weight.data[:].T
        if mode == "gain"
        else model.readout_offset.weight.data[:].T
    )

    regs = family.regions
    reg_keys = family.sample["reg_keys"]

    if abort and all(
        [
            all(
                coupling[np.where(reg_keys == i)[0], ax0]
                == coupling[np.where(reg_keys == i)[0], ax1]
            )
            for i in reg_keys
        ]
    ):
        print(f"Latent {ax0 + 1} and Latent {ax1 + 1} are equal, aborting")
        return

    fig, ax = plt.subplots(figsize=(3, 3))

    for i, reg in enumerate(regs):
        idxs = np.where(reg_keys == i)[0]
        idxs = idxs
        coupling_reg = coupling[idxs]
        ax.plot(
            coupling_reg[:, ax0],
            coupling_reg[:, ax1],
            data.markers_region[reg],
            color=data.colors_region[reg],
            label=reg,
        )
        # ax.axhline(torch.mean(coupling_reg), color=colors[i], linewidth=0.3, linestyle='--')

    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel(f"Latent {ax0 + 1}")
    ax.set_ylabel(f"Latent {ax1 + 1}")
    # fig.suptitle(f"{data.subject_ids[subj_idx]}, {data.session_ids[subj_idx][sess_idx]}; Total # Latents: {num_latents}")

    # ax.set_ylim([-0.5,0.5])
    # ax.set_xlim([-0.5,0.5])

    ax.legend()
    fig.tight_layout()
    if do_save:
        from utils.paths import FIGURES_DIR

        save_dir = FIGURES_DIR / "cweights"
        fpath = save_dir / fname
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fpath, dpi=300, bbox_inches="tight")
    if do_show:
        fig.show()


def plot_cweights_regions_sess(
    das,
    subj_idx,
    sess_idx,
    num_latents=8,
    is_mult=True,
    is_msess=False,
    ae=True,
    abort=True,
    do_save=False,
    do_show=True,
):
    for latents in range(num_latents):
        plot_cweights_regions_latent(
            das,
            latents,
            subj_idx,
            sess_idx,
            is_mult,
            is_msess,
            ae,
            abort,
            do_save,
            do_show,
        )


def plot_cweights_regions_latent(
    das,
    num_latents,
    subj_idx,
    sess_idx,
    is_mult=True,
    is_msess=False,
    use_das=False,
    ae=True,
    abort=True,
    do_save=False,
    do_show=True,
):
    for ax0 in range(num_latents):
        for ax1 in range(num_latents):
            if ax0 < ax1:
                plot_cweight_regions(
                    das,
                    ax0,
                    ax1,
                    num_latents,
                    subj_idx,
                    sess_idx,
                    is_mult,
                    is_msess,
                    use_das,
                    ae,
                    abort,
                    do_save,
                    do_show,
                )


def plot_cweights_mult(family):
    M = family.n_latents_mult
    if not family.no_mult:
        if M == 1:
            plot_cweights_regions_hist(
                family,
                family.mod_affine,
                n_latents=M,
                mode="gain",
            )
        else:
            for ax0 in range(M):
                for ax1 in range(M):
                    if ax1 > ax0:
                        _ = plot_cweight_regions(
                            family,
                            family.mod_affine,
                            ax0=ax0,
                            ax1=ax1,
                            num_latents=M,
                            mode="gain",
                            do_show=False,
                        )


def plot_latent_corr(model, mode="gain"):
    is_gain = mode == "gain" or mode == "affine"
    is_offset = mode == "offset" or mode == "affine"

    if not is_gain and not is_offset:
        raise ValueError("valid arguments for mode are gain, offset, and affine")

    X = []
    if is_gain:
        X.extend(model.gain_mu.get_weights().T)
    if is_offset:
        X.extend(model.offset_mu.get_weights().T)
    X = np.array(X)

    X = X - X.mean(axis=1, keepdims=True)
    X = X / X.std(axis=1, keepdims=True)

    corr = (X @ X.T) / X.shape[1]

    plt.figure()
    plt.imshow(corr, vmin=-1, vmax=1, cmap="PRGn")
    plt.xticks(np.arange(corr.shape[0]), np.arange(corr.shape[0]) + 1)
    plt.yticks(np.arange(corr.shape[0]), np.arange(corr.shape[0]) + 1)
    plt.xlabel("Latents")
    plt.ylabel("Latents")
    plt.colorbar(label=r"$r$")
    plt.tight_layout()
    plt.show()


def get_best_model(r2s, folder):
    """
    families = {}
    for folder in folders:
        families[folder] = get_best_model(folder)
    """

    m, a = np.where(r2s[folder] == np.max(r2s[folder]))
    m = m[0]
    a = a[0]

    if m == 0 and a == 0:
        return
        # NOT IMPLEMENTED FOR NOW
        return None
    else:
        with open(f"../vars/families/0312-lm/{folder}/family-m{m}a{a}.pkl", "rb") as f:
            family = pickle.load(f)
            family.eval()
    return family


def plot_strategy(families, folder):
    """
    folders = ["all", "ACC", "DMS", "M2", "DLS"]
    r2s = {}
    for f in folders:
        r2s[f] = get_latent_r(folder=f)
    """

    family_ = families[folder]

    if family_ is None:
        return

    mx_latents = max(family_.n_latents_mult, family_.n_latents_addt)
    if family_.no_mult or family_.no_addt:
        fig, axes = plt.subplots(
            nrows=1, ncols=mx_latents, figsize=(1.5 * mx_latents, 1.5), squeeze=False
        )
    else:
        fig, axes = plt.subplots(
            nrows=2, ncols=mx_latents, figsize=(1.5 * mx_latents, 2.5)
        )

    if not family_.no_mult:
        for m in range(family_.n_latents_mult):
            ax = axes[0, m]
            mf = family_.mod_affine.gain_mu.get_weights()[(family_.strategy == 0), 0]
            mb = family_.mod_affine.gain_mu.get_weights()[(family_.strategy == 1), 0]
            ax.hist(
                mf, bins=np.linspace(-3, 3, 17), alpha=0.5, color="#222FA9", label="MF"
            )
            ax.hist(
                mb, bins=np.linspace(-3, 3, 17), alpha=0.5, color="#E1A714", label="MB"
            )
            ax.set_title(f"M. Latent {m + 1}")
            ax.legend()
    if not family_.no_addt:
        idx = 1 if not family_.no_mult else 0
        for a in range(family_.n_latents_addt):
            ax = axes[idx, a]
            mf = family_.mod_affine.offset_mu.get_weights()[(family_.strategy == 0), 0]
            mb = family_.mod_affine.offset_mu.get_weights()[(family_.strategy == 1), 0]
            ax.hist(
                mf, bins=np.linspace(-3, 3, 17), alpha=0.5, color="#222FA9", label="MF"
            )
            ax.hist(
                mb, bins=np.linspace(-3, 3, 17), alpha=0.5, color="#E1A714", label="MB"
            )
            ax.set_title(f"A. Latent {a + 1}")
            ax.legend()
    fig.tight_layout()

    from utils.paths import FIGURES_DIR

    save_dir = FIGURES_DIR / "strategy_latent"
    fname = Path("0312-lm") / f"strategy_latent_{folder}.png"
    fpath = save_dir / fname
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fpath, dpi=300, bbox_inches="tight")


def plot_r2s_line(r2s, folder):
    r2 = r2s[folder]

    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(8, 2))

    for m in range(6):
        r2_slice = r2[m]
        ax = axes[0, m]
        ax.plot(r2_slice)
        ax.set_xticks(np.arange(6))
        ax.set_xlabel("# A. Latents")
        ax.set_ylabel("$r$")
        ax.set_title(f"{m} M. Latents")

    for a in range(6):
        r2_slice = r2[:, a]
        ax = axes[1, a]
        ax.plot(r2_slice)
        ax.set_xticks(np.arange(6))
        ax.set_xlabel("# M. Latents")
        ax.set_ylabel("$r$")
        ax.set_title(f"{a} A. Latents")
    fig.tight_layout()

    from utils.paths import FIGURES_DIR

    save_dir = FIGURES_DIR / "r2s_latents"
    fname = Path("0312-lm") / f"corr-line_{folder}.png"
    fpath = save_dir / fname
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fpath, dpi=300, bbox_inches="tight")


def everything(family, folder):
    if family is None:
        return
    if not family.no_mult:
        plot_summary(
            family,
            model=family.mod_affine,
            potato=family.strategy,
            mode="gain",
            save_fig=True,
            fname=Path("0312-lm") / folder / "m-latents.png",
        )
    if not family.no_addt:
        plot_summary(
            family,
            model=family.mod_affine,
            potato=family.strategy,
            mode="offset",
            save_fig=True,
            fname=Path("0312-lm") / folder / "a-latents.png",
        )

    M = family.n_latents_mult
    if not family.no_mult:
        if M == 1:
            plot_cweights_regions_hist(
                family,
                family.mod_affine,
                n_latents=M,
                mode="gain",
                do_save=True,
                fname=Path("0312-lm") / folder / "cweights-m0.png",
            )
        else:
            for ax0 in range(M):
                for ax1 in range(M):
                    if ax1 > ax0:
                        _ = plot_cweight_regions(
                            family,
                            family.mod_affine,
                            ax0=ax0,
                            ax1=ax1,
                            num_latents=M,
                            mode="gain",
                            do_save=True,
                            fname=Path("0312-lm")
                            / folder
                            / f"cweights-m{ax0}m{ax1}.png",
                        )
    A = family.n_latents_addt
    if not family.no_addt:
        if A == 1:
            plot_cweights_regions_hist(
                family,
                family.mod_affine,
                n_latents=A,
                mode="offset",
                do_save=True,
                fname=Path("0312-lm") / folder / "cweights-a0.png",
            )
        else:
            for ax0 in range(A):
                for ax1 in range(A):
                    if ax1 > ax0:
                        _ = plot_cweight_regions(
                            family,
                            family.mod_affine,
                            ax0=ax0,
                            ax1=ax1,
                            num_latents=A,
                            mode="offset",
                            do_save=True,
                            fname=Path("0312-lm")
                            / folder
                            / f"cweights-a{ax0}a{ax1}.png",
                        )

    model = family.mod_affine
    if family.no_mult:
        X = model.offset_mu.get_weights().T
    elif family.no_addt:
        X = model.gain_mu.get_weights().T
    else:
        X = np.concatenate(
            (model.gain_mu.get_weights().T, model.offset_mu.get_weights().T)
        )

    X = X - X.mean(axis=1, keepdims=True)
    X = X / X.std(axis=1, keepdims=True)

    corr = (X @ X.T) / X.shape[1]

    fig, ax = plt.subplots()
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="PRGn")
    ax.set_xticks(np.arange(corr.shape[0]), np.arange(corr.shape[0]) + 1)
    ax.set_yticks(np.arange(corr.shape[0]), np.arange(corr.shape[0]) + 1)
    ax.set_xlabel("Latents")
    ax.set_ylabel("Latents")
    ax.set_title(f"{M} mult latents and {A} addt latents")
    fig.colorbar(im, label=r"$r$")
    fig.tight_layout()
    fig.show()

    from utils.paths import FIGURES_DIR

    save_dir = FIGURES_DIR / "latent_corr"
    fname = Path("0312-lm") / folder / "latent_corr.png"
    fpath = save_dir / fname
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fpath, dpi=300, bbox_inches="tight")


def get_res(family, mode="taskvar"):
    robs = family.robs
    if mode == "taskvar":
        rhat = family.res_taskvar["rhat"]
    elif mode == "taskvar_latent":
        mod_offset = deepcopy(family.mod_offset)
        mod_offset.readout_offset.weight.data[:] = 0
        res_offset = eval_model(mod_offset, family.data_gd, family.test_dl.dataset)

        rhat = res_offset["rhat"]
    elif mode == "latent":
        rhat = family.res_offset["rhat"]
    else:
        raise ValueError("valid modes are taskvar, taskvar_latent, and latent")
    res = robs - rhat
    return res


def res_taskvar_corr(family, mode="taskvar", plot_res=True, plot_r2_dist=True):
    res = get_res(family, mode)

    block_side = family.block_side
    choice = family.response
    reward = family.rewarded

    # rob_block_corr = utils.spearmanr_vec(block_side, robs)
    # rob_choice_corr = utils.spearmanr_vec(choice, robs)
    # rob_reward_corr = utils.spearmanr_vec(reward, robs)
    res_block_corr = spearmanr_vec(block_side, res)
    res_choice_corr = spearmanr_vec(choice, res)
    res_reward_corr = spearmanr_vec(reward, res)

    if plot_res:
        plt.figure()
        plt.imshow(
            res.T,
            aspect="auto",
            interpolation="none",
            cmap="coolwarm",
            vmin=-0.2,
            vmax=0.2,
        )
        plt.colorbar(label="Residuals")
        plt.xlabel("Trials")
        plt.ylabel("Neurons")
        plt.tight_layout()
        plt.show()
    if plot_r2_dist:
        plt.figure()
        plt.hist(res_block_corr, bins=np.arange(-0.45, 0.45, 0.1))
        plt.xlabel(r"$\rho$")
        plt.ylabel("Frequency")
        plt.title("Block")
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.hist(res_choice_corr, bins=np.arange(-0.45, 0.45, 0.1))
        plt.xlabel(r"$\rho$")
        plt.ylabel("Frequency")
        plt.title("Choice")
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.hist(res_reward_corr, bins=np.arange(-0.45, 0.45, 0.1))
        plt.xlabel(r"$\rho$")
        plt.ylabel("Frequency")
        plt.title("Reward")
        plt.tight_layout()
        plt.show()
