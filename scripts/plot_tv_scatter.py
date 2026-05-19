from core.data import subject_ids, session_ids
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from joblib import Parallel, delayed

import scienceplots  # noqa: F401

from utils.paths import MODELS_DIR, FIGURES_DIR
import pickle

# pretty plots
plt.style.use(["nature"])
plt.rcParams["figure.dpi"] = 200


subj_ids = ["MR82", "MR83"]

n_cvs = 5
regions = ["all", "DMS", "DLS"]
epoch_keys = ["full", "choice", "reward", "iti"]

tvs = ["response", "rewarded"]

seed = 0


def get_betas(subj_id, tv, verbose=False):
    betas_sess = []
    sess_ids = session_ids[np.where(subject_ids == subj_id)[0][0]]

    tv_labels = []
    tv_labels_done = False
    for sess_id in sess_ids:
        no_families = False
        betas = {
            reg: {
                epoch: {strategy: [] for strategy in ["both", "mb", "mf"]}
                for epoch in epoch_keys
            }
            for reg in regions
        }

        # load the families
        file_path = (
            MODELS_DIR
            / "fit"
            / subj_id
            / sess_id
            / "tributaries"
            / "balance_and_norm"
            / "results_dict.pkl"
        )

        if not file_path.is_file():
            continue

        with open(file_path, "rb") as f:
            res_dict = pickle.load(f)

        for reg in regions:
            if verbose:
                print(">", reg)
            for epoch in epoch_keys:
                if verbose:
                    print(">>", epoch)
                families_both = res_dict[reg][epoch]["both"]["families"]
                families_mb = res_dict[reg][epoch]["mb"]["families"]
                families_mf = res_dict[reg][epoch]["mf"]["families"]

                if len(families_mb) == 0:
                    no_families = True
                    break

                family_both = families_both[seed]
                family_mb = families_mb[seed]
                family_mf = families_mf[seed]

                beta_both = family_both.mod_taskvar.tv.weight.data[:]
                beta_mb = family_mb.mod_taskvar.tv.weight.data[:]
                beta_mf = family_mf.mod_taskvar.tv.weight.data[:]

                tv_idxs = []
                counter = 0
                for tv_ in family_mb.task_vars:
                    for val in family_mb.trial_data[tv_].unique():
                        if tv_ == tv:
                            tv_idxs.append(counter)
                            if tv_ == "response":
                                if val == 1:
                                    val_str = "left"
                                elif val == -1:
                                    val_str = "right"
                            if tv_ == "rewarded":
                                if val == 1:
                                    val_str = "correct"
                                elif val == 0:
                                    val_str = "incorrect"
                            if not tv_labels_done:
                                tv_labels.append(f"{tv_}_{val_str}")
                        counter += 1
                if not tv_labels_done:
                    tv_labels_done = True

                betas[reg][epoch]["both"] = np.array(
                    [beta_both[tv_idx] for tv_idx in tv_idxs]
                )
                betas[reg][epoch]["mb"] = np.array(
                    [beta_mb[tv_idx] for tv_idx in tv_idxs]
                )
                betas[reg][epoch]["mf"] = np.array(
                    [beta_mf[tv_idx] for tv_idx in tv_idxs]
                )
        if not no_families:
            betas_sess.append(betas)
    return betas_sess, tv_labels


def plot_betas_scatter(subj_id, betas_sess, tv, tv_labels):
    num_tvs = np.shape(betas_sess[0]["all"]["choice"]["mb"])[0]

    colors_tv = ["#17855D", "#9554BD"]

    fig, axes = plt.subplots(ncols=3, nrows=4, figsize=(6, 7))

    for i, reg in enumerate(regions):
        for j, epoch in enumerate(epoch_keys):
            ax = axes[j][i]

            for betas in betas_sess:
                betas_mb = betas[reg][epoch]["mb"]
                betas_mf = betas[reg][epoch]["mf"]

                for tv_idx in range(num_tvs):
                    ax.scatter(
                        betas_mb[tv_idx],
                        betas_mf[tv_idx],
                        s=0.5,
                        color=colors_tv[tv_idx],
                        alpha=0.5,
                        label=f"{tv_labels[tv_idx]}",
                    )

            ax.plot([-2, 2], [-2, 2], linewidth=0.5, color="#000000", linestyle="-")
            ax.axhline(y=0, linewidth=0.5, color="#888888", linestyle="--")
            ax.axvline(x=0, linewidth=0.5, color="#888888", linestyle="--")
            ax.set_xlim([-2, 2])
            ax.set_ylim([-2, 2])
            ax.set_xlabel(r"$\beta$ mb")
            ax.set_ylabel(r"$\beta$ mf")

    fig.suptitle(tv)
    fig.tight_layout()

    fpath_png = (
        FIGURES_DIR
        / "beta_strategy"
        / subj_id
        / f"{tv}_beta_strategy_scatter-{seed}.png"
    )
    fpath_svg = (
        FIGURES_DIR
        / "beta_strategy"
        / subj_id
        / f"{tv}_beta_strategy_scatter-{seed}.svg"
    )
    fpath_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fpath_png, dpi=300, bbox_inches="tight")
    fig.savefig(fpath_svg, dpi=300, bbox_inches="tight")


def plot_betas_hist2d(subj_id, betas_sess, tv):
    num_tvs = np.shape(betas_sess[0]["all"]["choice"]["mb"])[0]

    fig, axes = plt.subplots(
        ncols=3, nrows=4, figsize=(6.6, 7), sharex="all", sharey="all"
    )

    for i, reg in enumerate(regions):
        for j, epoch in enumerate(epoch_keys):
            ax = axes[j][i]

            betas_mb = []
            betas_mf = []
            for betas in betas_sess:
                for tv_idx in range(num_tvs):
                    betas_mb.extend(betas[reg][epoch]["mb"][tv_idx])
                    betas_mf.extend(betas[reg][epoch]["mf"][tv_idx])

            im = ax.hist2d(
                betas_mb,
                betas_mf,
                range=[[-2, 2], [-2, 2]],
                bins=50,
                cmap="Blues",
                norm="log",
                vmax=10,
                density=True,
            )

            ax.plot([-2, 2], [-2, 2], linewidth=0.5, color="#414141", linestyle="-")
            ax.axhline(y=0, linewidth=0.5, color="#888888", linestyle="--")
            ax.axvline(x=0, linewidth=0.5, color="#888888", linestyle="--")
            ax.set_xlim([-2, 2])
            ax.set_ylim([-2, 2])
            ax.set_xlabel(r"$\beta$ mb")
            ax.set_ylabel(r"$\beta$ mf")

    fig.suptitle(tv)
    fig.tight_layout()

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im[3], cax=cbar_ax, ticks=np.logspace(-1, 1, 3, base=10))

    fpath_png = (
        FIGURES_DIR
        / "beta_strategy"
        / subj_id
        / f"{tv}_beta_strategy_hist2d-{seed}.png"
    )
    fpath_svg = (
        FIGURES_DIR
        / "beta_strategy"
        / subj_id
        / f"{tv}_beta_strategy_hist2d-{seed}.svg"
    )
    fpath_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fpath_png, dpi=300, bbox_inches="tight")
    fig.savefig(fpath_svg, dpi=300, bbox_inches="tight")


def plot_betas_hist(subj_id, betas_sess, tv, clump=True):
    from core.data import colors_strategy

    colors_reg = {"all": "#666666", "DMS": "#B1CC16", "DLS": "#07B265"}

    num_tvs = np.shape(betas_sess[0]["all"]["choice"]["mb"])[0]

    fig, axes = plt.subplots(
        ncols=3, nrows=4, figsize=(6, 7), sharex="all", sharey="all", tight_layout=True
    )

    for i, reg in enumerate(regions):
        for j, epoch in enumerate(epoch_keys):
            ax = axes[j][i]

            betas_mb = []
            betas_mf = []
            betas_both = []
            for betas in betas_sess:
                for tv_idx in range(num_tvs):
                    betas_mb.extend(betas[reg][epoch]["mb"][tv_idx])
                    betas_mf.extend(betas[reg][epoch]["mf"][tv_idx])
                    betas_both.extend(betas[reg][epoch]["both"][tv_idx])

            if clump:
                ax.hist(
                    np.concatenate(
                        (
                            betas_mb,
                            betas_mf,
                        )
                    ),
                    bins=np.linspace(-2, 2, 26),
                    weights=np.ones(len(betas_mb) + len(betas_mf))
                    / (len(betas_mb) + len(betas_mf)),
                    color=colors_reg[reg],
                    label="mb/mf",
                    histtype="step",
                )
                ax.hist(
                    betas_both,
                    bins=np.linspace(-2, 2, 26),
                    weights=np.ones(len(betas_both)) / (len(betas_both)),
                    color="#222222",
                    label="both",
                    histtype="step",
                )
            else:
                ax.hist(
                    betas_mb,
                    bins=np.linspace(-2, 2, 26),
                    weights=np.ones(len(betas_mb)) / len(betas_mb),
                    color=colors_strategy["mb"],
                    histtype="step",
                    label=r"$\beta$ mb",
                )
                ax.hist(
                    betas_mf,
                    bins=np.linspace(-2, 2, 26),
                    weights=np.ones(len(betas_mf)) / len(betas_mf),
                    color=colors_strategy["mf"],
                    histtype="step",
                    label=r"$\beta$ mf",
                )

            ax.legend()
            ax.set_xlim([-2, 2])

    fig.suptitle(tv)

    clump_str = "_clump" if clump else ""
    fpath_png = (
        FIGURES_DIR
        / "beta_strategy"
        / subj_id
        / f"{tv}_beta_strategy_hist{clump_str}-{seed}.png"
    )
    fpath_svg = (
        FIGURES_DIR
        / "beta_strategy"
        / subj_id
        / f"{tv}_beta_strategy_hist{clump_str}-{seed}.svg"
    )
    fpath_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fpath_png, dpi=300, bbox_inches="tight")
    fig.savefig(fpath_svg, dpi=300, bbox_inches="tight")


def fit(subj_id, tv, force_redo=False, verbose=False):
    print(f"getting betas for {subj_id}, {tv}")

    save_dir = MODELS_DIR / "fit" / subj_id / "betas" / "balance_and_norm"
    save_path = save_dir / f"{tv}.pkl"
    if save_path.is_file() and not force_redo:
        with open(save_path, "rb") as f:
            res = pickle.load(f)
            betas_sess = res["betas_sess"]
            tv_labels = res["tv_labels"]
    else:
        betas_sess, tv_labels = get_betas(subj_id, tv, verbose=verbose)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({"betas_sess": betas_sess, "tv_labels": tv_labels}, f)

    print(f"plotting betas for {subj_id}, {tv}")
    # plot_betas_scatter(subj_id, betas_sess, tv, tv_labels)
    # plot_betas_hist2d(subj_id, betas_sess, tv)
    plot_betas_hist(subj_id, betas_sess, tv, clump=True)
    print(f"DONE for {subj_id}, {tv}")


subj_tvs = product(subj_ids, tvs)

Parallel(n_jobs=2)(
    delayed(fit)(subj_id, tv, force_redo=True) for (subj_id, tv) in subj_tvs
)
