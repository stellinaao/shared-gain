import numpy as np
import pickle
import matplotlib.pyplot as plt
from utils.paths import MODELS_DIR, FIGURES_DIR
from core.data import subject_ids, session_ids
from scipy.stats import sem

from joblib import Parallel, delayed

subj_ids = ["MR82", "MR83"]

regions = ["all", "DMS", "DLS"]
epoch_keys = ["full", "choice", "reward", "iti"]
strategies = ["both", "mb", "mf"]


def get_n_cids_units(subj_id):
    sess_ids = session_ids[np.where(subject_ids == subj_id)[0][0]]

    n_units = {reg: {epoch: [] for epoch in epoch_keys} for reg in regions}
    n_cids = {
        reg: {epoch: {strategy: [] for strategy in strategies} for epoch in epoch_keys}
        for reg in regions
    }

    for sess_id in sess_ids:
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
            for epoch in epoch_keys:
                for seed in range(5):
                    family = res_dict[reg][epoch]["both"]["families"][seed]
                    try:
                        family_mb = res_dict[reg][epoch]["mb"]["families"][seed]
                        family_mf = res_dict[reg][epoch]["mf"]["families"][seed]
                    except IndexError:
                        break

                    n_cids[reg][epoch]["both"].append(family.cids.shape[0])
                    n_cids[reg][epoch]["mb"].append(family_mb.cids.shape[0])
                    n_cids[reg][epoch]["mf"].append(family_mf.cids.shape[0])

                    n_units[reg][epoch].append(family.num_units)
    return n_units, n_cids


def plot_responsive_pie(n_cids, n_units, subj_id, strategy):
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(5, 5))

    for i, reg in enumerate(regions):
        for j, epoch in enumerate(epoch_keys):
            n_cids_ = np.sum(n_cids[reg][epoch][strategy])
            n_units_ = np.sum(n_units[reg][epoch])
            axes[j][i].pie(
                [n_units_ - n_cids_, n_cids_],
                colors=["#666666", "#F1AEAE"],
                autopct="%.1f%%",
                startangle=90,
            )

    fig.suptitle(f"{subj_id}, {strategy}")

    fpath_png = (
        FIGURES_DIR / "p_responsive" / subj_id / f"p_responsive_{strategy}_pie.png"
    )
    fpath_svg = (
        FIGURES_DIR / "p_responsive" / subj_id / f"p_responsive_{strategy}_pie.svg"
    )
    fpath_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fpath_png, dpi=300, bbox_inches="tight")
    fig.savefig(fpath_svg, dpi=300, bbox_inches="tight")


def plot_responsive_bar(n_cids, n_units, subj_id):
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(6, 7), tight_layout=True)

    for i, reg in enumerate(regions):
        for j, epoch in enumerate(epoch_keys):
            n_cids_ = n_cids[reg][epoch]
            n_units_ = n_units[reg][epoch][0]

            p_responsive = [
                np.nanmean(np.array(n_cids_[strategy]) / n_units_)
                for strategy in strategies
            ]
            p_responsive_sem = [
                sem(np.array(n_cids_[strategy]) / n_units_, nan_policy="omit")
                for strategy in strategies
            ]

            axes[j][i].bar(
                strategies, p_responsive, color=["#666666", "#D7A007", "#183488"]
            )
            axes[j][i].errorbar(
                strategies,
                p_responsive,
                p_responsive_sem,
                capsize=2,
                fmt=".",
                color="k",
            )
            axes[j][i].set_ylabel("p(responsive)")

    fig.suptitle(f"{subj_id}")

    fpath_png = FIGURES_DIR / "p_responsive" / subj_id / "p_responsive_bar.png"
    fpath_svg = FIGURES_DIR / "p_responsive" / subj_id / "p_responsive_bar.svg"
    fpath_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fpath_png, dpi=300, bbox_inches="tight")
    fig.savefig(fpath_svg, dpi=300, bbox_inches="tight")


def main(subj_id, force_redo=False):
    print(f"starting {subj_id}")

    save_dir = MODELS_DIR / "fit" / subj_id / "p_responsive" / "balance_and_norm"
    save_path = save_dir / "p_responsive.pkl"

    if save_path.is_file() and not force_redo:
        with open(save_path, "rb") as f:
            res = pickle.load(f)
            n_units = res["n_units"]
            n_cids = res["n_cids"]
    else:
        n_units, n_cids = get_n_cids_units(subj_id)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({"n_units": n_units, "n_cids": n_cids}, f)

    print(f"plotting {subj_id}")
    for strategy in strategies:
        plot_responsive_pie(n_cids, n_units, subj_id, strategy)
    plot_responsive_bar(n_cids, n_units, subj_id)

    print(f"DONE with {subj_id}")


Parallel(n_jobs=2)(delayed(main)(subj_id) for subj_id in subj_ids)
