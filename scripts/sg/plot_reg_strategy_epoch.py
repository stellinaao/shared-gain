import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import sem

from core.data import subject_ids, session_ids, colors_epoch
from utils.paths import MODELS_DIR

from joblib import Parallel, delayed

subj_ids = ["MR83"]  # ["MR82", "MR83", "MM012"]
metrics = ["r2test_taskvar", "r2test_affine", "qi"]

regions = ["all", "ACC", "M2", "DMS", "DLS"]
strategies = ["both", "mb", "mf"]
epochs = [
    {"key": "choice", "alignment": "choice", "tpre": 0.5, "tpost": 0.5},
    {"key": "reward", "alignment": "reward", "tpre": 0, "tpost": 1},
    {"key": "iti", "alignment": "trial_start", "tpre": 1.5, "tpost": -0.5},
]
epochs_key = [epoch["key"] for epoch in epochs]

n_cv = 5


def get_metrics(subj_id, reg, strategy, metric):
    sess_ids = session_ids[np.where(subject_ids == subj_id)[0][0]]
    metrics = {epoch: np.full((len(sess_ids), n_cv), np.nan) for epoch in epochs_key}

    for i, sess_id in enumerate(sess_ids):
        file_path = (
            MODELS_DIR
            / "fit"
            / subj_id
            / sess_id
            / "river_n_tributaries"
            / "no_cid_enforcement"
            / "results_dict.pkl"
        )

        if not file_path.is_file():
            continue

        with open(file_path, "rb") as f:
            res_dict = pickle.load(f)

        for epoch in epochs_key:
            if strategy == "both":
                families = res_dict[reg][epoch][strategy]["families"]
                if len(families) == 0:
                    # region doesn't exist for this session
                    break
            else:
                try:
                    families = res_dict[reg][epoch][strategy]["families"]
                except KeyError:
                    # region doesn't exist for this session
                    break

            for j, family in enumerate(families):
                family.eval()
                if metric == "qi":
                    metric_ = family.qi
                elif metric == "r2test_taskvar":
                    try:
                        metric_ = family.res_taskvar["r2test"].nanmedian()
                    except AttributeError:
                        metric_ = np.nan
                elif metric == "r2test_affine":
                    try:
                        metric_ = family.res_affine["r2test"].nanmedian()
                    except AttributeError:
                        metric_ = np.nan
                metrics[epoch][i][j] = metric_
    return metrics


def get_metrics_3x3(subj_id, metric):
    metrics = {}
    for reg in regions:
        metrics[reg] = {}
        for strategy in strategies:
            print(f"collecting metrics for {subj_id}, {metric}, {reg}, {strategy}")
            metrics[reg][strategy] = {}
            metrics[reg][strategy] = get_metrics(subj_id, reg, strategy, metric)

        if np.array(
            [
                np.isnan(metrics[reg][strategy][epoch])
                for epoch in epochs_key
                for strategy in strategies
            ]
        ).all():
            metrics.pop(reg, None)
    print(f"DONE for {subj_id}, {metric}")
    return metrics


def plot_metrics_3x3(subj_id, metrics, metric, do_save=True):
    fig, axes = plt.subplots(ncols=len(metrics), nrows=3, figsize=(5, 4), sharey=True)

    for i, reg in enumerate(metrics):
        for j, strategy in enumerate(metrics[reg]):
            for epoch in metrics[reg][strategy]:
                metrics_epoch = metrics[reg][strategy][epoch]
                metrics_avg = np.nanmean(metrics_epoch, axis=1)
                metrics_sem = sem(metrics_epoch, axis=1, nan_policy="omit")

                session_idxs = np.arange(metrics_epoch.shape[0])

                axes[j][i].plot(
                    session_idxs, metrics_avg, color=colors_epoch[epoch], label=epoch
                )
                axes[j][i].fill_between(
                    session_idxs,
                    metrics_avg - metrics_sem,
                    metrics_avg + metrics_sem,
                    color=colors_epoch[epoch],
                    alpha=0.5,
                )
            axes[j][i].set_xlabel("Sessions")
            axes[j][i].set_ylabel(metric)
            axes[j][i].legend()

    fig.tight_layout()
    if do_save:
        from utils.paths import FIGURES_DIR

        fpath_png = (
            FIGURES_DIR
            / "reg_strategy_epoch"
            / subj_id
            / "no_cid_enforcement"
            / f"{metric}.png"
        )
        fpath_svg = (
            FIGURES_DIR
            / "reg_strategy_epoch"
            / subj_id
            / "no_cid_enforcement"
            / f"{metric}.svg"
        )
        fpath_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fpath_png, dpi=300, bbox_inches="tight")
        fig.savefig(fpath_svg, dpi=300, bbox_inches="tight")


def plot_metrics_3x3_bar(subj_id, metrics, metric, do_save=True):
    fig, axes = plt.subplots(
        ncols=len(metrics), nrows=3, figsize=(3 * len(metrics), 8), sharey="all"
    )

    for i, reg in enumerate(metrics):
        for j, strategy in enumerate(metrics[reg]):
            epochs = metrics[reg][strategy].keys()
            metrics_avg = [
                np.nanmean(metrics[reg][strategy][epoch])
                for epoch in metrics[reg][strategy]
            ]
            metrics_sem = [
                sem(metrics[reg][strategy][epoch], axis=None, nan_policy="omit")
                for epoch in metrics[reg][strategy]
            ]
            axes[j][i].bar(
                epochs,
                metrics_avg,
                color=[colors_epoch[epoch] for epoch in metrics[reg][strategy]],
            )
            axes[j][i].errorbar(
                epochs, metrics_avg, yerr=metrics_sem, capsize=2, fmt=".", color="k"
            )
            axes[j][i].set_ylabel(metric)

    fig.tight_layout()
    if do_save:
        from utils.paths import FIGURES_DIR

        fpath_png = (
            FIGURES_DIR
            / "reg_strategy_epoch"
            / subj_id
            / "no_cid_enforcement"
            / f"{metric}_bars.png"
        )
        fpath_svg = (
            FIGURES_DIR
            / "reg_strategy_epoch"
            / subj_id
            / "no_cid_enforcement"
            / f"{metric}_bars.svg"
        )
        fpath_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fpath_png, dpi=300, bbox_inches="tight")
        fig.savefig(fpath_svg, dpi=300, bbox_inches="tight")


def fit(subj_id, metric, force_redo=False):
    print(f"STARTING {subj_id}, {metric}")
    save_dir = MODELS_DIR / "fit" / subj_id / "metrics" / "no_cid_enforcement"
    save_path = save_dir / f"{metric}.pkl"
    if save_path.is_file() and not force_redo:
        with open(save_path, "rb") as f:
            metrics = pickle.load(f)
    else:
        metrics = get_metrics_3x3(subj_id, metric)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(metrics, f)

    print(f"PLOTTING {subj_id}, {metric}")
    plot_metrics_3x3(subj_id, metrics, metric, do_save=True)
    plot_metrics_3x3_bar(subj_id, metrics, metric, do_save=True)


subj_metric = [(subj_id, metric) for subj_id in subj_ids for metric in metrics]
Parallel(n_jobs=2)(
    delayed(fit)(subj_id, metric, force_redo=False) for subj_id, metric in subj_metric
)
