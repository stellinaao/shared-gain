import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import sem

from core.data import subject_ids, session_ids, colors_epoch
from utils.paths import MODELS_DIR

from joblib import Parallel, delayed

subj_ids = ["MM012", "MR82", "MR83"]
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
    metrics = {}

    for epoch in epochs_key:
        print(epoch)
        sess_ids = session_ids[np.where(subject_ids == subj_id)[0][0]]
        metrics[epoch] = np.full((len(sess_ids), n_cv), np.nan)

        for i, sess_id in enumerate(sess_ids):
            print(sess_id)
            file_path = (
                MODELS_DIR
                / "fit"
                / subj_id
                / sess_id
                / "river_n_tributaries"
                / "results_dict.pkl"
            )

            if not file_path.is_file():
                continue

            with open(file_path, "rb") as f:
                res_dict = pickle.load(f)

            families = res_dict[reg][epoch][strategy]["families"]
            if len(families) == 0:
                raise ValueError("region doesn't exist")

            for j, family in enumerate(families):
                if metric == "qi":
                    metric = family.qi
                elif metric == "r2test_taskvar":
                    metric = family.res_taskvar["r2test"].mean()
                elif metric == "r2test_affine":
                    metric = family.res_affine["r2test"].mean()
                metrics[epoch][i][j] = metric
    return metrics


def get_metrics_3x3(subj_id, metric):
    metrics = {}
    for reg in regions:
        metrics[reg] = {}
        for strategy in strategies:
            try:
                metrics[reg][strategy] = get_metrics(subj_id, reg, strategy, metric)
            except ValueError:
                metrics.pop(reg)
                break
    return metrics


def plot_metrics_3x3(subj_id, metrics, metric, do_save=True):
    fig, axes = plt.subplots(ncols=len(metrics), nrows=3, figsize=(5, 4))

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
                    label=epoch,
                )
            axes[j][i].set_xlabel("Sessions")
            axes[j][i].set_ylabel(metric)

    fig.tight_layout()
    if do_save:
        from utils.paths import FIGURES_DIR

        fpath_png = FIGURES_DIR / "reg_strategy_epoch" / subj_id / f"{metric}.png"
        fpath_svg = FIGURES_DIR / "reg_strategy_epoch" / subj_id / f"{metric}.svg"
        fpath_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fpath_png, dpi=300, bbox_inches="tight")
        fig.savefig(fpath_svg, dpi=300, bbox_inches="tight")


def fit(subj_id, metric):
    print(f"calculating for {subj_id}, {metric}")
    metrics = get_metrics_3x3(subj_id, metric)
    plot_metrics_3x3(subj_id, metrics, metric, do_save=True)


subj_metric = [(subj_id, metric) for subj_id in subj_ids for metric in metrics]
Parallel(n_jobs=8)(delayed(fit)(subj_id, metric) for subj_id, metric in subj_metric)
