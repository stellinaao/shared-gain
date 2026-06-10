import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

from core.data import subject_ids, session_ids, load_sess, colors_strategy
from utils.paths import FIGURES_DIR

from joblib import Parallel, delayed


def get_pcond_tv_strategy(subj_id, tv):
    if not (tv == "response" or tv == "rewarded"):
        raise ValueError("valid values for tv are 'response' or 'rewarded'")

    sess_ids = session_ids[np.where(subject_ids == subj_id)[0][0]]
    pcond = {"mb": np.zeros((len(sess_ids),)), "mf": np.zeros((len(sess_ids),))}
    for i, sess_id in enumerate(sess_ids):
        _, trial_data, _, _, _ = load_sess(
            subj_id=subj_id,
            sess_id=sess_id,
            tpre=0.5,
            tpost=0.5,
            alignment="choice",
            alignment_ref=None,
            binwidth_ms=25,
            thresh=1,
        )

        trial_data_mb = trial_data[trial_data["strategy"] == 1]
        trial_data_mf = trial_data[trial_data["strategy"] == -1]

        if tv == "response":
            if len(trial_data_mb) > 20:
                pcond["mb"][i] = (trial_data_mb["response"] == 1).mean()  # p(l|mb)
            else:
                pcond["mb"][i] = np.nan
            if len(trial_data_mf) > 20:
                pcond["mf"][i] = (trial_data_mf["response"] == 1).mean()  # p(l|mf)
            else:
                pcond["mf"][i] = np.nan
        elif tv == "rewarded":
            if len(trial_data_mb) > 20:
                pcond["mb"][i] = (trial_data_mb["rewarded"] == 1).mean()  # p(c|mb)
            else:
                pcond["mb"][i] = np.nan
            if len(trial_data_mf) > 20:
                pcond["mf"][i] = (trial_data_mf["rewarded"] == 1).mean()  # p(c|mf)
            else:
                pcond["mf"][i] = np.nan
    return pcond


def plot_pcond(subj_id):
    pcond_response = get_pcond_tv_strategy(subj_id, tv="response")
    pcond_rewarded = get_pcond_tv_strategy(subj_id, tv="rewarded")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4, 2))

    # response
    pcond_response_mean = (
        np.nanmean(pcond_response["mb"]),
        np.nanmean(pcond_response["mf"]),
    )
    pcond_response_sem = (
        sem(pcond_response["mb"], nan_policy="omit"),
        sem(pcond_response["mf"], nan_policy="omit"),
    )

    axes[0].bar(
        ["mb", "mf"],
        pcond_response_mean,
        color=[colors_strategy["mb"], colors_strategy["mf"]],
    )
    axes[0].errorbar(
        ["mb", "mf"],
        pcond_response_mean,
        pcond_response_sem,
        capsize=2,
        color="k",
        fmt=".",
    )
    axes[0].set_ylabel("p(L)")

    # rewarded
    pcond_rewarded_mean = (
        np.nanmean(pcond_rewarded["mb"]),
        np.nanmean(pcond_rewarded["mf"]),
    )
    pcond_rewarded_sem = (
        sem(pcond_rewarded["mb"], nan_policy="omit"),
        sem(pcond_rewarded["mf"], nan_policy="omit"),
    )

    axes[1].bar(
        ["mb", "mf"],
        pcond_rewarded_mean,
        color=[colors_strategy["mb"], colors_strategy["mf"]],
    )
    axes[1].errorbar(
        ["mb", "mf"],
        pcond_rewarded_mean,
        pcond_rewarded_sem,
        capsize=2,
        color="k",
        fmt=".",
    )
    axes[1].set_ylabel("p(corr)")

    fig.tight_layout()

    fpath_png = FIGURES_DIR / "pcond_tv_strategy" / subj_id / "pcond_tv_strategy.png"
    fpath_svg = FIGURES_DIR / "pcond_tv_strategy" / subj_id / "pcond_tv_strategy.svg"

    fpath_png.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(fpath_png, dpi=300, bbox_inches="tight")
    fig.savefig(fpath_svg, dpi=300, bbox_inches="tight")


subj_ids = ["MR82", "MR83", "MM012"]
Parallel(n_jobs=4)(delayed(plot_pcond)(subj_id) for subj_id in subj_ids)
