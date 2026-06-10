import numpy as np
import matplotlib.pyplot as plt
import pickle

from core.data import subject_ids, session_ids
from utils.paths import MODELS_DIR, FIGURES_DIR

from joblib import Parallel, delayed

subj_ids = ["MR82", "MR83"]
regions = ["DMS", "DLS"]


def get_ssi(psths, trial_data, reg, do_plot=False):
    mb_idxs = np.where(trial_data["strategy"] == 1)[0]
    mf_idxs = np.where(trial_data["strategy"] == -1)[0]

    psths_mb = psths[reg][:, mb_idxs, :]
    psths_mf = psths[reg][:, mf_idxs, :]
    psths = psths[reg]

    mfr_mb = psths_mb.mean(axis=(1, 2))
    mfr_mf = psths_mf.mean(axis=(1, 2))
    mfr = psths.mean(axis=(1, 2))

    ssi = (mfr_mb - mfr_mf) / mfr
    return ssi


def get_ssi_subj(subj_id):
    sess_ids = session_ids[np.where(subject_ids == subj_id)[0][0]]
    ssi = {reg: [] for reg in regions}

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
            ssi_reg = []
            for seed in range(5):
                family = res_dict[reg]["full"]["both"]["families"][seed]
                ssi_seed = get_ssi(family.psths, family.trial_data, reg=reg)
                ssi_reg.append(ssi_seed)
            ssi[reg].extend(np.mean(ssi_reg, axis=0))
    return ssi


def plot_ssi(subj_id, ssi):
    colors_reg = {"DMS": "#B1CC16", "DLS": "#07B265"}
    fig, ax = plt.subplots()
    for reg in regions:
        ax.hist(
            ssi[reg],
            bins=np.linspace(-2, 2, 40),
            weights=np.ones(len(ssi[reg])) / len(ssi[reg]),
            color=colors_reg[reg],
            alpha=0.8,
            histtype="step",
            label=reg,
        )
    ax.axvline(x=0, color="k")
    ax.legend()

    fpath_png = FIGURES_DIR / "ssi" / subj_id / "ssi.png"
    fpath_svg = FIGURES_DIR / "ssi" / subj_id / "ssi.svg"
    fpath_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fpath_png, dpi=300, bbox_inches="tight")
    fig.savefig(fpath_svg, dpi=300, bbox_inches="tight")


def main(subj_id):
    print(f"computing {subj_id}")
    ssi = get_ssi_subj(subj_id)
    plot_ssi(subj_id, ssi)
    print(f"DONE for {subj_id}")


Parallel(n_jobs=2)(delayed(main)(subj_id) for subj_id in subj_ids)
