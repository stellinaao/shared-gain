"""
fit_onelatent.py

Fit two LVM with one mult and one addt latent to every session,
one for MB and one for MF trials.

Author: Stellina X. Ao
Created: 2026-05-05
Last Modified: 2026-05-05
Python Version: 3.11.14
"""

import pickle

import numpy as np
from sg.fitter import Encoder
from utils.paths import MODELS_DIR
from core.data import subject_ids, session_ids

from joblib import Parallel, delayed

subj_ids = ["MR82", "MR83", "MM012"]
regions = ["all", "ACC", "M2", "DMS", "DLS"]

n_cv = 5


def fit(subj_id, sess_id, no_dupl=True):
    save_dir = (
        MODELS_DIR
        / "fit"
        / subj_id
        / sess_id
        / "separate_strategy"
        / "encoder_no_update_cid"
    )

    if save_dir.is_dir() and no_dupl:
        print(f"DONE for {subj_id}, {sess_id}")
        return

    results_dict = {
        "subj_id": subj_id,
        "sess_id": sess_id,
    }

    for region in regions:
        results_dict[region] = {}
        for strategy in ["mb", "mf"]:
            # find the best regl constant first
            best_regl_const = None
            best_score = -np.inf
            print(
                f"Finding regl consts for {subj_id}, {sess_id}, {strategy}, region {region}"
            )

            for regl_tv in np.logspace(-3, 0, 4, base=10):
                family = Encoder(
                    subj_id=subj_id,
                    sess_id=sess_id,
                    regions=None if region == "all" else [region],
                    tpre=0.5,
                    tpost=1,
                    binwidth_ms=25,
                    mb_only=True if strategy == "mb" else False,
                    mf_only=True if strategy == "mf" else False,
                    norm_activity=True,
                    seed=1234,
                    tv_reg={"l2": regl_tv},
                )
                try:
                    family.get_data()
                    if family.enough_trials:
                        family.fit_baseline()
                        family.fit_taskvar()
                        family.get_cids()
                except ValueError:
                    break
                family.eval()

                # the entire session is trash, return
                if not family.enough_trials:
                    print(f"DONE for {subj_id}, {sess_id}")
                    return

                if (
                    best_regl_const is None
                    or best_score < family.res_taskvar["r2test"].mean()
                ):
                    best_regl_const = regl_tv
                    best_score = family.res_taskvar["r2test"].mean()

            if best_regl_const is None:
                continue

            results_dict[region][strategy] = {"families": []}
            for seed in range(n_cv):
                print(f"Fitting for {subj_id}, {sess_id}, region {region}, seed {seed}")

                family = Encoder(
                    subj_id=subj_id,
                    sess_id=sess_id,
                    regions=None if region == "all" else [region],
                    tpre=0.5,
                    tpost=1,
                    binwidth_ms=25,
                    norm_activity=True,
                    mb_only=True if strategy == "mb" else False,
                    mf_only=True if strategy == "mf" else False,
                    seed=seed,
                    tv_reg={"l2": best_regl_const},
                )
                family.get_data()
                family.fit_baseline()
                family.fit_taskvar()
                family.get_cids()

                family.eval()

                results_dict[region][strategy]["families"].append(family)

    # save
    save_path = save_dir / "results_dict.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(results_dict, f)

    print(f"DONE for {subj_id}, {sess_id}")


# for subj_id in subj_ids:
#     subj_idx = np.where(subject_ids == subj_id)[0][0]

subj_sess = [
    (subj_id, sess_id)
    for subj_id in ["MR82", "MR83", "MM012"]
    for sess_id in session_ids[np.where(subject_ids == subj_id)[0][0]]
]

Parallel(n_jobs=8)(
    delayed(fit)(subj_id, sess_id, no_dupl=False) for (subj_id, sess_id) in subj_sess
)
