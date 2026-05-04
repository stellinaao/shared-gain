"""
fit_onelatent.py

Fit an LVM with one mult and one addt latent to every session.

Author: Stellina X. Ao
Created: 2026-05-04
Last Modified: 2026-05-04
Python Version: 3.11.14
"""

import pickle

import numpy as np
from sg.fitter import LVMFamily
from utils.paths import MODELS_DIR

from itertools import product

subj_ids = ["MR83", "MM012", "MR82"]
regions = ["all", "ACC", "M2", "DMS", "DLS"]

n_cv = 5


def fit(subj_id, sess_id, no_dupl=True):
    save_dir = MODELS_DIR / "fit" / subj_id / sess_id / "one_latent"

    if save_dir.is_dir() and no_dupl:
        return

    results_dict = {
        "subj_id": subj_id,
        "sess_id": sess_id,
    }

    for region in regions:
        # find the best regl constants first
        best_regl_consts = None
        best_score = -np.inf
        print(f"Finding regl consts for {subj_id}, {sess_id}, region {region}")

        for regl_tv, regl in product(
            np.logspace(-3, 0, 4, base=10), np.logspace(-3, 0, 4, base=10)
        ):
            family = LVMFamily(
                subj_id=subj_id,
                sess_id=sess_id,
                n_latents_mult=1,
                n_latents_addt=1,
                regions=None if region == "all" else [region],
                refit=False,
                tpre=0.5,
                tpost=1,
                binwidth_ms=25,
                norm_activity=True,
                seed=1234,
                tv_reg={"l2": regl_tv},
                reg={"l2": regl},
            )
            try:
                family.fit_all()
            except ValueError:
                break
            family.eval()

            if (
                best_regl_consts is None
                or best_score
                < family.res_taskvar["r2test"].mean()
                + family.res_affine["r2test"].mean()
            ):
                best_regl_consts = (regl_tv, regl)
                best_score = (
                    family.res_taskvar["r2test"].mean()
                    + family.res_affine["r2test"].mean()
                )

        if best_regl_consts is None:
            continue

        results_dict[region] = {"families": [], "res_tv_lvms": []}
        for seed in range(n_cv):
            print(f"Fitting for {subj_id}, {sess_id}, region {region}, seed {seed}")

            family = LVMFamily(
                subj_id=subj_id,
                sess_id=sess_id,
                n_latents_mult=1,
                n_latents_addt=1,
                regions=None if region == "all" else [region],
                refit=False,
                tpre=0.5,
                tpost=1,
                binwidth_ms=25,
                norm_activity=True,
                seed=seed,
                tv_reg={"l2": best_regl_consts[0]},
                reg={"l2": best_regl_consts[1]},
            )
            family.fit_all()
            family.eval()

            results_dict[region]["families"].append(family)
            results_dict[region]["res_tv_lvms"].append(
                {
                    "r2test_taskvar": family.res_taskvar["r2test"].mean(),
                    "r2test_affine": family.res_affine["r2test"].mean(),
                    "r2test_diff": (
                        family.res_affine["r2test"] - family.res_taskvar["r2test"]
                    ).mean(),
                    "qi": family.qi,
                }
            )

    # save
    save_path = save_dir / "results_dict.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(results_dict, f)
    print(f"DONE for {subj_id}, {sess_id}")


fit("MR82", "20251030_150221", no_dupl=False)
# for subj_id in subj_ids:
#     subj_idx = np.where(subject_ids == subj_id)[0][0]
#     Parallel(n_jobs=8)(
#         delayed(fit)(subj_id, sess_id) for sess_id in session_ids[subj_idx]
#     )
