"""
fit_tributaries.py

Fit (1) one LVM with balanced strategies and (2) two LVMs with one strategy,
to every session, on different epochs.

Author: Stellina X. Ao
Created: 2026-05-06
Last Modified: 2026-05-06
Python Version: 3.11.14
"""

import pickle
import numpy as np
from sg.fitter import LVMFamily
from core.data import subject_ids, session_ids
from utils.paths import MODELS_DIR

from joblib import Parallel, delayed
from itertools import product

subj_ids = ["MR82"]  # ["MR82", "MR83", "MM012"]
regions = ["all", "ACC", "M2", "DMS", "DLS"]
epochs = [
    {
        "key": "full",
        "alignment": "choice",
        "tpre": 0.5,
        "tpost": 1.5,
    },  # full is choice + reward
    {"key": "choice", "alignment": "choice", "tpre": 0.5, "tpost": 0.5},
    {"key": "reward", "alignment": "reward", "tpre": 0, "tpost": 1},
    {"key": "iti", "alignment": "trial_start", "tpre": 1.5, "tpost": -0.5},
]

epoch_ref = {"key": "ref", "alignment": "choice", "tpre": 0.5, "tpost": 1.5}

n_cv = 5

# FIT ALL TRIALS
# subsample for an even number of mb and mf trials
# search through all 16 regl latents
# fit everything (encoder + lvm)
# fit to different epochs

# FIT MB/MF
# use the same split
# use the same regl constant
# fit everything (encoder + lvm)
# fit to different epochs


def gs_regl(subj_id, sess_id, region, epoch):
    # find the best regl constants first
    best_regl_consts = None
    best_score = -np.inf
    print(
        f"Finding regl consts for {subj_id}, {sess_id}, region {region}, epoch {epoch['alignment']}"
    )

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
            alignment=epoch["alignment"],
            tpre=epoch["tpre"],
            tpost=epoch["tpost"],
            alignment_ref=epoch_ref["alignment"],
            tpre_ref=epoch_ref["tpre"],
            tpost_ref=epoch_ref["tpost"],
            binwidth_ms=25,
            norm_activity=True,
            n_splines=2,
            balance_strategy=True,  # subsample for an even number of mb and mf trials
            seed=1234,
            tv_reg={"l2": regl_tv},
            reg={"l2": regl},
        )
        family.fit_all()

        # the entire session is trash, return
        if not family.enough_trials:
            return None
        if not family.lvms_fit:
            continue

        family.eval()

        if (
            best_regl_consts is None
            or best_score
            < family.res_taskvar["r2test"].mean() + family.res_affine["r2test"].mean()
        ):
            best_regl_consts = (regl_tv, regl)
            best_score = (
                family.res_taskvar["r2test"].mean() + family.res_affine["r2test"].mean()
            )
    return best_regl_consts


def fit(subj_id, sess_id, no_dupl=True):
    save_dir = (
        MODELS_DIR
        / "fit"
        / subj_id
        / sess_id
        / "river_n_tributaries"
        / "no_cid_enforcement"
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

        # fit to different epochs
        for epoch in epochs:
            results_dict[region][epoch["key"]] = {"both": {}, "mb": {}, "mf": {}}

            # FIT ALL TRIALS
            results_dict[region][epoch["key"]]["both"] = {
                "families": [],
            }
            # search through all 16 regl latents
            try:
                best_regl_consts = gs_regl(subj_id, sess_id, region, epoch)
            except ValueError:  # region doesn't exist for this session
                break

            # if not enough trials, continue to the next session
            if best_regl_consts is None:
                print(f"DONE for {subj_id}, {sess_id}")
                return

            # fit everything (encoder + lvm)
            counter = 0
            seed_sample = 0
            seeds = np.zeros((5,), dtype=np.int32)
            while counter < n_cv:
                print(
                    f"Fitting for {subj_id}, {sess_id}, region {region}, strategy both, epoch {epoch['alignment']}, number {counter}"
                )

                family = LVMFamily(
                    subj_id=subj_id,
                    sess_id=sess_id,
                    n_latents_mult=1,
                    n_latents_addt=1,
                    regions=None if region == "all" else [region],
                    refit=False,
                    alignment=epoch["alignment"],
                    tpre=epoch["tpre"],
                    tpost=epoch["tpost"],
                    alignment_ref=epoch_ref["alignment"],
                    tpre_ref=epoch_ref["tpre"],
                    tpost_ref=epoch_ref["tpost"],
                    binwidth_ms=25,
                    n_splines=2,
                    norm_activity=True,
                    balance_strategy=True,
                    seed=seed_sample,
                    tv_reg={"l2": best_regl_consts[0]},
                    reg={"l2": best_regl_consts[1]},
                )
                family.fit_all(update_cids=False)

                seed_sample += 1  # gotta do this before it potentially terminates

                if not family.lvms_fit:
                    continue

                family.eval()
                results_dict[region][epoch["key"]]["both"]["families"].append(family)

                seeds[counter] = seed_sample
                counter += 1

            # FIT MB/MF
            for strategy in ["mb", "mf"]:
                results_dict[region][epoch["key"]][strategy] = {
                    "families": [],
                }

                # fit everything
                for i, seed in enumerate(seeds):
                    print(
                        f"Fitting for {subj_id}, {sess_id}, region {region}, strategy {strategy}, epoch {epoch['alignment']}, number {i}"
                    )

                    # use the same split and cids
                    if strategy == "mb":
                        family = results_dict[region][epoch["key"]]["both"]["families"][
                            i
                        ]
                        idxs_subsamp = family.idxs_subsamp_mb
                        # cids = family.cids
                    elif strategy == "mf":
                        family = results_dict[region][epoch["key"]]["both"]["families"][
                            i
                        ]
                        idxs_subsamp = family.idxs_subsamp_mf
                        # cids = family.cids

                    family = LVMFamily(
                        subj_id=subj_id,
                        sess_id=sess_id,
                        n_latents_mult=1,
                        n_latents_addt=1,
                        regions=None if region == "all" else [region],
                        refit=False,
                        alignment=epoch["alignment"],
                        tpre=epoch["tpre"],
                        tpost=epoch["tpost"],
                        alignment_ref=epoch_ref["alignment"],
                        tpre_ref=epoch_ref["tpre"],
                        tpost_ref=epoch_ref["tpost"],
                        binwidth_ms=25,
                        norm_activity=True,
                        n_splines=2,
                        idxs_subsamp=idxs_subsamp,
                        seed=seed,
                        tv_reg={
                            "l2": best_regl_consts[0]
                        },  # use the same regularization constant
                        reg={"l2": best_regl_consts[1]},
                    )
                    family.fit_all(update_cids=False)  # cids=None)  # cids)

                    # could be that there are > 20 mb and mf trials, but < 20 indv mb/mf trials
                    if not family.enough_trials:
                        break

                    family.eval()
                    results_dict[region][epoch["key"]][strategy]["families"].append(
                        family
                    )

    # save
    save_path = save_dir / "results_dict.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(results_dict, f)

    print(f"DONE for {subj_id}, {sess_id}")


# subj_id = "MR82"
# sess_id = "20251027_152036"

# fit(subj_id, sess_id, no_dupl=False)

subj_sess = [
    (subj_id, sess_id)
    for subj_id in ["MR82", "MR83", "MM012"]
    for sess_id in session_ids[np.where(subject_ids == subj_id)[0][0]]
]

Parallel(n_jobs=6)(
    delayed(fit)(subj_id, sess_id, no_dupl=False) for (subj_id, sess_id) in subj_sess
)
