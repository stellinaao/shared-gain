"""
gs_latents.py

Functions to load and process neural and behavioral data
collected from the dynamic foraging task.

Author: Stellina X. Ao
Created: 2026-03-26 # definitely created before, but lost the record
Last Modified: 2026-03-26
Python Version: 3.11.14
"""

import pickle

import numpy as np
from sg.fitter import LVMFamily
from utils.paths import PROJECT_ROOT

from joblib import Parallel, delayed
from itertools import product

subj_ids = ["MM012", "MR82", "MR83"]
# sess_idxs = ["20251027_162326"]
regions = ["all", "ACC", "M2", "DMS", "DLS"]

m_latents = np.linspace(0, 5, 6, dtype=int)
a_latents = np.linspace(0, 5, 6, dtype=int)

n_cv = 5


def gs_regl(subj_id, sess_id, region, m, a):
    # find the best regl constants first
    best_regl_consts = None
    best_score = -np.inf
    print(
        f"Finding regl consts for {subj_id}, {sess_id}, region {region}, m {m}, a {a}"
    )

    for regl_tv, regl in product(
        np.logspace(-3, 0, 4, base=10), np.logspace(-3, 0, 4, base=10)
    ):
        family = LVMFamily(
            subj_id=subj_id,
            sess_id=sess_id,
            n_latents_mult=m,
            n_latents_addt=a,
            regions=None if region == "all" else [region],
            refit=False,
            alignment="choice",
            tpre=0.5,
            tpost=1.5,
            binwidth_ms=25,
            norm_activity=True,
            n_splines=2,
            balance_strategy=False,  # True,  # subsample for an even number of mb and mf trials
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


def fit(subj_id, sess_id, m, a):
    if m == 0 and a == 0:
        return

    results_dict = {
        "subj_id": subj_id,
        "sess_id": sess_id,
        "n_latents_mult": m,
        "n_latents_addt": a,
    }

    for region in regions:
        results_dict[region] = {"res_affine": []}

        # search through all 16 regl latents
        try:
            best_regl_consts = gs_regl(subj_id, sess_id, region, m, a)
        except ValueError:  # region doesn't exist for this session
            break

        # if not enough trials, continue to the next session
        if best_regl_consts is None:
            print(f"DONE for {subj_id}, {sess_id}, {m}, {a}")
            return

        # fit everything (encoder + lvm)
        counter = 0
        seed_sample = 0
        seeds = np.zeros((5,), dtype=np.int32)
        while counter < n_cv:
            print(
                f"Fitting for {subj_id}, {sess_id}, region {region}, m {m}, a {a}, number {counter}"
            )

            family = LVMFamily(
                subj_id=subj_id,
                sess_id=sess_id,
                n_latents_mult=m,
                n_latents_addt=a,
                regions=None if region == "all" else [region],
                refit=False,
                alignment="choice",
                tpre=0.5,
                tpost=1.5,
                binwidth_ms=25,
                n_splines=2,
                norm_activity=True,
                balance_strategy=False,  # True,
                seed=seed_sample,
                tv_reg={"l2": best_regl_consts[0]},
                reg={"l2": best_regl_consts[1]},
            )
            family.fit_all()

            seed_sample += 1  # gotta do this before it potentially terminates

            if not family.lvms_fit:
                continue

            family.eval()

            results_dict[region]["res_affine"].append(family.res_affine["r2test"])

            # only save the first and last sets of models
            if counter == 0:
                results_dict["family_0"] = family
            if counter == n_cv - 1:
                results_dict["family_1"] = family

            seeds[counter] = seed_sample
            counter += 1

    # save
    save_path = (
        PROJECT_ROOT.parents[0]
        / "gs"
        / subj_id
        / sess_id
        / "two_spline"
        / f"results_dict_m{m}a{a}.pkl"
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(results_dict, f)

    print(f"DONE for {subj_id}, {sess_id}, {m}, {a}")


subj_id = "MR83"
sess_id = "20251027_162326"
Parallel(n_jobs=8)(
    delayed(fit)(subj_id, sess_id, m, a) for (m, a) in product(m_latents, a_latents)
)
