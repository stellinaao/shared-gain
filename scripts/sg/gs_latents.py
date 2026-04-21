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

subj_ids = ["MR82"]
sess_idxs = ["20251027_152036"]
regions = ["all", "DMS", "DLS"]

m_latents = np.linspace(0, 5, 6, dtype=int)
a_latents = np.linspace(0, 5, 6, dtype=int)

n_cv = 5


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
        for seed in range(n_cv):
            print(
                f"Fitting for {int(m)} multiplicative latents and {int(a)} additive latents, region {region}, seed {seed}"
            )
            family = LVMFamily(
                subj_id=subj_id,
                sess_id=sess_id,
                n_latents_mult=m,
                n_latents_addt=a,
                regions=None if region == "all" else [region],
                refit=True,
                max_iter=10,
                norm_activity=True,
                seed=seed,
            )
            family.fit_all()
            family.eval()

            results_dict[region]["res_affine"].append(family.res_affine["r2test"])

            # only save the first and last sets of models
            if seed == 0:
                results_dict["family_0"] = family
            if seed == n_cv - 1:
                results_dict["family_1"] = family

    # save
    save_path = (
        PROJECT_ROOT.parents[0]
        / "gs"
        / subj_id
        / sess_id
        / f"results_dict_m{m}a{a}.pkl"
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(results_dict, f)


for i in [0]:
    subj_id = subj_ids[i]
    sess_id = sess_idxs[i]

    Parallel(n_jobs=8)(
        delayed(fit)(subj_id, sess_id, m, a) for (m, a) in product(m_latents, a_latents)
    )
