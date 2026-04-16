"""
TODO:
1. check that this refactored version works serially first
2. parallelize and sonic the hedgehog
"""

import pickle

import numpy as np
from sg.fitter import LVMFamily
from utils.paths import PROJECT_ROOT
from itertools import product
from joblib import Parallel, delayed

from core.data import subject_ids, session_ids

subj_ids = ["MM012", "MR82", "MR83"]

m_latents = np.linspace(0, 5, 6, dtype=int)
a_latents = np.linspace(0, 5, 6, dtype=int)


def fit_all(subj_id, sess_id, m, a):
    if m == 0 and a == 0:
        return
    family_all = fit_family(subj_id, sess_id, m, a, region="all")
    for reg in family_all.regions:
        fit_family(subj_id, sess_id, m, a, region=reg)


def fit_family(subj_id, sess_id, m, a, region="all"):
    print(
        f"Fitting {subj_id}, {sess_id} with {m} multiplicative latents and {a} additive latents, region {region}"
    )
    family = LVMFamily(
        subj_id=subj_id,
        sess_id=sess_id,
        regions=None if region == "all" else [region],
        n_latents_mult=m,
        n_latents_addt=a,
        task_vars={
            "digital": [
                "response",
                "rewarded",
                "block_side",
                "strategy",
                "response_prev",
                "rewarded_prev",
            ],
            "analog": [],
        },
        tpre=0.5,
        tpost=0.5,
    )
    family.fit_all()
    family.eval()

    save_path = (
        PROJECT_ROOT.parents[0]
        / f"vars/families/{subj_id}/{sess_id}/no_pr_onesec/{region}/family-m{m}a{a}.pkl"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(family, f)

    return family


for subj_id in subj_ids:
    subj_idx = np.where(subject_ids == subj_id)[0][0]
    for sess_id in session_ids[subj_idx]:
        # sess_id = session_ids[subj_idx][5] # take the fifth session for all of them
        if not sess_id == session_ids[subj_idx][5]:  # because we already did this one
            Parallel(n_jobs=8, verbose=10)(
                delayed(fit_all)(subj_id, sess_id, m, a)
                for m, a in product(m_latents, a_latents)
            )
