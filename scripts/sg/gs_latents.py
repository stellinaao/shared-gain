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


subj_ids = ["MM012", "MM012", "MR82", "MR83"]
sess_idxs = [4, 5, "20251027_152036", 5]
modes = ["old", "old", "new", "new"]

m_latents = np.linspace(0, 5, 6, dtype=int)
a_latents = np.linspace(0, 5, 6, dtype=int)

for i in [2]:  # range(4):
    subj_id = subj_ids[i]
    sess_id = sess_idxs[i]
    mode = modes[i]

    # unit_spike_times, trial_data, session_data, regions = load_sess(
    #     subj_id=subj_id, sess_idx=sess_idx, mode=mode
    # )
    # print(type(unit_spike_times), type(trial_data), type(session_data), type(regions))

    for m in m_latents:
        for a in a_latents:
            for seed in range(3):
                if m == 0 and a == 0:
                    continue
                print(
                    f"Fitting for {int(m)} multiplicative latents and {int(a)} additive latents, seed {seed}"
                )
                family = LVMFamily(
                    subj_id=subj_id,
                    sess_id=sess_id,
                    n_latents_mult=int(m),
                    n_latents_addt=int(a),
                    refit=True,
                    max_iter=10,
                    norm_activity=True,
                    seed=seed,
                )
                family.fit_all()

                save_path = (
                    PROJECT_ROOT.parents[0]
                    / f"vars/families/{subj_id}/{sess_id}/9acfb66/all/family-m{int(m)}a{int(a)}-seed{seed}.pkl"
                )
                save_path.parent.mkdir(parents=True, exist_ok=True)

                with open(save_path, "wb") as f:
                    # print(type(family))
                    pickle.dump(family, f)

    # for reg in regions:
    #     for m in m_latents:
    #         for a in a_latents:
    #             if m == 0 and a == 0:
    #                 continue
    #             print(
    #                 f"Fitting for {reg}, {int(m)} multiplicative latents and {int(a)} additive latents"
    #             )
    #             family = LVMFamily(
    #                 trial_data=trial_data,
    #                 spike_times={reg: unit_spike_times[reg]},
    #                 session_data=session_data,
    #                 regions=[reg],
    #                 n_latents_mult=int(m),
    #                 n_latents_addt=int(a),
    #                 refit=True,
    #                 max_iter=10,
    #                 norm_activity=True,
    #             )
    #             family.fit_all()

    #             save_path = (
    #                 PROJECT_ROOT.parents[0]
    #                 / f"vars/families/{subj_id}/{sess_idx}/{reg}/family-m{int(m)}a{int(a)}.pkl"
    #             )
    #             save_path.parent.mkdir(parents=True, exist_ok=True)

    #             with open(save_path, "wb") as f:
    #                 pickle.dump(family, f)
