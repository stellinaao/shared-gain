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

from core.data import load_sess

subj_ids = ["MR82", "MR83", "MR85"]
sess_idxs = [5, 5, 5]
modes = ["new", "new", "new"]

m_latents = np.linspace(0, 5, 6, dtype=int)
a_latents = np.linspace(0, 5, 6, dtype=int)

for i in range(3):
    subj_id = subj_ids[i]
    sess_idx = sess_idxs[i]
    mode = modes[i]

    unit_spike_times, trial_data, session_data, regions = load_sess(
        subj_id=subj_id, sess_idx=sess_idx, mode=mode
    )

    for m in m_latents:
        for a in a_latents:
            if m == 0 and a == 0:
                continue
            print(
                f"Fitting for {int(m)} multiplicative latents and {int(a)} additive latents"
            )
            family = LVMFamily(
                trial_data=trial_data,
                spike_times=unit_spike_times,
                session_data=session_data,
                regions=regions,
                n_latents_mult=int(m),
                n_latents_addt=int(a),
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
                refit=True,
                max_iter=10,
                norm_activity=True,
            )
            family.fit_all()

            save_path = (
                PROJECT_ROOT.parents[0]
                / f"vars/families/{subj_id}/{sess_idx}/all/family-m{int(m)}a{int(a)}.pkl"
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, "wb") as f:
                # print(type(family))
                pickle.dump(family, f)

    for reg in regions:
        for m in m_latents:
            for a in a_latents:
                if m == 0 and a == 0:
                    continue
                print(
                    f"Fitting for {reg}, {int(m)} multiplicative latents and {int(a)} additive latents"
                )
                family = LVMFamily(
                    trial_data=trial_data,
                    spike_times={reg: unit_spike_times[reg]},
                    session_data=session_data,
                    regions=[reg],
                    n_latents_mult=int(m),
                    n_latents_addt=int(a),
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
                    refit=True,
                    max_iter=10,
                    norm_activity=True,
                )
                family.fit_all()

                save_path = (
                    PROJECT_ROOT.parents[0]
                    / f"vars/families/{subj_id}/{sess_idx}/{reg}/family-m{int(m)}a{int(a)}.pkl"
                )
                save_path.parent.mkdir(parents=True, exist_ok=True)

                with open(save_path, "wb") as f:
                    pickle.dump(family, f)
