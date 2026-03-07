import pickle

import numpy as np
from sg.fitter import LVMFamily

trial_data_all = np.load(
    "../vars/trial_data_all_MM012_MM013_all5.npz", allow_pickle=True
)["arr_0"]
session_data_all = np.load(
    "../vars/session_data_all_MM012_MM013_all5.npz", allow_pickle=True
)["arr_0"]
unit_spike_times_all = np.load(
    "../vars/unit_spike_times_all_MM012_MM013_all5.npz", allow_pickle=True
)["arr_0"]
regions_all = np.load("../vars/regions_all_MM012_MM013_all5.npz", allow_pickle=True)[
    "arr_0"
]

subj_idx = 0
sess_idx = 5

unit_spike_times = unit_spike_times_all[subj_idx][sess_idx]
trial_data = trial_data_all[subj_idx][sess_idx]
session_data = session_data_all[subj_idx][sess_idx]
regions = regions_all[subj_idx][sess_idx]
trial_data["block_side"] = np.where(trial_data["block_side"] == "left", 1, -1)

m_latents = np.linspace(1, 10, 10)
a_latents = np.linspace(1, 10, 10)

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
        )
        family.fit_all()

        with open(f"../vars/families/family-m{int(m)}a{int(a)}.pkl", "wb") as f:
            pickle.dump(family, f)
