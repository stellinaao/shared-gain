import numpy as np
from core.data import subject_ids, session_ids

from sg.eval_models import get_latent_r

subj_ids = ["MM012"]  # , "MR83"]

max_n_latents = 5


def get_r2s_helper(subj_id, sess_id, regions):
    r2s = {}
    for reg in regions:
        r2s[reg] = get_latent_r(
            subj_id,
            sess_id,
            n_m=max_n_latents,
            n_a=max_n_latents,
            region=reg,
            do_plot=True,
            do_save=True,
        )
    return r2s


for subj_id in subj_ids:
    subj_idx = np.where(subject_ids == subj_id)[0][0]
    for sess_id in session_ids[subj_idx]:
        # sess_id = session_ids[subj_idx][5]
        get_r2s_helper(subj_id, sess_id, regions=["all", "ACC", "M2", "DLS", "DMS"])
