import pickle
import numpy as np
from utils.paths import PROJECT_ROOT

from core.data import load_sess, subject_ids, session_ids
from sg.corr_utils import get_scorr, get_ncorr, plot_scorr_all, plot_ncorr_all

from joblib import Parallel, delayed

subj_ids = ["MM012", "MR82", "MR83"]


def fit(subj_id, sess_id):
    print(f"calculating for {subj_id}, {sess_id}")
    _, trial_data, psths, _, regions = load_sess(
        subj_id=subj_id,
        sess_id=sess_id,
        tpre=0.5,
        tpost=1,
        binwidth_ms=25,
        alignment="choice",
        thresh=1,
    )

    scorr = get_scorr(psths, regions)
    ncorr = get_ncorr(psths, regions, trial_data)
    corr_dict = {
        "subj_id": subj_id,
        "sess_id": sess_id,
        "scorr": scorr,
        "ncorr": ncorr,
    }

    plot_scorr_all(scorr, regions, subj_id, sess_id, vmax=1, do_save=True)
    plot_ncorr_all(ncorr, regions, subj_id, sess_id, vmax=0.2, do_save=True)

    # save
    save_path = PROJECT_ROOT.parents[0] / "corr" / subj_id / sess_id / "sncorr.pkl"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(corr_dict, f)


for subj_id in subj_ids:
    subj_idx = np.where(subject_ids == subj_id)[0][0]
    print(subj_idx)

    Parallel(n_jobs=8)(
        delayed(fit)(subj_id, sess_id) for sess_id in session_ids[subj_idx]
    )
