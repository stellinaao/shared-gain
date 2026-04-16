import numpy as np
import pickle

from core.data import subject_ids, session_ids
from sg.fitter import LVMFamily
from sg.eval_models import plot_latents, plot_r2_comp, plot_cweights
from utils.paths import PROJECT_ROOT

subj_ids = ["MR82", "MR83"]


def fit_family(subj_id, sess_id):
    family = LVMFamily(
        subj_id=subj_id,
        sess_id=sess_id,
        n_latents_mult=1,
        n_latents_addt=1,
        n_splines=5,
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

    return family


def save_family(family):
    save_path = (
        PROJECT_ROOT.parents[0]
        / f"vars/families/{subj_id}/{sess_id}/all/family-default.pkl"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f:
        # print(type(family))
        pickle.dump(family, f)


def save_latents(family):
    _ = plot_latents(
        family,
        model=family.mod_affine,
        potato=family.strategy,
        label="strategy",
        mode="affine",
        save_fig=True,
    )
    _ = plot_latents(
        family,
        model=family.mod_offset,
        potato=family.strategy,
        label="strategy",
        mode="offset",
        save_fig=True,
    )
    _ = plot_latents(
        family,
        model=family.mod_gain,
        potato=family.strategy,
        label="strategy",
        mode="gain",
        save_fig=True,
    )


def save_r2s(family):
    plot_r2_comp(
        family.res_baseline,
        family.res_taskvar,
        label_a="baseline",
        label_b="taskvar",
        title=r"$r^2$",
        subj_id=family.subj_id,
        sess_id=family.sess_id,
        mode="unity",
        save=True,
    )
    plot_r2_comp(
        family.res_taskvar,
        family.res_affine,
        label_a="taskvar",
        label_b="affine",
        title=r"$r^2$",
        subj_id=family.subj_id,
        sess_id=family.sess_id,
        mode="unity",
        save=True,
    )
    plot_r2_comp(
        family.res_taskvar,
        family.res_offset,
        label_a="taskvar",
        label_b="offset",
        title=r"$r^2$",
        subj_id=family.subj_id,
        sess_id=family.sess_id,
        mode="unity",
        save=True,
    )
    plot_r2_comp(
        family.res_taskvar,
        family.res_gain,
        label_a="taskvar",
        label_b="gain",
        title=r"$r^2$",
        subj_id=family.subj_id,
        sess_id=family.sess_id,
        mode="unity",
        save=True,
    )


def save_cweights(family):
    plot_cweights(family, mode="mult", do_save=True)
    plot_cweights(family, mode="addt", do_save=True)


for subj_id in subj_ids:
    subj_idx = np.where(subject_ids == subj_id)[0][0]
    for sess_id in session_ids[subj_idx]:
        # fit all and individual regions

        # fit
        family = fit_family(subj_id, sess_id)
        save_family(family)

        # latents
        save_latents(family)

        # r2 comp
        save_r2s(family)

        # coupling weight plot
        save_cweights(family)
