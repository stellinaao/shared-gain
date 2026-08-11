"""
data.py

Functions to load and process neural and behavioral data
collected from the dynamic foraging task.

Author: Stellina X. Ao
Created: 2025-12-18
Last Modified: 2026-04-05
Python Version: 3.11.14
"""

import pickle

import numpy as np
import pandas as pd
import re
import os
import shutup

from sklearn.preprocessing import OneHotEncoder as OHE
from scipy.stats import zscore

from spks.utils import get_cluster_spike_times
from spks.event_aligned import compute_spike_count

# from damn.alignment import compute_spike_count
from ndnt.utils.NDNutils import tent_basis_generate
from utils.paths import DATA_DIR

from joblib import Parallel, delayed

shutup.please()

# CONSTANTS
session_pattern = re.compile(r"^\d{8}_\d{6}$")
subject_ids = np.sort(
    [subj_id for subj_id in os.listdir(DATA_DIR) if not subj_id.startswith(".")]
)
session_ids = [
    np.sort(
        [
            sess_id
            for sess_id in os.listdir(DATA_DIR / subj_id)
            if session_pattern.match(sess_id)
        ]
    )
    for subj_id in subject_ids
]

probes = ["imec0", "imec1"]

tv_vals = {
    "response": ["left", "right"],
    "rewarded": ["incorr", "corr"],
    "block_side": ["left", "right"],
    "response_prev": ["left", "none", "right"],
    "rewarded_prev": ["incorr", "corr"],
    "strategy": ["mf", "mb"],
}

tv_name_map = {
    "response_1": "response_left",
    "response_-1": "response_right",
    "rewarded_0": "rewarded_incorr",
    "rewarded_1": "rewarded_corr",
    "block_side_1": "block_side_left",
    "block_side_-1": "block_side_right",
    "response_prev_1": "response_prev_left",
    "response_prev_0": "response_prev_none",
    "response_prev_-1": "response_prev_right",
    "rewarded_prev_0": "rewarded_prev_incorr",
    "rewarded_prev_1": "rewarded_prev_corr",
    "strategy_-1": "strategy_mf",
    "strategy_1": "strategy_mb",
}


# LOAD DATA
def load_sess(
    subj_id=None,
    sess_id=None,
    subj_idx=None,
    sess_idx=None,
    tpre=0.5,
    tpost=1,
    binwidth_ms=25,
    alignment="choice",
    tpre_ref=0.5,
    tpost_ref=1,
    alignment_ref=None,
    add_svd=False,
    add_licks=False,
    full_trial=False,
    trial_start_pre=0,
    thresh=1,
):
    """
    subj_id:    the actual id of the subject, e.g., MM012, MR83
    sess_id:    the actual id of the session, e.g., 20231211_172819

    ! THE FOLLOWING TWO ARE NOT RECOMMENDED FOR REPRODUCIBLE CODE ACROSS
    ! INDIVIDUALS BECAUSE OF DIFFERENCES IN THE DATA FILES PRESENT IN
    ! THEIR DATA FOLDERS

    subj_idx:   the index of the subject, e.g., 0 for the first subject in your data folder
    sess_idx:   the index of the session, e.g., -1 for the last session

    thresh:     the minimum firing rate to keep, defaults to 1 Hz
    mode:       'old' to load data from the old cohort (MM012 & MM013), 'new' to load data from the new cohort (MR82, MR83, MR85)
    """

    if (subj_id is None and subj_idx is None) or (sess_id is None and sess_idx is None):
        raise ValueError("wow all nones?! try again bucko.")
    else:
        if subj_id is None:
            subj_id = subject_ids[subj_idx]
        if sess_id is None:
            if subj_idx is None:
                subj_idx = np.where(subject_ids == subj_id)[0][0]
            sess_id = session_ids[subj_idx][sess_idx]

    if subj_id == "MM012" or subj_id == "MM013":
        mode = "old"
    elif subj_id.startswith("MR"):
        mode = "new"

    if mode == "new":
        fpath = DATA_DIR / subj_id / sess_id
        if not fpath.exists():
            raise FileNotFoundError

        # load from pkl
        neural_data = pd.read_pickle(fpath / "neural_data.pkl")

        spike_times = {
            region: values["spike_times"] for region, values in neural_data.items()
        }
        session_data = pd.read_pickle(fpath / "session_data.pkl")
        trial_data = pd.read_csv(fpath / "trialdata.csv")
        regions = np.array(list(neural_data.keys()))

        # trial_data edits and addendums
        trial_data["trial_start_time"] = session_data["events"].iloc[
            np.where(np.array(session_data["event_labels"]) == "trial_start")[0][0]
        ]["event_timestamps"]

        trial_data["block_side"] = np.where(
            trial_data["current_block_side"] == "left", 1, -1
        )

        trial_data = add_prev(trial_data)
        trial_data = add_strat(trial_data, session_data)

        trial_mask = get_trial_mask(trial_data)
        trial_data = trial_data[trial_mask]

        # get psths
        psths, spike_times, tbin_edges = get_psths_ref(
            spike_times,
            trial_data,
            session_data,
            regions,
            tpre,
            tpost,
            binwidth_ms,
            alignment,
            trial_start_pre,
            tpre_ref,
            tpost_ref,
            alignment_ref,
            mode,
            thresh=1,
        )

        # add svds
        if add_svd:
            svds_df = get_svd_df(
                subj_id,
                sess_id,
                trial_data,
                session_data,
                alignment=alignment,
                tpre=tpre,
                tpost=tpost,
                full_trial=full_trial,
            )
            if full_trial:
                trial_data = trial_data.join(svds_df)
            else:
                trial_data[svds_df.columns] = svds_df.reset_index(drop=True).values
        if add_licks:
            licks_df = get_licks_df(
                subj_id,
                sess_id,
                trial_data,
                session_data,
                alignment=alignment,
                tpre=tpre,
                tpost=tpost,
                full_trial=full_trial,
            )
            if full_trial:
                trial_data = trial_data.join(licks_df)
            else:
                trial_data[licks_df.columns] = licks_df.reset_index(drop=True).values

        return spike_times, trial_data, psths, session_data, regions, tbin_edges
    elif mode == "old":
        # load data and set variables needed for aligning spikes to behavioral events
        if subj_idx is None:
            subj_idx = np.where(np.array(subject_ids) == subj_id)[0][0]
        if sess_idx is None:
            sess_idx = np.where(session_ids[subj_idx] == sess_id)[0][0]
        _, _, trial_data_r, neural_data, animal_data, session_data = load_data_sess(
            subj_idx=subj_idx, sess_idx=sess_idx
        )
        spike_clusters, spike_times, _, _, _, regions = get_align_vars(
            neural_data, animal_data
        )
        spike_times = get_unit_spike_times(
            spike_times, spike_clusters, neural_data, regions
        )
        regions = np.concatenate(regions)

        # trial_data addendums
        trial_data = add_prev(trial_data_r)
        trial_data = add_strat(trial_data, session_data)

        trial_data["trial_start_time"] = trial_data["task_start_time"]
        trial_mask = get_trial_mask(trial_data)
        trial_data = trial_data[trial_mask]

        # get psths
        psths, trial_mask, zstd_units = get_psths(
            spike_times,
            trial_data,
            session_data,
            regions,
            tpre=tpre,
            tpost=tpost,
            binwidth_ms=binwidth_ms,
            alignment=alignment,
            trial_start_pre=trial_start_pre,
            reward_only=False,
            prev_filter=False,
            mode=mode,
        )
        # update spike_times with the removed units
        for reg in regions:
            if len(zstd_units[reg]) > 0:
                spike_times[reg] = [
                    st_unit
                    for i, st_unit in enumerate(spike_times[reg])
                    if i not in set(zstd_units[reg])
                ]

        trial_data = trial_data[trial_mask]

        psths, spike_times = rem_low_fr(
            psths, spike_times, thresh=thresh, binwidth_ms=binwidth_ms
        )

        return spike_times, trial_data, psths, session_data, regions, tbin_edges
    else:
        raise ValueError("valid values for mode are 'old' and 'new.'")


def get_psths_ref(
    spike_times,
    trial_data,
    session_data,
    regions,
    tpre,
    tpost,
    binwidth_ms,
    alignment,
    trial_start_pre,
    tpre_ref,
    tpost_ref,
    alignment_ref,
    mode,
    thresh=1,
):
    psths, trial_mask, tbin_edges = get_psths(
        spike_times,
        trial_data,
        session_data,
        regions,
        tpre=tpre,
        tpost=tpost,
        binwidth_ms=binwidth_ms,
        alignment=alignment,
        trial_start_pre=trial_start_pre,
        do_rem_zstd=False,
        reward_only=False,
        prev_filter=False,
        mode=mode,
    )

    if alignment_ref is not None and (
        not (alignment == alignment_ref)
        or not (tpre == tpre_ref)
        or not (tpost == tpost_ref)
    ):
        print("ahoy matey")
        print(alignment, alignment_ref)
        print(tpre, tpre_ref)
        print(tpost, tpost_ref)
        psths_ref, _, tbin_edges_ref = get_psths(
            spike_times,
            trial_data,
            session_data,
            regions,
            tpre=tpre_ref,
            tpost=tpost_ref,
            binwidth_ms=binwidth_ms,
            alignment=alignment_ref,
            trial_start_pre=trial_start_pre,
            do_rem_zstd=False,  # so that there are no indexing issues later
            reward_only=False,
            prev_filter=False,
            mode=mode,
        )

    else:
        psths_ref = None

    assert trial_mask.mean() == 1

    psths, spike_times = rem_low_fr(
        psths,
        spike_times,
        psths_ref=psths_ref,
        thresh=thresh,
        binwidth_ms=binwidth_ms,
    )

    return psths, spike_times, tbin_edges


def load_data(thresh=1):
    trial_data_r = []
    trial_data = []
    session_data = []
    unit_spike_times = []
    regions = []

    for subj_idx in range(len(subject_ids)):
        print(f"Subject: {subject_ids[subj_idx]}")
        (
            trial_data_r_subj,
            trial_data_subj,
            session_data_subj,
            unit_spike_times_subj,
            regions_subj,
        ) = load_subj(subj_idx, thresh=thresh)

        trial_data_r.append(trial_data_r_subj)
        trial_data.append(trial_data_subj)
        session_data.append(session_data_subj)
        unit_spike_times.append(unit_spike_times_subj)
        regions.append(regions_subj)

    return trial_data_r, trial_data, session_data, unit_spike_times, regions


def load_subj(subj_idx, thresh=1):
    trial_data_r = []
    trial_data = []
    session_data = []
    unit_spike_times = []
    regions = []

    for sess_idx in range(len(session_ids[subj_idx])):
        print(f"> Session: {session_ids[subj_idx][sess_idx]}")
        (
            trial_data_r_sess,
            trial_data_sess,
            session_data_sess,
            unit_spike_times_lite_sess,
            regions_sess,
        ) = load_sess(subj_idx, sess_idx, thresh=thresh)
        trial_data_r.append(trial_data_r_sess)
        trial_data.append(trial_data_sess)
        session_data.append(session_data_sess)
        unit_spike_times.append(unit_spike_times_lite_sess)
        regions.append(regions_sess)

    return trial_data_r, trial_data, session_data, unit_spike_times, regions


def load_data_sess(
    subj_id=None, sess_id=None, subj_idx=None, sess_idx=None, mode="new"
):
    fpath_data = DATA_DIR / subject_ids[subj_idx] / session_ids[subj_idx][sess_idx]

    riglog = np.load(
        f"{fpath_data}/riglog.npy", allow_pickle="TRUE"
    ).item()  # log from the rig of events that happened
    corrected_onsets = np.load(
        f"{fpath_data}/corrected_onsets.npy", allow_pickle="TRUE"
    ).item()  # TTL timestamps of behavioral events
    trial_data = pd.read_csv(
        f"{fpath_data}/trialdata.csv"
    )  # information about task on each trial

    # neural_data is a dict which has information about the spike-sorted neural data
    neural_data = []
    for probe in probes:
        with open(f"{fpath_data}/{probe}_neural_data.pkl", "rb") as f:
            neural_data.append(pickle.load(f))

    # animal_data is a dict which has information about the animal/implant locations
    with open(f"{fpath_data}/animal_data.pkl", "rb") as f:
        animal_data = pickle.load(f)

    # session_data is a dict which has information about the session, such as the strategy IDs for each block
    with open(f"{fpath_data}/session_data.pkl", "rb") as f:
        session_data = pickle.load(f)

    return riglog, corrected_onsets, trial_data, neural_data, animal_data, session_data


def get_align_vars(neural_data, animal_data):
    sc = [
        neural_data[probe_idx]["spike_clusters"]
        for probe_idx, probe in enumerate(probes)
    ]
    st = [
        neural_data[probe_idx]["spike_times"] for probe_idx, probe in enumerate(probes)
    ]
    srate = neural_data[0][
        "sampling_rate"
    ]  # hardcode as 0 probe idx because it should be same across all probes
    frame_rate = neural_data[0]["frame_rate"]
    apsyncdata = neural_data[0]["apsyncdata"]
    regions = [animal_data[f"{probe}_regions"] for probe in probes]

    return sc, st, srate, frame_rate, apsyncdata, regions


def get_unit_spike_times(spike_times, spike_clusters, neural_data, regions):
    unit_spike_times = {}
    for probe_idx, probe in enumerate(probes):
        for region in regions[probe_idx]:
            # unit_spike_times['ACC'][0] retrieves the spike times for the first unit in ACC
            unit_spike_times[region] = get_cluster_spike_times(
                spike_times[probe_idx],
                spike_clusters[probe_idx],
                good_unit_ids=neural_data[probe_idx][f"{region}_units"],
            )

    return unit_spike_times


def add_prev(trial_data):
    resp = np.array([0] + trial_data["response"].to_list())
    rewd = np.array([0] + trial_data["rewarded"].to_list())

    trial_data["response_prev"] = resp[:-1]
    trial_data["rewarded_prev"] = rewd[:-1]

    trial_data = trial_data.iloc[1:]

    return trial_data


def add_strat(trial_data, session_data):
    trial_data["strategy"] = np.select(
        [
            trial_data["iblock"].isin(session_data["MBblocks"]),
            trial_data["iblock"].isin(session_data["MFblocks"]),
        ],
        [1, -1],
        0,
    )
    return trial_data


def get_trial_mask(trial_data, strategy_only=True, reward_only=False):
    mask_resp = ~np.isnan(
        trial_data["response_time"]
    )  # always only consider for trials where there was a response
    mask = mask_resp

    if reward_only:
        mask = (mask) & (trial_data["rewarded"])

    if strategy_only:
        mask = (mask) & (~(trial_data["strategy"] == 0))

    return mask


# SVD
def get_svd_df(subj_id, sess_id, trial_data, session_data, **trial_idx_kwargs):
    trial_start = session_data["events"]["event_timestamps"][0]
    movie_frame = session_data["events"]["event_timestamps"][1]

    movements = np.load(
        DATA_DIR
        / subj_id
        / sess_id
        / f"{subj_id}_DynamicForaging_{sess_id}_cam1_run000_00000000.npy",
        allow_pickle=True,
    ).item()

    frame_trial_idxs = get_event_trial_idxs(
        trial_start, trial_data, movie_frame, **trial_idx_kwargs
    )

    svd_tavg = np.zeros((len(frame_trial_idxs), 200))
    for i, (start, end) in enumerate(frame_trial_idxs):
        svd_tavg[i] = movements["SVT"][:, start:end].mean(axis=1)

    svd_df = pd.DataFrame(svd_tavg, columns=[f"SVD_{i}" for i in range(200)])

    return svd_df


def get_licks_df(subj_id, sess_id, trial_data, session_data, **trial_idx_kwargs):
    trial_start = session_data["events"]["event_timestamps"][0]

    licks = {
        "left": session_data["events"]["event_timestamps"][3],
        "right": session_data["events"]["event_timestamps"][2],
    }

    n_licks = {
        f"n_{side}_licks": np.diff(
            get_event_trial_idxs(trial_start, trial_data, ts, **trial_idx_kwargs),
            axis=1,
        ).ravel()
        for side, ts in licks.items()
    }

    licks_df = pd.DataFrame(n_licks)

    return licks_df


def get_event_trial_idxs(
    trial_start,
    trial_data,
    movie_frame,
    alignment="choice",
    tpre=None,
    tpost=None,
    full_trial=False,
):

    if full_trial:
        frame_trial_idxs = np.zeros((len(trial_start), 2), dtype=np.int32)
        trial_start = np.append(
            trial_start, trial_start[-1] + trial_data["outcome_time"].iloc[-1]
        )  # TODO

        for trial_i in range(frame_trial_idxs.shape[0]):
            # do same epoch
            start_idx = np.searchsorted(movie_frame, trial_start[trial_i])
            end_idx = np.searchsorted(movie_frame, trial_start[trial_i + 1])

            frame_trial_idxs[trial_i] = (start_idx, end_idx)

    else:
        if alignment == "choice":
            ts = trial_data["trial_start_time"] + trial_data["response_time"]  # s
        else:
            raise NotImplementedError(
                f"alignment {alignment} is not currently implemented"
            )

        trial_idxs = np.vstack((ts - tpre, ts + tpost)).T

        frame_trial_idxs = np.zeros((len(trial_idxs), 2), dtype=np.int32)

        for trial_i, (trial_pre, trial_post) in enumerate(trial_idxs):
            start_idx = np.searchsorted(movie_frame, trial_pre)
            end_idx = np.searchsorted(movie_frame, trial_post)

            frame_trial_idxs[trial_i] = (start_idx, end_idx)

    return frame_trial_idxs


# PR
def get_pr(psths, regions, num_units):
    pr = (
        np.array([(psths[reg].sum(axis=2)).sum(axis=0) for reg in regions]).sum(0)
        / num_units
    )
    return pr


# PSTHS
def get_psths(
    unit_spike_times,
    trial_data,
    session_data,
    regions,
    tpre=2,
    tpost=2,
    binwidth_ms=50,
    alignment="choice",
    trial_start_pre=0,  # can be > 0 to account for alignment to some time before trial start
    balance=True,
    reward_only=True,
    do_rem_zstd=True,
    shuffle=False,
    prev_filter=True,
    mode="new",
):
    mask_resp = (
        ~np.isnan(trial_data["response_time"])
        if ("response_prev" not in trial_data.columns or not prev_filter)
        else (~np.isnan(trial_data["response_time"]))
        & (~trial_data["response_prev"] == 0)
    )  # account for trials where there was no response
    mask_reward = trial_data["rewarded"]

    mask = (mask_resp) & (mask_reward) if reward_only else (mask_resp)
    assert np.mean(mask) == 1  # the trial data passed in should be clean already

    idx = trial_data[
        mask
    ].index  # np.where(mask)[0] if 'response_prev' not in trial_data.columns else np.where(mask)[0] + 1

    if alignment == "choice":
        ts = (
            trial_data["trial_start_time"][mask] + trial_data["response_time"][mask]
        )  # s
    elif alignment == "trial_start":
        ts = trial_data["trial_start_time"][mask] - trial_start_pre
    elif alignment == "reward":
        try:
            ts = trial_data["trial_start_time"][mask] + trial_data["outcome_time"][mask]
        except KeyError:
            if mode == "old":
                ts = (
                    trial_data["trial_start_time"][mask]
                    + trial_data["response_time"][mask]
                    + 0.2
                )
            else:
                raise KeyError
    else:
        raise ValueError(f"{alignment} alignment not implemented yet")

    psths = {}

    tasks = [(reg, unit) for reg in regions for unit in unit_spike_times[reg]]

    psths_all = np.squeeze(
        Parallel(n_jobs=8)(
            delayed(_compute_spike_count_first)(
                event_times=ts,
                spike_times=unit,
                pre_seconds=tpre,
                post_seconds=tpost,
                binwidth_ms=binwidth_ms,
            )
            for _, unit in tasks
        )
    )

    _, tbin_edges, _ = compute_spike_count(ts, tasks[0][1], tpre, tpost, binwidth_ms)

    # psth_matrix, timebin_edges, event_index = compute_spike_count(
    #     event_times=ts, spike_times=tasks[0][1], pre_seconds=tpre+0.001, post_seconds=tpost+0.001, binwidth_ms=binwidth_ms,
    # )

    idx = 0
    for reg in regions:
        n = len(unit_spike_times[reg])
        psths[reg] = psths_all[idx : idx + n]
        idx += n

    if do_rem_zstd:
        [psths], units_to_rem = rem_zstd([psths], regions)
        return psths, mask, units_to_rem
    return psths, mask, tbin_edges  # (psth_matrix, timebin_edges, event_index)


def _compute_spike_count_first(*args, **kwargs):
    return compute_spike_count(*args, **kwargs)[0]


def get_zstd_units(psths_all, regions):
    from collections import defaultdict

    units_to_rem = defaultdict(list)
    for region in regions:
        for psths_ in psths_all:
            # delete all units that have a std of 0 for the signal (i.e., psth is a constant line, slope 0)
            signal_stds = np.std(np.mean(psths_[region], axis=1), axis=1)
            units_to_rem[region].extend(np.where(signal_stds == 0)[0])
            # print(f"{region}.1, {len(np.where(signal_stds == 0)[0])}")

            # delete all units that have a std of 0 for the noise
            noise_stds = np.array(
                [
                    np.std(
                        np.concatenate(
                            [psth - np.mean(psths_a, axis=0) for psth in psths_a]
                        )
                    )
                    for psths_a in psths_[region]
                ]
            )
            units_to_rem[region].extend(np.where(noise_stds == 0)[0])
        units_to_rem[region] = np.unique(units_to_rem[region]).astype(
            dtype=np.int32
        )  # filter for unique

    return units_to_rem


def rem_zstd(psths_all, regions):
    units_to_rem = get_zstd_units(psths_all, regions)

    for i in range(len(psths_all)):
        for region in regions:
            # print(f"> {len(units_to_rem[region])}")
            psths_all[i][region] = np.delete(
                psths_all[i][region], units_to_rem[region], axis=0
            )

    return psths_all, units_to_rem


# get conditional psths/choice timestamps
def get_psths_cond(psths, trial_data, mode="both"):
    if mode == "both":
        psths_cond = {
            "left_corr": psths[
                :,
                (trial_data["response"] == 1) & (trial_data["rewarded"] == 1),
            ],
            "right_corr": psths[
                :,
                (trial_data["response"] == -1) & (trial_data["rewarded"] == 1),
            ],
            "left_incorr": psths[
                :,
                (trial_data["response"] == 1) & (trial_data["rewarded"] == 0),
            ],
            "right_incorr": psths[
                :,
                (trial_data["response"] == -1) & (trial_data["rewarded"] == 0),
            ],
        }
    elif mode == "response":
        psths_cond = {
            "left": psths[:, (trial_data["response"] == 1)],
            "right": psths[:, (trial_data["response"] == -1)],
        }
    elif mode == "rewarded":
        psths_cond = {
            "corr": psths[:, (trial_data["rewarded"] == 1)],
            "incorr": psths[:, (trial_data["rewarded"] == 0)],
        }
    elif mode == "strategy":
        psths_cond = {
            "mb": psths[:, (trial_data["strategy"] == 1)],
            "mf": psths[:, (trial_data["strategy"] == -1)],
        }
    else:
        raise NotImplementedError(
            "valid arguments for mode are 'response,' 'rewarded,' 'both,' and 'strategy.'"
        )
    return psths_cond


def get_choice_ts(trial_data, mode="both"):
    lc_mask = (trial_data.response == 1) & (trial_data.rewarded)
    rc_mask = (trial_data.response == -1) & (trial_data.rewarded)
    li_mask = (trial_data.response == 1) & (~trial_data.rewarded)
    ri_mask = (trial_data.response == -1) & (~trial_data.rewarded)

    if mode == "both":
        choice_ts = {
            "left_corr": trial_data[lc_mask]["trial_start_time"]
            + trial_data[lc_mask]["response_time"],
            "right_corr": trial_data[rc_mask]["trial_start_time"]
            + trial_data[rc_mask]["response_time"],
            "left_incorr": trial_data[li_mask]["trial_start_time"]
            + trial_data[li_mask]["response_time"],
            "right_incorr": trial_data[ri_mask]["trial_start_time"]
            + trial_data[ri_mask]["response_time"],
        }
    elif mode == "response":
        choice_ts = {
            "left": trial_data[(trial_data.response == 1)]["trial_start_time"]
            + trial_data[trial_data.response == 1]["response_time"],
            "right": trial_data[trial_data.response == -1]["trial_start_time"]
            + trial_data[trial_data.response == -1]["response_time"],
        }
    elif mode == "rewarded":
        choice_ts = {
            "corr": trial_data[(lc_mask) | (rc_mask)]["trial_start_time"]
            + trial_data[(lc_mask) | (rc_mask)]["response_time"]
            + 0.5,
            "incorr": trial_data[(li_mask) | (ri_mask)]["trial_start_time"]
            + trial_data[(li_mask) | (ri_mask)]["response_time"]
            + 0.5,
        }
    elif mode == "strategy":
        choice_ts = {
            "mb": trial_data[(trial_data.strategy == 1)]["trial_start_time"]
            + trial_data[(trial_data.strategy == 1)]["response_time"],
            "mf": trial_data[(trial_data.strategy == -1)]["trial_start_time"]
            + trial_data[(trial_data.strategy == -1)]["response_time"],
        }
    else:
        raise NotImplementedError(
            "valid arguments for mode are 'response,' 'rewarded,' 'both,' and 'strategy.'"
        )
    return choice_ts


# ROBS
def get_tavg_sc_cond(
    robs, trial_data, cond, robs_to_subtract=None, subtract_robs=False
):
    if cond == "response":
        left_mask = trial_data.response == 1
        right_mask = trial_data.response == -1

        if subtract_robs:
            robs = robs - robs_to_subtract
        sc_tavg = {
            "left": robs[left_mask].mean(axis=0),
            "right": robs[right_mask].mean(axis=0),
        }
    elif cond == "rewarded":
        corr_mask = trial_data.rewarded == 1
        incorr_mask = trial_data.rewarded == 0

        if subtract_robs:
            robs = robs - robs_to_subtract

        sc_tavg = {
            "corr": robs[corr_mask].mean(axis=0),
            "incorr": robs[incorr_mask].mean(axis=0),
        }
    return sc_tavg


def get_strategy_filter_idxs(
    trial_data,
    strategy="both",
    balance_strategy=True,  # conditional balancing
    cond_balance=False,
    num_trial_thresh=20,
):
    if not (strategy == "both" or strategy == "mb" or strategy == "mf"):
        raise ValueError("valid values for strategy are 'both', 'mb', and 'mf'")

    mb_mask = trial_data["strategy"] == 1
    mf_mask = trial_data["strategy"] == -1

    if balance_strategy:
        if cond_balance:
            # print("hello")
            mb_cond_masks = {
                "left_corr": (mb_mask)
                & (trial_data["response"] == 1)
                & (trial_data["rewarded"] == 1),
                "right_corr": (mb_mask)
                & (trial_data["response"] == -1)
                & (trial_data["rewarded"] == 1),
                "left_incorr": (mb_mask)
                & (trial_data["response"] == 1)
                & (trial_data["rewarded"] == 0),
                "right_incorr": (mb_mask)
                & (trial_data["response"] == -1)
                & (trial_data["rewarded"] == 0),
            }

            mf_cond_masks = {
                "left_corr": (mf_mask)
                & (trial_data["response"] == 1)
                & (trial_data["rewarded"] == 1),
                "right_corr": (mf_mask)
                & (trial_data["response"] == -1)
                & (trial_data["rewarded"] == 1),
                "left_incorr": (mf_mask)
                & (trial_data["response"] == 1)
                & (trial_data["rewarded"] == 0),
                "right_incorr": (mf_mask)
                & (trial_data["response"] == -1)
                & (trial_data["rewarded"] == 0),
            }

            num_trials_cond = [
                min(mb_cond_masks[key].sum(), mf_cond_masks[key].sum())
                for key in mb_cond_masks
            ]
            num_trial_strategy = np.sum(num_trials_cond)

            if (
                (strategy == "mb" or strategy == "mf")
                and num_trial_strategy < num_trial_thresh
            ) or ((strategy == "both") and 2 * num_trial_strategy < num_trial_thresh):
                raise RuntimeError("not enough trials after balancing")

            if strategy == "both" or strategy == "mb":
                idxs_subsamp_mb = []
            if strategy == "both" or strategy == "mf":
                idxs_subsamp_mf = []

            for i, cond in enumerate(mb_cond_masks):
                if strategy == "both" or strategy == "mb":
                    idxs_subsamp_mb.extend(
                        np.random.choice(
                            np.where(mb_cond_masks[cond])[0],
                            num_trials_cond[i],
                            replace=False,
                        )
                    )
                if strategy == "both" or strategy == "mf":
                    idxs_subsamp_mf.extend(
                        np.random.choice(
                            np.where(mf_cond_masks[cond])[0],
                            num_trials_cond[i],
                            replace=False,
                        )
                    )

            if strategy == "both":
                idxs_subsamp_mb = np.sort(idxs_subsamp_mb)
                idxs_subsamp_mf = np.sort(idxs_subsamp_mf)
                idxs_subsamp = np.sort(
                    np.concatenate((idxs_subsamp_mb, idxs_subsamp_mf))
                )

                return {
                    "both": idxs_subsamp,
                    "mb": idxs_subsamp_mb,
                    "mf": idxs_subsamp_mf,
                }

            elif strategy == "mb":
                idxs_subsamp_mb = np.sort(idxs_subsamp_mb)

                return {"both": None, "mb": idxs_subsamp_mb, "mf": None}

            elif strategy == "mf":
                idxs_subsamp_mf = np.sort(idxs_subsamp_mf)

                return {"both": None, "mb": None, "mf": idxs_subsamp_mf}

        # no cond balancing, just trial count
        else:
            mb_mask = trial_data["strategy"] == 1
            mf_mask = trial_data["strategy"] == -1

            num_trials = min(mb_mask.sum(), mf_mask.sum())

            return {
                "mb": np.sort(
                    np.random.choice(np.where(mb_mask)[0], num_trials, replace=False)
                ),
                "mf": np.sort(
                    np.random.choice(np.where(mf_mask)[0], num_trials, replace=False)
                ),
            }
    else:
        # print("3")
        return {
            "both": np.arange(len(trial_data)),
            "mb": np.where(mb_mask)[0],
            "mf": np.where(mf_mask)[0],
        }


def get_robs(psths, regions, norm=True):
    robs = (
        np.concatenate([np.sum(psths[region], axis=2) for region in regions]).T
        # ** 0.5
    )

    if norm:
        s = np.std(robs, axis=0) + 1e-10
        mu = np.mean(robs, axis=0)
        robs = (robs - mu) / s
    return robs


def get_reg_keys(psths, regions):
    reg_mask = np.concatenate(
        [np.repeat(i, len(psths[region])) for i, region in enumerate(regions)]
    )
    reg_idxs = {reg: np.where(reg_mask == i)[0] for i, reg in enumerate(regions)}
    return reg_idxs


def get_dm(trial_data, tv_keys, add_svd, num_svd, add_licks, num_tents, do_ohe=False):
    # task variables
    # non movement
    if tv_keys is not None:
        if do_ohe:
            ohe = OHE().fit(trial_data[tv_keys])
            tvs = np.array(ohe.transform(trial_data[tv_keys]).todense())
        else:
            tvs = trial_data[tv_keys]
            for k in tv_keys:
                if "rewarded" in k:
                    tvs[k] = tvs[k].replace(0, -1)
            tvs = np.array(tvs, dtype="float32")
            # tvs = np.array(zscore(tvs))
        # tvs = zscore(tvs, axis=0)  # FLAG
    else:
        tvs = None

    # movement
    svd_names = []
    if add_svd:
        svd_keys = [f"SVD_{i}" for i in range(num_svd)]
        tvs_svd = zscore(trial_data[svd_keys], axis=0)
        tvs = np.hstack((tvs, tvs_svd)) if tvs is not None else tvs_svd
        svd_names = [f"svd_{i}" for i in range(num_svd)]

    lick_names = []
    if add_licks:
        lick_keys = [f"n_{side}_licks" for side in ["left", "right"]]
        # tvs_licks = zscore(trial_data[lick_keys], axis=0)
        tvs_licks = trial_data[lick_keys]
        tvs = np.hstack((tvs, tvs_licks)) if tvs is not None else tvs_licks
        lick_names = lick_keys

    # tents
    num_trials = trial_data.shape[0]
    xs = np.linspace(0, num_trials - 1, num_tents)
    tents = tent_basis_generate(xs)

    # dm
    dm = np.hstack((tents, tvs))

    if do_ohe:
        tv_names = ohe.get_feature_names_out() if tv_keys is not None else []
        for i, tv_name in enumerate(tv_names):
            tv_names[i] = tv_name_map[tv_name]
    else:
        tv_names = tv_keys

    dm_names = np.concatenate(
        (
            [f"tents_{i}" for i in range(tents.shape[1])],
            tv_names,
            svd_names,
            lick_names,
        )
    )

    dm_idxs = {dm_name: i for i, dm_name in enumerate(dm_names)}
    return dm, (tents, tvs), (dm_names, dm_idxs)


def get_encoder_io(
    psths,
    trial_data,
    regions,
    norm=True,
    num_tents=10,
    tv_keys=["response", "rewarded", "block_side", "response_prev", "rewarded_prev"],
    add_svd=True,
    num_svd=10,
    add_licks=True,
    binwidth_ms=25,
    do_ohe=False,
):
    # neural activity (Y/O)
    robs = get_robs(psths, regions, norm)
    reg_idxs = get_reg_keys(psths, regions)

    # trial data (X/I)

    dm, (tents, tvs), (dm_names, dm_idxs) = get_dm(
        trial_data, tv_keys, add_svd, num_svd, add_licks, num_tents, do_ohe=do_ohe
    )
    return tents, tvs, dm, robs, dm_names, dm_idxs, reg_idxs


# BALANCING
# def balance_strategy(trial_data, mb_idx, mf_idx):
#     """
#     balance should be proportional balancing, so:
#         L  R
#         MF 10 20
#         MB 10 20
#     """
#     # balance the mb and mf trial proportionally on the L vs R trials
#     mb_left = mb_idx[trial_data.loc[mb_idx]["rewarded_side"] == "left"]
#     mb_right = mb_idx[trial_data.loc[mb_idx]["rewarded_side"] == "right"]

#     mf_left = mf_idx[trial_data.loc[mf_idx]["rewarded_side"] == "left"]
#     mf_right = mf_idx[trial_data.loc[mf_idx]["rewarded_side"] == "right"]

#     print(len(mb_left), len(mf_left), len(mb_right), len(mf_right))
#     n_left = min(len(mb_left), len(mf_left))
#     n_right = min(len(mb_right), len(mf_right))

#     # randomly select n_left idxs from mb_left and mf_left and same for right, then concat both mb and mf and return
#     mb_idx = np.concatenate(
#         (np.random.choice(mb_left, n_left), np.random.choice(mb_right, n_right))
#     )
#     mf_idx = np.concatenate(
#         (np.random.choice(mf_left, n_left), np.random.choice(mf_right, n_right))
#     )

#     return mb_idx, mf_idx


# UTILS
def rem_low_fr(psths, spike_times, psths_ref=None, thresh=1, binwidth_ms=25):
    psths_lite = {}
    spike_times_lite = {}

    if psths_ref is None:
        psths_ref = psths

    for region in psths.keys():
        frs = np.mean(
            psths_ref[region], axis=(1, 2)
        )  # across trials and bins, what's the avg # spks/bin?
        low_fr_idxs = np.where(frs < (thresh * binwidth_ms / 1000))[
            0
        ]  # thresh*binwidth_ms/1000 -> # spks/bin to get thresh spks/s

        # thresh was hardcoded in the bool op...the messy hardcode strikes again..never forget (1.13.26)
        psths_lite[region] = np.delete(psths[region], low_fr_idxs, axis=0)
        spike_times_lite[region] = [
            spike_times[region][i]
            for i in range(psths[region].shape[0])
            if i not in low_fr_idxs
        ]
    return psths_lite, spike_times_lite
