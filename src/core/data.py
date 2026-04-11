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
from spks.utils import get_cluster_spike_times
from damn.alignment import compute_spike_count
from utils.paths import DATA_DIR

shutup.please()

# CONSTANTS
session_pattern = re.compile(r"^\d{8}_\d{6}$")
subject_ids = np.array([subj_id for subj_id in os.listdir(DATA_DIR)])
session_ids = [
    [
        sess_id
        for sess_id in os.listdir(DATA_DIR / subj_id)
        if session_pattern.match(sess_id)
    ]
    for subj_id in subject_ids
]

probes = ["imec0", "imec1"]
bin_size = 0.001  # s
colors_model = {
    "default": "#333333",
    "drift": "#666666",
    "tv": "#E5A400",
    "affine": "#1A4D99",
}

colors_region = {
    "ACC": "#140C6A",
    "DMS": "#7166E9",
    "M2": "#845910",
    "DLS": "#F7E164",
    "M1": "#409B3D",
}

markers_region = {"ACC": "v", "DMS": "^", "M2": "x", "DLS": "*", "M1": "."}


# LOAD DATA
"""PLEASE USE THIS FUNCTION!!!"""


def load_sess(
    subj_id=None,
    sess_id=None,
    subj_idx=None,
    sess_idx=None,
    thresh=1,
    mode="new",
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

    bin_size = 0.001  # 1 ms, or 0.001 s

    if (subj_id is None and subj_idx is None) or (sess_id is None and sess_idx is None):
        raise ValueError("wow all nones?! try again bucko.")
    else:
        if subj_id is None:
            subj_id = subject_ids[subj_idx]
        if sess_id is None:
            if subj_idx is None:
                subj_idx = np.where(subject_ids == subj_id)[0][0]
            sess_id = session_ids[subj_idx][sess_idx]
    print(subj_id, sess_id)
    if mode == "new":
        fpath = DATA_DIR / subj_id / sess_id
        fpath.exists()

        neural_data = pd.read_pickle(fpath / "neural_data.pkl")

        unit_spike_times = {
            region: values["spike_times"] for region, values in neural_data.items()
        }
        session_data = pd.read_pickle(fpath / "session_data.pkl")
        trial_data = pd.read_csv(fpath / "trialdata.csv")
        regions = np.array(list(neural_data.keys()))

        trial_data["trial_start_time"] = session_data["events"].iloc[
            np.where(np.array(session_data["event_labels"]) == "trial_start")[0][0]
        ]["event_timestamps"]
        trial_data["block_side"] = np.where(
            trial_data["current_block_side"] == "left", 1, -1
        )

        trial_dur_s = int(
            np.ceil(
                np.max(
                    [
                        max(unit_spike_times_reg)
                        for reg in regions
                        for unit_spike_times_reg in unit_spike_times[reg]
                    ]
                )
            )
        )  # s
        trial_dur_ms = trial_dur_s * (1 / bin_size)  # ms

        trial_data = add_prev(trial_data)
        trial_data = add_strat(trial_data, session_data)

        # trial_data["trial_start_time"] = trial_data["task_start_time"]
        trial_mask = get_trial_mask(trial_data)
        trial_data = trial_data[trial_mask]

        # trial_data_choice = trial_data[~(trial_data['response']==0) & ~(trial_data['response_prev']==0)]
        # trial_data_choice = trial_data[~(trial_data['response']==0)] # filter for choice made only

        unit_spike_times = rem_low_fr(
            unit_spike_times, trial_dur_ms=trial_dur_ms, thresh=thresh
        )

        return unit_spike_times, trial_data, session_data, regions
    elif mode == "old":
        # load data and set variables needed for aligning spikes to behavioral events
        _, _, trial_data_r, neural_data, animal_data, session_data = load_data_sess(
            subj_idx=subj_idx, sess_idx=sess_idx
        )
        spike_clusters, spike_times, _, _, _, regions = get_align_vars(
            neural_data, animal_data
        )
        unit_spike_times = get_unit_spike_times(
            spike_times, spike_clusters, neural_data, regions
        )
        regions = np.concatenate(regions)

        trial_dur_s = int(
            np.ceil(
                np.max(
                    [
                        max(unit_spike_times_reg)
                        for reg in regions
                        for unit_spike_times_reg in unit_spike_times[reg]
                    ]
                )
            )
        )  # s
        trial_dur_ms = trial_dur_s * (1 / bin_size)  # ms

        trial_data = add_prev(trial_data_r)
        trial_data = add_strat(trial_data, session_data)

        # trial_data_choice = trial_data[~(trial_data['response']==0) & ~(trial_data['response_prev']==0)]
        # trial_data_choice = trial_data[~(trial_data['response']==0)] # filter for choice made only

        unit_spike_times_lite = rem_low_fr(
            unit_spike_times, trial_dur_ms=trial_dur_ms, thresh=thresh
        )  # remove low fr

        trial_data["trial_start_time"] = trial_data["task_start_time"]
        trial_mask = get_trial_mask(trial_data)
        trial_data = trial_data[trial_mask]

        return unit_spike_times_lite, trial_data, session_data, regions
    else:
        raise ValueError("valid values for mode are 'old' and 'new.'")


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


# PR
def get_pr(psths, regions, num_units):
    pr = (
        np.array([psths[reg].sum(axis=0).sum(axis=1) for reg in regions]).sum(0)
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
    get_strategy=False,
    balance=True,
    reward_only=True,
    do_rem_zstd=True,
    shuffle=False,
    prev_filter=True,
):
    if alignment == "choice":
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

        choice_ts = (
            trial_data["trial_start_time"][mask] + trial_data["response_time"][mask]
        )  # s

        if get_strategy:
            mb_idx = trial_data[
                trial_data["iblock"].isin(session_data["MBblocks"]) & (mask)
            ].index
            mf_idx = trial_data[
                trial_data["iblock"].isin(session_data["MFblocks"]) & (mask)
            ].index

            # mb_idx = np.delete(mb_idx, np.where(mb_idx == 0))
            # mf_idx = np.delete(mf_idx, np.where(mf_idx == 0))

            if balance:
                mb_idx, mf_idx = balance_strategy(trial_data, mb_idx, mf_idx)

            if shuffle:
                pool = np.concatenate((mb_idx, mf_idx))
                mb_idx = np.random.choice(pool, len(mb_idx))
                mf_idx = np.random.choice(pool, len(mf_idx))

        psths = {}

        if get_strategy:
            psths_mb = {}
            psths_mf = {}

        for region in regions:
            # dimensions will be cells x trials x time
            psths[region] = np.squeeze(
                [
                    compute_spike_count(
                        choice_ts, unit, tpre, tpost, binwidth_ms / 1000
                    )[0]
                    for unit in unit_spike_times[region]
                ]
            )
            # print(len(psths[region]))
            if get_strategy:
                psths_mb[region] = np.squeeze(
                    [
                        compute_spike_count(
                            choice_ts.loc[mb_idx], unit, tpre, tpost, binwidth_ms / 1000
                        )[0]
                        for unit in unit_spike_times[region]
                    ]
                )
                psths_mf[region] = np.squeeze(
                    [
                        compute_spike_count(
                            choice_ts.loc[mf_idx], unit, tpre, tpost, binwidth_ms / 1000
                        )[0]
                        for unit in unit_spike_times[region]
                    ]
                )

        if get_strategy:
            if do_rem_zstd:
                [psths, psths_mb, psths_mf], units_to_rem = rem_zstd(
                    [psths, psths_mb, psths_mf], regions
                )
            return psths, psths_mb, psths_mf, idx, mb_idx, mf_idx, mask, units_to_rem
        else:
            if do_rem_zstd:
                [psths], units_to_rem = rem_zstd([psths], regions)
            return psths, mask, units_to_rem
    else:
        return NotImplementedError(
            f"ERROR: not yet implemented for alignment {alignment}"
        )


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
def get_psths_cond(psths, trial_data, trial_mask, mode="both"):
    if mode == "both":
        psths_cond = {
            "left_corr": psths[
                :,
                (trial_data[trial_mask]["response"] == 1)
                & (trial_data[trial_mask]["rewarded"] == 1),
            ],
            "right_corr": psths[
                :,
                (trial_data[trial_mask]["response"] == -1)
                & (trial_data[trial_mask]["rewarded"] == 1),
            ],
            "left_incorr": psths[
                :,
                (trial_data[trial_mask]["response"] == 1)
                & (trial_data[trial_mask]["rewarded"] == 0),
            ],
            "right_incorr": psths[
                :,
                (trial_data[trial_mask]["response"] == -1)
                & (trial_data[trial_mask]["rewarded"] == 0),
            ],
        }
    elif mode == "response":
        psths_cond = {
            "left": psths[:, (trial_data[trial_mask]["response"] == 1)],
            "right": psths[:, (trial_data[trial_mask]["response"] == -1)],
        }
    elif mode == "rewarded":
        psths_cond = {
            "corr": psths[:, (trial_data[trial_mask]["rewarded"] == 1)],
            "incorr": psths[:, (trial_data[trial_mask]["rewarded"] == 0)],
        }
    else:
        raise NotImplementedError(
            "valid arguments for mode are 'response,' 'rewarded,' and 'both.'"
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
            + trial_data[(lc_mask) | (rc_mask)]["response_time"],
            "incorr": trial_data[(li_mask) | (ri_mask)]["trial_start_time"]
            + trial_data[(li_mask) | (ri_mask)]["response_time"],
        }
    else:
        raise NotImplementedError(
            "valid arguments for mode are 'response,' 'rewarded,' and 'both.'"
        )
    return choice_ts


# BALANCING
def balance_strategy(trial_data, mb_idx, mf_idx):
    """
    balance should be proportional balancing, so:
        L  R
        MF 10 20
        MB 10 20
    """
    # balance the mb and mf trial proportionally on the L vs R trials
    mb_left = mb_idx[trial_data.loc[mb_idx]["rewarded_side"] == "left"]
    mb_right = mb_idx[trial_data.loc[mb_idx]["rewarded_side"] == "right"]

    mf_left = mf_idx[trial_data.loc[mf_idx]["rewarded_side"] == "left"]
    mf_right = mf_idx[trial_data.loc[mf_idx]["rewarded_side"] == "right"]

    print(len(mb_left), len(mf_left), len(mb_right), len(mf_right))
    n_left = min(len(mb_left), len(mf_left))
    n_right = min(len(mb_right), len(mf_right))

    # randomly select n_left idxs from mb_left and mf_left and same for right, then concat both mb and mf and return
    mb_idx = np.concatenate(
        (np.random.choice(mb_left, n_left), np.random.choice(mb_right, n_right))
    )
    mf_idx = np.concatenate(
        (np.random.choice(mf_left, n_left), np.random.choice(mf_right, n_right))
    )

    return mb_idx, mf_idx


# UTILS
def rem_low_fr(unit_spike_times, trial_dur_ms, thresh=1):
    unit_spike_times_lite = {}
    for region in unit_spike_times:
        unit_spike_times_lite[region] = rem_low_fr_reg(
            unit_spike_times[region], trial_dur_ms, thresh=thresh
        )
    return unit_spike_times_lite


def rem_low_fr_reg(unit_spike_times_reg, trial_dur_ms, thresh=1):
    frs = np.array(
        [get_mfr(spike_times, trial_dur_ms) for spike_times in unit_spike_times_reg]
    )
    unit_spike_times_reg = [
        unit_spike_times_reg[i]
        for i in range(len(unit_spike_times_reg))
        if i not in np.where(frs < thresh)[0]
    ]  # thresh was hardcoded in the bool op...the messy hardcode strikes again..never forget (1.13.26)
    return unit_spike_times_reg


def get_mfr(spike_times, trial_dur_ms):
    return 1000 * len(spike_times) / trial_dur_ms
