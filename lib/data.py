import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from lib import constants

import spks.event_aligned as ea
from spks.utils import get_cluster_spike_times

import shutup
shutup.please()

# CONSTANTS
subject_ids = ['MM012', 'MM013']
session_ids = [['20231211_172819', '20231218_170114', '20231221_134112', '20231222_145357', '20231225_123125'], ['20231218_175023', '20231220_142345', '20231222_141517', '20231225_130825', '20231227_132538']]
probes = ['imec0', 'imec1']
bin_size = 0.001 # s
colors_model = {"default": "#333333",
                "drift": "#666666",
                "tv": "#E5A400",
                "affine": "#1A4D99"
    }

colors_region = {"ACC": "#140C6A", 
                 "DMS": "#7166E9", 
                 "M2": "#845910", 
                 "DLS": "#F7E164"
}

markers_region = {"ACC": 'v', 
                 "DMS": '^', 
                 "M2": 'x', 
                 "DLS": '*'
}

# LOAD DATA
def load_data(thresh=1):
    trial_data_r = []
    trial_data = []
    session_data = []
    unit_spike_times = []
    regions = []
    
    for subj_idx in range(len(subject_ids)):
        print(f"Subject: {subject_ids[subj_idx]}")
        trial_data_r_subj, trial_data_subj, session_data_subj, unit_spike_times_subj, regions_subj = load_subj(subj_idx, thresh=thresh)
        
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
        trial_data_r_sess, trial_data_sess, session_data_sess, unit_spike_times_lite_sess, regions_sess = load_sess(subj_idx, sess_idx, thresh=thresh)
        trial_data_r.append(trial_data_r_sess)
        trial_data.append(trial_data_sess)
        session_data.append(session_data_sess)
        unit_spike_times.append(unit_spike_times_lite_sess)
        regions.append(regions_sess)
        
    return trial_data_r, trial_data, session_data, unit_spike_times, regions

def load_sess(subj_idx, sess_idx, bin_size=0.001, thresh=1):
    # load data and set variables needed for aligning spikes to behavioral events
    _, _, trial_data_r, neural_data, animal_data, session_data = load_data_sess(subj_idx=subj_idx, sess_idx=sess_idx)
    spike_clusters, spike_times, _, _, _, regions = get_align_vars(neural_data, animal_data) 
    unit_spike_times = get_unit_spike_times(spike_times, spike_clusters, neural_data, regions)
    regions = np.concatenate(regions)

    trial_dur_s = int(np.ceil(np.max([max(unit_spike_times_reg) for reg in regions for unit_spike_times_reg in unit_spike_times[reg]]))) # s
    trial_dur_ms = trial_dur_s * (1/bin_size) # ms

    trial_data = add_prev(trial_data_r)
    trial_data = add_strat(trial_data, session_data)
    
    # trial_data_choice = trial_data[~(trial_data['response']==0) & ~(trial_data['response_prev']==0)]
    # trial_data_choice = trial_data[~(trial_data['response']==0)] # filter for choice made only
    
    unit_spike_times_lite = rem_low_fr(unit_spike_times, trial_dur_ms=trial_dur_ms, thresh=thresh) # remove low fr
    
    return trial_data_r, trial_data, session_data, unit_spike_times_lite, regions

def load_data_sess(subj_idx, sess_idx):
    fpath_data = f"../../data-np/{subject_ids[subj_idx]}/{session_ids[subj_idx][sess_idx]}"
    
    riglog = np.load(f'{fpath_data}/riglog.npy',allow_pickle='TRUE').item() #log from the rig of events that happened
    corrected_onsets = np.load(f'{fpath_data}/corrected_onsets.npy',allow_pickle='TRUE').item() #TTL timestamps of behavioral events
    trial_data = pd.read_csv(f'{fpath_data}/trialdata.csv') #information about task on each trial

    #neural_data is a dict which has information about the spike-sorted neural data
    neural_data = []
    for probe in probes:
        with open(f'{fpath_data}/{probe}_neural_data.pkl', 'rb') as f:
            neural_data.append(pickle.load(f))

    #animal_data is a dict which has information about the animal/implant locations
    with open(f'{fpath_data}/animal_data.pkl', 'rb') as f:
        animal_data = pickle.load(f)

    #session_data is a dict which has information about the session, such as the strategy IDs for each block
    with open(f'{fpath_data}/session_data.pkl', 'rb') as f:
        session_data = pickle.load(f)
    
    return riglog, corrected_onsets, trial_data, neural_data, animal_data, session_data

def get_align_vars(neural_data, animal_data):
    sc = [neural_data[probe_idx]['spike_clusters'] for probe_idx, probe in enumerate(probes)]
    st = [neural_data[probe_idx]['spike_times'] for probe_idx, probe in enumerate(probes)]
    srate = neural_data[0]['sampling_rate'] # hardcode as 0 probe idx because it should be same across all probes
    frame_rate = neural_data[0]['frame_rate']
    apsyncdata = neural_data[0]['apsyncdata']
    regions = [animal_data[f'{probe}_regions'] for probe in probes]

    return sc, st, srate, frame_rate, apsyncdata, regions

def get_unit_spike_times(spike_times, spike_clusters, neural_data, regions):
    unit_spike_times = {}
    for probe_idx, probe in enumerate(probes):
        for region in regions[probe_idx]:
            # unit_spike_times['ACC'][0] retrieves the spike times for the first unit in ACC
            unit_spike_times[region] = get_cluster_spike_times(spike_times[probe_idx], spike_clusters[probe_idx], good_unit_ids=neural_data[probe_idx][f'{region}_units'])
    
    return unit_spike_times

def add_prev(trial_data):
    resp = np.array([0] + trial_data['response'].to_list())
    rewd = np.array([0] + trial_data['rewarded'].to_list())

    trial_data['response_prev'] = resp[:-1]
    trial_data['rewarded_prev'] = rewd[:-1]
    
    trial_data = trial_data.iloc[1:]
    
    return trial_data

def add_strat(trial_data, session_data):
    trial_data['is_mb'] = trial_data['iblock'].isin(session_data['MBblocks']).astype(int)
    return trial_data

# PSTHS
def get_psths(unit_spike_times, trial_data, session_data, regions, tpre=2, tpost=2, binwidth_ms=50, alignment='choice', balance=True, reward_only=True, do_rem_zstd=True, shuffle=False, prev_filter=True):
    if alignment=='choice':
        mask_resp = ~np.isnan(trial_data['response_time']) if ('response_prev' not in trial_data.columns or prev_filter==False) else (~np.isnan(trial_data['response_time'])) & (~trial_data['response_prev']==0) # account for trials where there was no response
        mask_reward = trial_data['rewarded']

        mask = (mask_resp) & (mask_reward) if reward_only else (mask_resp)
        idx = trial_data[mask].index # np.where(mask)[0] if 'response_prev' not in trial_data.columns else np.where(mask)[0] + 1

        choice_ts = trial_data['task_start_time'][mask]+trial_data['response_time'][mask] # s

        mb_idx = trial_data[trial_data['iblock'].isin(session_data['MBblocks']) & (mask)].index
        mf_idx = trial_data[trial_data['iblock'].isin(session_data['MFblocks']) & (mask)].index
        
        # mb_idx = np.delete(mb_idx, np.where(mb_idx == 0))
        # mf_idx = np.delete(mf_idx, np.where(mf_idx == 0))

        if balance: mb_idx, mf_idx = balance_strategy(trial_data, mb_idx, mf_idx)

        if shuffle:
            pool = np.concatenate((mb_idx, mf_idx))
            mb_idx = np.random.choice(pool, len(mb_idx))
            mf_idx = np.random.choice(pool, len(mf_idx))

        psths = {}
        psths_mb = {}
        psths_mf = {}

        for region in regions:
            # dimensions will be cells x trials x time
            psths[region] = np.squeeze([ea.compute_firing_rate(choice_ts, unit, tpre, tpost, binwidth_ms)[0] for unit in unit_spike_times[region]]) 
            # print(len(psths[region]))
            psths_mb[region] = np.squeeze([ea.compute_firing_rate(choice_ts.loc[mb_idx], unit, tpre, tpost, binwidth_ms)[0] for unit in unit_spike_times[region]])
            psths_mf[region] = np.squeeze([ea.compute_firing_rate(choice_ts.loc[mf_idx], unit, tpre, tpost, binwidth_ms)[0] for unit in unit_spike_times[region]])
        
        if do_rem_zstd: [psths, psths_mb, psths_mf] = rem_zstd([psths, psths_mb, psths_mf], regions)

    else:
        return NotImplementedError(f"ERROR: not yet implemented for alignment {alignment}")
    return psths, psths_mb, psths_mf, idx, mb_idx, mf_idx, mask

def rem_zstd(psths_all, regions):
    units_to_rem = get_zstd_units(psths_all, regions)

    for i in range(len(psths_all)):
        for region in regions:
            # print(f"> {len(units_to_rem[region])}")
            psths_all[i][region] = np.delete(psths_all[i][region], units_to_rem[region], axis=0)

    return psths_all

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
            noise_stds = np.array([np.std(np.concatenate([psth - np.mean(psths_a, axis=0) for psth in psths_a])) for psths_a in psths_[region]])
            units_to_rem[region].extend(np.where(noise_stds == 0)[0])
            # print(f"{region}.2, {len(np.where(noise_stds == 0)[0])}")
        
    return units_to_rem

def plot_psths(psths, region, nrows=5, ncols=5, pre=2, binwidth_ms=50):
    np.random.seed(0)
    bins_per_sec = 1000/binwidth_ms

    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(constants.SUBPLOT_SIDELEN_MED*5, constants.SUBPLOT_SIDELEN_MED*5))
    idxs = np.random.choice(len(psths[region]), size=nrows*ncols)

    for i, ax in enumerate(axes.flat):
        ax.plot(np.mean(psths[region][idxs[i]], axis=0), c='#333333')
        ax.axvline(x=(bins_per_sec*pre), c="#6366FF", linestyle='--')
        tick_positions = ax.get_xticks()
        tick_labels = [f"{((pos)-(bins_per_sec*pre))/(bins_per_sec)}" for pos in tick_positions]

        ax.set_xticks(tick_positions); ax.set_xticklabels(tick_labels) 
        ax.set_xlabel("Time (s)")
        ax.set_title(f"Unit {idxs[i]}")
        
        # ax.set_xticklabels([]); ax.set_xticks([], []); ax.set_yticklabels([]); ax.set_yticks([], [])
    fig.suptitle(f"{region} Sample Unit PSTHs")
    plt.tight_layout()
    
def get_psths_lr(psths, trial_data, region, pre=2, post=2, binsize_ms=50):
    mask_resp = ~np.isnan(trial_data['response_time']) # account for trials where there was no response
    mask_reward = trial_data['rewarded']

    mask = (mask_resp) & (mask_reward)
    idxs = np.where(mask)[0]
    
    # use idx when querying the trial (and note the symmetry between np.where to get the idx and iloc when using the idx)
    # but, you want the index of the idx when querying psths, becuase psths has already filtered for idxs only
    l_idx = [i for i, idx in enumerate(idxs) if trial_data['rewarded_side'].iloc[idx]=='left'] # this is correct, just think about it for five minutes
    r_idx = [i for i, idx in enumerate(idxs) if trial_data['rewarded_side'].iloc[idx]=='right']

    psth = np.mean(psths[region], axis=1) 
    psth_l = np.mean(psths[region][:,l_idx], axis=1) 
    psth_r = np.mean(psths[region][:,r_idx], axis=1) 
    
    psth_std_l = np.std(psths[region][:,l_idx], axis=1) 
    psth_std_r = np.std(psths[region][:,r_idx], axis=1) 
    
    return psth, psth_l, psth_r, psth_std_l, psth_std_r

def plot_grand_cond_avg(psths, trial_data, region, pre=2, post=2, binsize_ms=50):
    mask_resp = ~np.isnan(trial_data['response_time']) # account for trials where there was no response
    mask_reward = trial_data['rewarded']

    mask = (mask_resp) & (mask_reward)
    idxs = np.where(mask)[0]
    l_idx = [i for i, idx in enumerate(idxs) if trial_data['rewarded_side'][idx]=='left']
    r_idx = [i for i, idx in enumerate(idxs) if trial_data['rewarded_side'][idx]=='right']

    psth = np.mean(psths, axis=0) 
    psth_l = np.mean(psths[l_idx], axis=0) 
    psth_r = np.mean(psths[r_idx], axis=0) 

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9,3))
    axes[0].plot(psth)
    axes[0].axvline((pre*1000/binsize_ms)+0.5, c='k', linestyle='--')
    axes[0].set_xticks(np.linspace(0,len(psth)+1,5), np.linspace(-pre,post,5))
    axes[0].set_xlabel("Time (s)"); axes[0].set_title("Grand Average")
    
    axes[1].plot(psth_l)
    axes[1].axvline((pre*1000/binsize_ms)+0.5, c='k', linestyle='--')
    axes[1].set_xticks(np.linspace(0,len(psth)+1,5), np.linspace(-pre,post,5))
    axes[1].set_xlabel("Time (s)"); axes[1].set_title(f"Left Choice (n={len(l_idx)})")

    axes[2].plot(psth_r)
    axes[2].axvline((pre*1000/binsize_ms)+0.5, c='k', linestyle='--')
    axes[2].set_xticks(np.linspace(0,len(psth)+1,5), np.linspace(-pre,post,5))
    axes[2].set_xlabel("Time (s)"); axes[2].set_title(f"Right Choice (n={len(r_idx)})")
    
    fig.suptitle(f"{region}")
    plt.show()

# BALANCING
def balance_strategy(trial_data, mb_idx, mf_idx):
    '''
    balance should be proportional balancing, so:
        L  R
        MF 10 20
        MB 10 20
    '''
    # balance the mb and mf trial proportionally on the L vs R trials
    mb_left = mb_idx[trial_data.loc[mb_idx]['rewarded_side']=='left']
    mb_right = mb_idx[trial_data.loc[mb_idx]['rewarded_side']=='right']

    mf_left = mf_idx[trial_data.loc[mf_idx]['rewarded_side']=='left']
    mf_right = mf_idx[trial_data.loc[mf_idx]['rewarded_side']=='right']

    print(len(mb_left),  len(mf_left), len(mb_right), len(mf_right))
    n_left = min(len(mb_left), len(mf_left))
    n_right = min(len(mb_right), len(mf_right))

    # randomly select n_left idxs from mb_left and mf_left and same for right, then concat both mb and mf and return
    mb_idx = np.concatenate((np.random.choice(mb_left, n_left), np.random.choice(mb_right, n_right)))
    mf_idx = np.concatenate((np.random.choice(mf_left, n_left), np.random.choice(mf_right, n_right)))

    return mb_idx, mf_idx

# PLOTTING
def plot_fr_regs(unit_spike_times, regions, bin_size=0.001):
    trial_dur_s = int(np.ceil(np.max([max(unit_spike_times_reg) for reg in regions for unit_spike_times_reg in unit_spike_times[reg]]))) # s
    trial_dur_ms = trial_dur_s * (1/bin_size)
    
    frs = [[get_mfr(spike_times, trial_dur_ms) for spike_times in unit_spike_times[reg]] for reg in regions]
    fig, axes = plt.subplots(ncols=2, nrows=2)
    for i, ax in enumerate(axes.flat):
        ax.hist(frs[i])
        ax.set_title(regions[i]); ax.set_xlabel("Mean Firing Rate (Hz)")
    fig.tight_layout()
    fig.show()

# UTILS
def time2train(spike_times, num_ms):
    spike_train = np.zeros(num_ms)
    spike_train[np.round(spike_times*1000).astype(int)] = 1
    return spike_train

def rem_low_fr(unit_spike_times, trial_dur_ms, thresh=1):
    unit_spike_times_lite = {}
    for region in unit_spike_times:
        unit_spike_times_lite[region] = rem_low_fr_reg(unit_spike_times[region], trial_dur_ms, thresh=thresh)
    return unit_spike_times_lite

def rem_low_fr_reg(unit_spike_times_reg, trial_dur_ms, thresh=1):
    frs = np.array([get_mfr(spike_times, trial_dur_ms) for spike_times in unit_spike_times_reg])
    unit_spike_times_reg = [unit_spike_times_reg[i] for i in range(len(unit_spike_times_reg)) if i not in np.where(frs < thresh)[0]] # thresh was hardcoded in the bool op...the messy hardcode strikes again..never forget (1.13.26)
    return unit_spike_times_reg

def half_gaussian_kernel(size=21, sigma=3, side='right'):
    x = np.linspace(-size//2, size//2, size)
    g = np.exp(-x**2 / (2 * sigma**2))
    if side == 'right':
        g[x < 0] = 0
    elif side == 'left':
        g[x > 0] = 0
    return g / g.sum()

def get_fr(spike_times, binsize_ms=1):
    '''
    returns fr in ms (each val is binsize_ms ms)
    '''

    edges = np.arange(0, max(spike_times), binsize_ms/1000)
    [fr] = binary_spikes([spike_times], edges, kernel=half_gaussian_kernel(side='left')) / (binsize_ms/1000)
    return fr

def get_mfr(spike_times, trial_dur_ms):
    return 1000*len(spike_times)/trial_dur_ms

def s2ms(sec):
    return int(np.ceil(sec*1000))

def ms2s(ms):
    return int(np.ceil(ms/1000))