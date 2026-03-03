import numpy as np
from joblib import Parallel, delayed
import spks.event_aligned as ea

def get_nspikes_trial(unit_spike_times, trial_data, trial_dur_s, regions):
    boundary_times = trial_data['task_start_time'].tolist() + [trial_dur_s]
    n_spikes_trial = Parallel(n_jobs=8, backend='loky')(delayed(lambda i: 
      [get_nspikes_unit(unit_spike_times[region][unit_idx], boundary_times[i], boundary_times[i+1]) 
       for region in regions for unit_idx in range(len(unit_spike_times[region]))])(i) 
       for i in range(len(boundary_times)-1)) 

    # n_spikes_per_trial = [[np.searchsorted(spike_times,boundary_times[i+1],'right')-np.searchsorted(spike_times,boundary_times[i],'left') for spike_times in unit_spike_times] for i in range(len(boundary_times)-1)]
    return np.array(n_spikes_trial)

def get_nspikes_choice(unit_spike_times, trial_data, regions, pre=1, post=1):
    choice_ts = trial_data['task_start_time']+trial_data['response_time'] # s
    
    # n_spikes_choice = Parallel(n_jobs=8, backend='loky')(delayed(lambda i: 
    #   [get_nspikes_unit(unit_spike_times[region][unit_idx], c_ts-pre, c_ts+post) 
    #    for region in regions for unit_idx in range(len(unit_spike_times[region]))])(c_ts) 
    #    for c_ts in range(len(choice_ts))) # iterate through trials
    
    np.squeeze([ea.compute_firing_rate(choice_ts, unit, pre, post, binwidth_ms=(pre+post)*1000)[0] for unit in unit_spike_times[region]]) 
        
    return np.array(n_spikes_choice)

            

def get_nspikes_unit(spike_times_unit, start, end):
    with open('get_nspikes_trial_unit_log.txt', 'wb') as f:
        print("get_nspikes_trial_unit")
    return np.searchsorted(spike_times_unit, end, 'right')-np.searchsorted(spike_times_unit, start, 'left')
