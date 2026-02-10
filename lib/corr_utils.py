import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scipy.signal import correlate
from scipy.stats import pearsonr as r2

from spks.utils import binary_spikes
from spks import event_aligned as ea

from joblib import Parallel, delayed

from lib import data, constants

# SPIKE-TRIGGERED AVERAGES & SPIKE-TRIGGERED GAINS
def get_sta(spike_times_a, spike_times_b, pre=1, post=2, binwidth_ms=25, n_rasters=np.inf, do_plot=False, title=None, subtitle=None):
    bins_per_sec = 1000/binwidth_ms
    spike_times_seg = [(bins_per_sec*pre)+bins_per_sec*(spike_times_b[np.searchsorted(spike_times_b, spike_time-pre, side='right'):np.searchsorted(spike_times_b, spike_time+post, side='left')]-spike_time)
                       for spike_time in spike_times_a]

    psth, _, _ = ea.compute_spike_count(spike_times_a, spike_times_b, pre_seconds=pre, post_seconds=post, binwidth_ms=binwidth_ms)
    trace = bins_per_sec * np.mean(psth, axis=0)

    if do_plot:
        event_idxs = np.random.choice(np.arange(len(spike_times_seg)), size=min(len(spike_times_seg), n_rasters), replace=False)
       
        fig, ax1 = plt.subplots()

        ea.plot_raster([spike_times for i, spike_times in enumerate(spike_times_seg) if i in event_idxs], ax=ax1)

        ax2 = ax1.twinx()
        ax2.axvline(x=(bins_per_sec*pre), c="#6366FF", linestyle='--')
        ax2.plot(np.linspace(0.5, len(trace)+0.5, len(trace)), trace, c="#0004FF")

        
        tick_positions = ax1.get_xticks()
        #tick_labels = [f"{((pos)-(bins_per_sec*pre))/(bins_per_sec):.1f}" for pos in tick_positions]
        tick_labels = [f"{((pos)-(bins_per_sec*pre))/(bins_per_sec)}" for pos in tick_positions]

        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(tick_labels)

        ax1.set_xlabel("Time (s)"); ax2.set_ylabel("FR (Hz)"); ax1.set_ylabel("Event ID")

        if title==None: 
            title = "Spike-Triggered Activity"
            if not subtitle == None:
                title += f" - {subtitle}"
        fig.suptitle(title)

    return psth, trace

def get_stgs(unit_spike_times, regions, trial_dur, window_ms=50, binwidth_ms=5, do_plot=False):
    window_s = window_ms/1000
    bins_per_s = 1000/binwidth_ms
    start = int(window_s*bins_per_s) - 1


    def get_stgain_reg(spike_times_reg_a, spike_times_reg_b, start):
        return Parallel(n_jobs=-1, backend='loky')(delayed(lambda spike_times_a: [get_stgain(spike_times_a, spike_times_b, start) for spike_times_b in spike_times_reg_b])(spike_times_a) for spike_times_a in spike_times_reg_a) 

    def get_stgain(spike_times_a, spike_times_b, start):
        with open(f"get_stgain_log.txt", "a") as f:
                print(f"running get_stgain", file=f)
        _, trace = get_sta(spike_times_a, spike_times_b, binwidth_ms=binwidth_ms, pre=window_s, post=window_s)
        return ((np.mean(trace[start:]))-np.mean(trace[:start]))/(data.get_mfr(spike_times_b, trial_dur_ms=int(trial_dur)))
    
    spike_triggered_gains_regs = Parallel(n_jobs=-1, backend='loky')(delayed(lambda region_a: [get_stgain_reg(unit_spike_times[region_a], unit_spike_times[region_b], start) for region_b in regions])(region_a) for region_a in regions) 
    
    stgs = {}
    for i, region_a in enumerate(regions):
         for j, region_b in enumerate(regions):
              stgs[f'{region_a}-{region_b}'] = np.array(spike_triggered_gains_regs[i][j])

    if do_plot:
        plot_hmaps_all(stgs, regions, title="Spike-Triggered Gains")
    
    return stgs, spike_triggered_gains_regs

# CROSS-CORRELATION
def ccorr(spike_times_a, spike_times_b, binsize_ms=1, max_time=None, trial_dur_ms=None, method='fft', norm=True, acorr=False, doPlot=False, durPlot=1000, title=None, subtitle=None):
    # spike_train_a = data.time2train(spike_times_a, trial_dur)
    # spike_train_b = data.time2train(spike_times_b, trial_dur)
    from spks.utils import binary_spikes

    binsize_s = binsize_ms / 1000

    if not max_time == None:
        spike_times_a = spike_times_a[np.where(spike_times_a <= max_time)]
        spike_times_b = spike_times_b[np.where(spike_times_b <= max_time)]

    edges = np.arange(0, max(max(spike_times_a), max(spike_times_b)), binsize_s)
    [spike_train_a, spike_train_b] = binary_spikes([spike_times_a, spike_times_b], edges) # / binsize_s

    mfr_bin_a = binsize_ms*data.get_mfr(spike_times_a, trial_dur_ms)/1000
    mfr_bin_b = binsize_ms*data.get_mfr(spike_times_b, trial_dur_ms)/1000

    crosscorr = correlate(spike_train_b-mfr_bin_b, spike_train_a-mfr_bin_a, method=method)
    # crosscorr = fftconvolve(spike_train_a, spike_train_b[::-1], mode='full')
    crosscorr = crosscorr[crosscorr.size//2:] # get the ccorr vals where spike_train_b is stationary and spike_train_a moves rightward
    if norm: crosscorr /= max(crosscorr)

    if acorr: assert 0 in np.where(crosscorr==max(crosscorr)), "No peak at lag 0 for autocorr" #confirm that there is a peak at 0 for autocorr

    if doPlot:
        fig, ax = plt.subplots()
        if not acorr: ax.plot(np.arange(0,durPlot), (crosscorr[0:durPlot])) # note that the peak at 0 is not shown for plotting clarity
        else: ax.plot(np.arange(1,durPlot), (crosscorr[1:durPlot])) 

        tick_loc, _ = plt.xticks(); tick_loc = np.array([loc for loc in tick_loc if loc > 0 and loc <= durPlot])
        ax.set_xlabel("Lag (s)"); ax.set_xticks(tick_loc, tick_loc/1000); plt.ylabel("Cross-Correlation Coefficient")
        

        if title==None: 
            title = "Cross-Correlation" if not acorr else "Auto-Correlation"
            if not subtitle == None:
                title += f" - {subtitle}"
        fig.suptitle(title)

        plt.show()

    return crosscorr

def get_ccorrs(unit_spike_times, regions, binsize_ms=5, dur_ms=50, do_norm=False, trial_dur_ms=None, include_lag_zero=True, do_plot=False):
    def get_ccorr_reg(spike_times_reg_a, spike_times_reg_b, trial_dur_ms):
        return Parallel(n_jobs=-1, backend='loky')(delayed(lambda spike_times_a: [get_ccorr(spike_times_a, spike_times_b, trial_dur_ms) for spike_times_b in spike_times_reg_b])(spike_times_a) for spike_times_a in spike_times_reg_a) 

    def get_ccorr(spike_times_a, spike_times_b, trial_dur_ms):
        with open(f"get_ccorr_log.txt", "a") as f:
                print(f"running get_ccorr", file=f)
        corr = ccorr(spike_times_a, spike_times_b, binsize_ms=binsize_ms, norm=do_norm, trial_dur_ms=trial_dur_ms)
        if include_lag_zero: 
            extremes = np.array([np.max(corr[:int(dur_ms/binsize_ms)]), np.min(corr[:int(dur_ms/binsize_ms)])])
        else: 
            extremes = np.array([np.max(corr[1:int(dur_ms/binsize_ms)]), np.min(corr[1:int(dur_ms/binsize_ms)])])
        return extremes[np.argmax(np.abs(extremes))]

    ccorrs_vals = Parallel(n_jobs=-1, backend='loky')(delayed(lambda region_a: [get_ccorr_reg(unit_spike_times[region_a], unit_spike_times[region_b], trial_dur_ms) for region_b in regions])(region_a) for region_a in regions) 
    
    ccorrs = {}
    for i, region_a in enumerate(regions):
         for j, region_b in enumerate(regions):
              ccorrs[f'{region_a}-{region_b}'] = np.array(ccorrs_vals[i][j])
 

    if do_plot:
         plot_hmaps_all(ccorrs_vals, regions, title="Cross Correlation")
    
    return ccorrs, ccorrs_vals

# R_SC
def get_rscs(unit_spike_times, regions, trial_dur_ms, binsize_ms=10, dur_ms=50, do_norm=False, do_plot=False):
    def get_rsc_reg(spike_times_reg_a, spike_times_reg_b, trial_dur_ms):
        return Parallel(n_jobs=-1, backend='loky')(delayed(lambda spike_times_a: [get_rsc(spike_times_a, spike_times_b, trial_dur_ms) for spike_times_b in spike_times_reg_b])(spike_times_a) for spike_times_a in spike_times_reg_a) 

    def get_rsc(spike_times_a, spike_times_b, trial_dur_ms):
        with open(f"get_rsc_log.txt", "a") as f:
                print(f"running get_rsc", file=f)
        corr = ccorr(spike_times_a, spike_times_b, binsize_ms=binsize_ms, norm=do_norm, trial_dur_ms=trial_dur_ms)
        return np.sum(corr[:int(dur_ms/binsize_ms)]) # i.e., discrete integral

    rscs_vals = Parallel(n_jobs=-1, backend='loky')(delayed(lambda region_a: [get_rsc_reg(unit_spike_times[region_a], unit_spike_times[region_b], trial_dur_ms) for region_b in regions])(region_a) for region_a in regions) 
    
    rscs = {}
    for i, region_a in enumerate(regions):
         for j, region_b in enumerate(regions):
              rscs[f'{region_a}-{region_b}'] = np.array(rscs_vals[i][j])
 

    if do_plot:
         plot_hmaps_all(rscs_vals, regions, title="R_sc")
    
    return rscs, rscs_vals

# FIRING RATE CORRELATION
def frcorr(spike_times_a, spike_times_b, a=None, b=None, binsize_ms=25, doPlot=False, title=None, subtitle=None):

    edges = np.arange(0, max(max(spike_times_a), max(spike_times_b)), binsize_ms/1000)
    [fr_a, fr_b] = binary_spikes([spike_times_a, spike_times_b], edges, kernel=data.half_gaussian_kernel(side='left')) / (binsize_ms/1000)


    fr_corr = r2(fr_a, fr_b)[0] # get the pearson corr
    theta = np.linalg.lstsq(np.c_[fr_a, np.ones_like(fr_a)], fr_b)[0] # fit a line

    if doPlot:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(3*constants.SUBPLOT_SQUARE_SIDELEN, 1*constants.SUBPLOT_SQUARE_SIDELEN))
        axes[0].scatter(fr_a, fr_b, s=0.2, alpha=0.5)
        axes[0].plot(np.linspace(0, max(fr_a), 2), theta[0]*np.linspace(0, max(fr_a), 2)+theta[1], color="#0D00BE")
        axes[1].hist(fr_a, bins=100)
        axes[2].hist(fr_b, bins=100)

        axes[1].set_yscale('log'); axes[2].set_yscale('log')

        axes[0].set_title("Firing Rates"); axes[0].set_xlabel(f"Unit {a}"); axes[0].set_ylabel(f"Unit {b}")
        axes[1].set_title(f"Firing Rate Distribution, Unit {a}"); axes[1].set_xlabel(f"Firing Rate (Hz)"); axes[1].set_ylabel(f"Frequency")
        axes[2].set_title(f"Firing Rate Distribution, Unit {b}"); axes[2].set_xlabel(f"Firing Rate (Hz)"); axes[2].set_ylabel(f"Frequency")

        if title==None: 
            title = f"Firing Rate Correlation, R2 = {np.round(fr_corr, 3)}"
            if not subtitle == None:
                title += f" - {subtitle}"
        fig.suptitle(title)

        plt.show()
    
    return fr_corr

def get_fr_r2s(unit_spike_times, regions=None, binsize_ms=1000, do_plot=False):
    def get_fr_r2_reg(spike_times_reg_a, spike_times_reg_b):
        return Parallel(n_jobs=-1, backend='loky')(delayed(lambda spike_times_a: [get_fr_r2(spike_times_a, spike_times_b) for spike_times_b in spike_times_reg_b])(spike_times_a) for spike_times_a in spike_times_reg_a) 

    def get_fr_r2(spike_times_a, spike_times_b):
        with open(f"get_fr_r2_log.txt", "a") as f:
                print(f"running get_fr_r2", file=f)
        r2 = frcorr(spike_times_a, spike_times_b, binsize_ms=binsize_ms)
        return r2

    fr_r2_vals = Parallel(n_jobs=-1, backend='loky')(delayed(lambda region_a: [get_fr_r2_reg(unit_spike_times[region_a], unit_spike_times[region_b]) for region_b in regions])(region_a) for region_a in regions) 
    
    fr_r2s = {}
    for i, region_a in enumerate(regions):
         for j, region_b in enumerate(regions):
              fr_r2s[f'{region_a}-{region_b}'] = np.array(fr_r2_vals[i][j])
 

    if do_plot:
         plot_hmaps_all(fr_r2_vals, regions, title="Firing Rate R2")
    
    return fr_r2s, fr_r2_vals

def plot_ccorr_frr2(ccorrs, fr_r2s, regions, title=None, subtitle=None):
    fig, axes = plt.subplots(nrows=len(regions), ncols=len(regions), figsize=(len(regions)*constants.SUBPLOT_SQUARE_SIDELEN, len(regions)*constants.SUBPLOT_SQUARE_SIDELEN))

    for i, region_a in enumerate(regions):
        for j, region_b in enumerate(regions):
            ax = axes[i][j]
            ax.scatter(ccorrs[f'{region_a}-{region_b}'], fr_r2s[f'{region_a}-{region_b}'], s=0.5, c='k')
            ax.set_xlabel("Cross-Correlation Coefficient"); ax.set_ylabel("FR R2")
            ax.set_title(f"{region_a} to {region_b}")

    if title==None: 
        title = f"Cross-Correlation Coefficient and Firing Rate Correlation"
        if not subtitle == None:
            title += f" - {subtitle}"
    fig.suptitle(title)

    plt.tight_layout()
    plt.show()

# SPIKE LATENCIES
def get_pair_spike_latencies(spike_times_a, spike_times_b, doPlot=False, title=None, subtitle=None):

    latencies = [spike_times_b[np.searchsorted(spike_times_b, spike_time, side='right'):np.searchsorted(spike_times_b, spike_time+1, side='left')][0] - spike_time 
                 for spike_time in spike_times_a
                 if len(spike_times_b[np.searchsorted(spike_times_b, spike_time, side='right'):np.searchsorted(spike_times_b, spike_time+1, side='left')])>0]
    # latencies = np.array([val for latencies_spike in latencies_spike for val in latencies_spike])
    
    if doPlot:
        fig, ax = plt.subplots()

        ax.hist(latencies, bins=200)

        ax.set_xlabel("Latency (s)"); ax.set_ylabel("Frequency")

        if title==None: 
            title = "Spike Latency Distribution"
            if not subtitle == None:
                title += f" - {subtitle}"
        fig.suptitle(title)

        plt.show()

    return latencies

# PLOTTING
def plot_hmaps_all(hmaps, regions, title=None, vmax=None, cmap="BrBG", subtitle=None):
    n_units = [np.shape(hmaps[i][i])[0] for i in range(len(hmaps))]
    gs_units = [int(r / min(n_units)) for r in n_units]

    fig = plt.figure(figsize=(8, 8), constrained_layout=True)
    gs = GridSpec(nrows=sum(gs_units), ncols=sum(gs_units), figure=fig)

    axes = []
    for i, r_units in enumerate(gs_units):
        for j, c_units in enumerate(gs_units):
            # display heatmap
            ax = fig.add_subplot(gs[sum(gs_units[:i]):sum(gs_units[:i]) + r_units, sum(gs_units[:j]):sum(gs_units[:j]) + c_units])
            if vmax==None: 
                im = ax.imshow(hmaps[i][j], aspect='auto', cmap=cmap)
            else: 
                im = ax.imshow(hmaps[i][j], aspect='auto', vmin=-vmax, vmax=vmax, cmap=cmap)
            axes.append(ax)

            # show y-axis ticks + label on left column only
            if j == 0: ax.set_ylabel(f'{regions[i]}', fontsize=8)
            else: ax.set_yticks([]); ax.set_ylabel('')

            # show x-axis ticks + label on bottom row only
            if i == len(n_units) - 1: ax.set_xlabel(f'{regions[j]}', fontsize=8)
            else: ax.set_xticks([]); ax.set_xlabel('')
    if title==None: 
        title = ""
        if not subtitle == None:
            title += f" - {subtitle}"
    fig.suptitle(title)

    fig.colorbar(im, ax=axes)
    plt.show()

def plot_crossreg_comp(hmaps, reg1, reg2, vmax=None, vmax_diff=None, cmap_reg='coolwarm', cmap_comp='coolwarm', title=None, subtitle=None):
    fig, axes = plt.subplots(ncols=3, nrows=1) # one for each region and one for both

    if not vmax == None: 
        im1 = axes[0].imshow(hmaps[f'{reg1}-{reg2}'], vmin=-vmax, vmax=vmax, cmap=cmap_reg)
    else: 
        im1 = axes[0].imshow(hmaps[f'{reg1}-{reg2}'], cmap=cmap_reg)
    axes[0].set_title(f"{reg1}-{reg2}", fontsize=constants.TINY_SIZE)
    plt.colorbar(im1)
    
    if not vmax == None: 
        im2 = axes[1].imshow(hmaps[f'{reg2}-{reg1}'].T, vmin=-vmax, vmax=vmax, cmap=cmap_reg)
    else: 
        im2 = axes[1].imshow(hmaps[f'{reg2}-{reg1}'].T, cmap=cmap_reg)
    axes[1].set_title(f"{reg2}-{reg1}", fontsize=constants.TINY_SIZE)
    plt.colorbar(im2)

    if not vmax_diff == None: 
        reg_comp = axes[2].imshow(hmaps[f'{reg1}-{reg2}']-hmaps[f'{reg2}-{reg1}'].T, vmin=-vmax_diff, vmax=vmax_diff, cmap=cmap_comp)
    else: 
        reg_comp = axes[2].imshow(hmaps[f'{reg1}-{reg2}']-hmaps[f'{reg2}-{reg1}'].T, cmap=cmap_comp)
    axes[2].set_title(f"{reg1}-{reg2} - {reg2}-{reg1}", fontsize=constants.TINY_SIZE)
    plt.colorbar(reg_comp)

    if title==None: 
        title = ""
        if not subtitle == None:
            title += f" - {subtitle}"
    fig.suptitle(title)

    plt.tight_layout()
    plt.show()