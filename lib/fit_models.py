# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from joblib import Parallel, delayed

from lib import data, fitlvm_utils

def seed():
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def fit_all(trial_data_all, session_data_all, unit_spike_times_all, regions_all):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    das = []
    figs = []
    
    # print("Loading Data")
    # trial_data_r_all, trial_data_all, session_data_all, unit_spike_times_all, regions_all = data.load_data()
    print("Fitting Models")
    
    out = Parallel(n_jobs=8, backend='loky')(delayed(lambda subj_idx: [fit_sess(unit_spike_times_all[subj_idx][sess_idx], trial_data_all[subj_idx][sess_idx], session_data_all[subj_idx][sess_idx], regions_all[subj_idx][sess_idx], subj_idx, sess_idx) for sess_idx in range(len(data.session_ids[subj_idx]))])(subj_idx) for subj_idx in range(len(data.subject_ids)))
    
    das  = [[out[subj_idx][sess_idx][0] for sess_idx in range(len(data.session_ids[subj_idx]))] for subj_idx in range(len(data.subject_ids))]
    figs = [[out[subj_idx][sess_idx][1] for sess_idx in range(len(data.session_ids[subj_idx]))] for subj_idx in range(len(data.subject_ids))]

    return das, figs

def fit_all_latents_gs(trial_data_all, session_data_all, unit_spike_times_all, regions_all, max_n_latents=8):
    print("Fitting Models")
    def fit_sess_latents_gs(subj_idx, sess_idx):
        return Parallel(n_jobs=2, backend='loky')(delayed(fit_sess)(unit_spike_times_all[subj_idx][sess_idx], trial_data_all[subj_idx][sess_idx], session_data_all[subj_idx][sess_idx], regions_all[subj_idx][sess_idx], subj_idx, sess_idx, num_latents=num_latents) for num_latents in range(1,max_n_latents))
    
    def fit_subj_latents_gs(subj_idx):
        return Parallel(n_jobs=5, backend='loky')(delayed(fit_sess_latents_gs)(subj_idx, sess_idx) for sess_idx in range(len(data.session_ids[subj_idx])))
    
    out = Parallel(n_jobs=2, backend='loky')(delayed(fit_subj_latents_gs)(subj_idx) for subj_idx in range(len(data.subject_ids)))
    
    das  = [[[out[subj_idx][sess_idx][latent_idx][0] for latent_idx in range(max_n_latents-1)] for sess_idx in range(len(data.session_ids[subj_idx]))] for subj_idx in range(len(data.subject_ids))]
    figs = [[[out[subj_idx][sess_idx][latent_idx][1] for latent_idx in range(max_n_latents-1)] for sess_idx in range(len(data.session_ids[subj_idx]))] for subj_idx in range(len(data.subject_ids))]

    return out, das, figs

def fit_sess(unit_spike_times, trial_data, session_data, regions, subj_idx, sess_idx, num_latents=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    figs = []
    
    with open('logs/fit_sess_log.txt', 'a') as f:
        print(f">{data.session_ids[subj_idx][sess_idx]}, num_latents = {num_latents}", file=f)
    figs_sess = []
    
    # get session data
    psths, _, _, _, _, _, mask = data.get_psths(unit_spike_times, trial_data, session_data, regions, 
                                    tpre=0.5, tpost=0.5, binwidth_ms=25, alignment='choice', 
                                balance=True, reward_only=False, do_rem_zstd=True, shuffle=False, prev_filter=False)
    trial_data = trial_data[mask]
    
    data_gd, train_dl, val_dl, test_dl, train_inds, val_inds, test_inds, Mtrain, Mtest, sample, num_trials, num_tv, num_units = fitlvm_utils.get_data_model(psths, trial_data, regions, norm=False, num_tents=2, task_vars=['response', 'rewarded', 'response_prev', 'rewarded_prev'])
    
    # MODELING
    # Step 0: Check if dataset has stable low-dim structure at 4+ dims
    print("Step 0")
    cids_pca = fitlvm_utils.check_stable_lowd(data_gd, Mtrain, Mtest, num_units)
    
    # Step 1: Fit baseline model
    # > Baseline model: has no task vars, can capture slow drift in firing rate for each unit using b0-splines
    print("Step 1: Baseline")
    mod_baseline = fitlvm_utils.fit_baseline(train_dl, val_dl, num_tv, num_units, ntents=2)
    
    # Step 2: Fit model with task vars and slow drift
    # > Task vars & slow drift: Used to identify units driven by task vars
    print("Step 2: Task Var")
    mod_tv = fitlvm_utils.fit_tvs(train_dl, val_dl, num_tv, num_units, mod_baseline) 

    # Step 2a: Evaluate and plot comparison for baseline and task variable models
    print("Step 2a: Eval")
    res_baseline        = fitlvm_utils.eval_model(mod_baseline, data_gd, test_dl.dataset, do_plot=True, save=False)
    res_tv              = fitlvm_utils.eval_model(mod_tv, data_gd, test_dl.dataset, do_plot=True, save=False)
    
    figs_sess.append(fitlvm_utils.plot_r2_comp(figs, res_baseline, res_tv, label_a="Baseline", label_b="Curr/Prev Ch./Rw.", save=False))
    
    # Step 3a: Get units that had significant performance increase with a task variable model
    print("Step 3a: Inclusion Criteria")
    cids = fitlvm_utils.get_cids(cids_pca, res_tv)
    
    # Step 3b: Fit gain autoencoder
    print("Step 3b: Gain AE")
    mod_ae_gain = fitlvm_utils.fit_ae_gain(train_dl, val_dl, mod_tv, cids, num_tv, num_units, data_gd, num_latents=num_latents) 
    
    # Step 3c: Fit offset autoencoder
    print("Step 3c: Offset AE")
    mod_ae_offset = fitlvm_utils.fit_ae_offset(train_dl, val_dl, mod_tv, cids, num_tv, num_units, data_gd, num_latents=num_latents)
    
    # Step 3d: Fit affine autoencoder
    print("Step 3d: Affine AE")
    mod_ae_affine, r2 = fitlvm_utils.fit_ae_affine(train_dl, val_dl, test_dl, mod_tv, mod_ae_gain, mod_ae_offset, cids, num_tv, num_units, data_gd, device, num_latents=num_latents)
    
    # Step 3e: Convert ae to lvm
    print("Step 3e: AE to LVM")
    mod_ae_offset, mod_ae_gain, mod_ae_affine = fitlvm_utils.ae2lvm(train_dl, val_dl, mod_ae_offset, mod_ae_gain, mod_ae_affine, cids, num_tv, num_units, data_gd, num_latents=num_latents)
    
    # # Step 4: Fit affine model
    # print("Step 4: Fit Affine")
    # mod_affine = fitlvm_utils.fit_affine(train_dl, val_dl, mod_tv, cids, num_tv, num_units, data_gd, num_latents=num_latents)
    
    # # Step 5a: Fit gain only
    # print("Step 5a: Fit gain only")
    # mod_gain = fitlvm_utils.fit_gain(mod_ae_gain, mod_affine, train_dl, val_dl, cids, num_tv, num_units, data_gd, ntents=2, num_latents=num_latents)
    
    # # Step 5b: Fit offset only
    # print("Step 5b: Fit offset only")
    # mod_offset = fitlvm_utils.fit_offset(train_dl, val_dl, mod_ae_affine, mod_affine, cids, num_tv, num_units, data_gd, num_latents=num_latents)
    
    # EVALUATION
    print("Evaluating...")
    das_sess = fitlvm_utils.get_das(trial_data, regions, sample, train_inds, val_inds, test_inds, train_dl, test_dl, mod_baseline, mod_tv, mod_ae_offset, mod_ae_gain, mod_ae_affine, cids, data_gd, apath="vars/", aname=f"{data.subject_ids[subj_idx]}-{data.session_ids[subj_idx][sess_idx]}-fpass.pkl", do_save=True, do_plot=True)
    figs_sess.append(fitlvm_utils.plot_summary(das_sess, subj_idx=subj_idx, sess_idx=sess_idx))
    
    return das_sess, figs_sess