
from copy import deepcopy
import os, sys

from NDNT.training.trainer import Trainer
sys.path.insert(0, '/mnt/Data/Repos/')
sys.path.append("../")

import numpy as np
# from sklearn.decomposition import FactorAnalysis
import pickle
import matplotlib.pyplot as plt

# Import torch
import torch
from torch import nn
from scipy.io import loadmat

from NDNT.training import LBFGSTrainer
from torch.utils.data import Subset, DataLoader
from lib.models import SharedGain, SharedLatentGain, GenericDataset


from sklearn.preprocessing import OneHotEncoder as OHE
from lib import data, spks_utils, models

'''
Model fitting procedure for the shared gain / offset model

'''

from torch.utils.data import Subset, DataLoader

# STELLINA'S FUNCTIONS
def get_dataset_dm(psths, trial_data, regions, task_vars=['response'], num_tents=2, norm=True, binwidth_ms=25, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    # NEURAL DATA
    # robs 
    robs = np.concatenate([np.sum(psths[region]*(binwidth_ms/1000), axis=2) for region in regions]).T**0.5 # spks_utils.get_nspikes_choice(unit_spike_times, trial_data, regions, pre=1, post=1)**0.5
    reg_keys = np.concatenate([np.repeat(i, len(psths[region])) for i, region in enumerate(regions)]) # spks_utils.get_nspikes_choice(unit_spike_times, trial_data, regions, pre=1, post=1)**0.5
    
    # robs = spks_utils.get_nspikes_trial(unit_spike_times, trial_data, trial_dur_s, regions)
    
    print(f"originally {sum([len(psths[region]) for region in regions])} units")
    
    # dfs
    adiff = np.abs(robs - np.mean(robs, axis=0)) # abs diff from avg rate of units across all trials
    mad = np.median(adiff) 
    dfs = (adiff / mad) < 8 

    # filter for good units
    good = np.mean(dfs, axis=0) > .8
    print("good units %d" %np.sum(good))
    robs = robs[:,good]
    dfs = dfs[:,good]
    
    # normalize
    if norm:
        s = np.std(robs, axis=0)
        mu = np.mean(robs, axis=0)
        robs = (robs-mu)/s

    # TRIAL DATA
    # task variables (a.k.a. stim in liska)
    tvs = OHE().fit_transform(trial_data[task_vars]).todense() # only choice for now (when adding reward, just: task_vars=['response', 'rewarded'])

    # tents
    from external.NDNT.utils.NDNutils import tent_basis_generate
    num_trials = len(trial_data)
    xs = np.linspace(0, num_trials-1, num_tents)
    tents = tent_basis_generate(xs)
    
    data_dict = {
        'robs': torch.tensor(robs, dtype=torch.float32),
        'reg_keys': torch.tensor(reg_keys, dtype=torch.float32), 
        'dfs': torch.tensor(dfs, dtype=torch.float32),
        'tv': torch.tensor(tvs, dtype=torch.float32),
        'tents': torch.tensor(tents, dtype=torch.float32),
        'indices': torch.tensor(np.arange(num_trials), dtype=torch.int64) # indices (trial indices)
    }

    data_gd = models.GenericDataset(data_dict, device=device)
    # data_df = pd.DataFrame(data)

    return data_gd, data_dict

# MODELING
# Step 0: Check if dataset has stable low-dim structure at 4+ dims
def check_stable_lowd(data_gd, Mtrain, Mtest, num_units, rank=1):
    data = data_gd.covariates['robs']
    U, Vt, tre, te = cv_pca(data=data_gd.covariates['robs'], rank=rank, Mtrain=Mtrain, Mtest=Mtest)
    resid = U@Vt - data
    mu = torch.sum(data*Mtrain, dim=0)/torch.sum(Mtrain, dim=0)

    total_err = data - mu
    te = 1 - torch.sum(resid**2*Mtest, dim=0) / torch.sum(total_err**2*Mtest, dim=0)

    cids_pca = np.where(te.detach().cpu()>0)[0]
    print("Found %d /%d units with stable low-dimensional structure at rank %d" %(len(cids_pca), num_units, rank))

    return cids_pca

# Step 1: Fit baseline model
# > Baseline model: has no task vars, can capture slow drift in firing rate for each unit using b0-splines
def fit_baseline(train_dl, val_dl, num_tv, num_units, ntents=5):
    mod_baseline = models.SharedGain(num_tv,
            num_units=num_units,
            cids=None,
            num_latent=1,
            num_tents=ntents,
            include_tv=False,
            include_gain=False,
            include_offset=False,
            tents_as_input=False,
            output_nonlinearity='Identity',
            latent_noise=True,
            tv_act_func='lin',
            tv_reg_vals={'l2':0.01},
            reg_vals={'l2':0.001},
            act_func='lin')

    print("Fitting baseline model...", end='')
    fit_model(mod_baseline, train_dl, val_dl, use_lbfgs=True, verbose=0)
    print("done")

    return mod_baseline

# Step 2: Fit model with task vars and slow drift
# > Task vars & slow drift: Used to identify units driven by task vars
def fit_tvs(train_dl, val_dl, num_tv, num_units, mod_baseline, ntents=5):
    mod_tv = models.SharedGain(num_tv,
            num_units=num_units,
            cids=None,
            num_latent=1,
            num_tents=ntents,
            include_tv=True,
            include_gain=False,
            include_offset=False,
            tents_as_input=False,
            output_nonlinearity='Identity',
            latent_noise=True,
            tv_act_func='lin',
            tv_reg_vals={'l2':0.001},
            reg_vals={'l2':0.001},
            act_func='lin')

    mod_tv.drift.weight.data = mod_baseline.drift.weight.data.clone()

    print("Fitting task variable model...", end="")
    fit_model(mod_tv, train_dl, val_dl, verbose=0, use_lbfgs=True)
    print("done")

    return mod_tv

# Step 2a: Evaluate and plot comparison for baseline and task variable models
def plot_r2_comp(figs, results_a, results_b, label_a="", label_b="", save=True, fpath=None):
    fig, ax = plt.subplots(figsize=(4,4))
    figs.append(fig)
    ax.plot(results_a['r2test'], 'o', color='#666666', label=label_a)
    ax.plot(results_b['r2test'], 'o', color='#E5A400', label=label_b)
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_ylabel('$r^2$'); ax.set_xlabel("Unit ID")
    ax.set_title(f"R2, {label_a}: {torch.mean(np.delete(results_a['r2test'], np.where(results_a['r2test']==float('-inf')))):.3f},  {label_b}: {torch.mean(np.delete(results_b['r2test'], np.where(results_b['r2test']==float('-inf')))):.3f}")
    ax.set_ylim([-1,1])
    fig.tight_layout()
    ax.legend()
    
    if save:
        plt.savefig(fpath)

    return figs

# Step 3a: Get units that had significant performance increase with a task variable model
def get_cids(cids_pca, res_tv):
    cids = np.where(res_tv['r2test'] > 0)[0]
    print("Found %d /%d units with significant task variable + drift model" %(len(cids), len(res_tv['r2test'])))
    
    print(max(cids), max(cids_pca))
    cids = np.union1d(cids_pca, cids)
    print("Using %d total units for modeling" %len(cids))

    return cids

# Step 3b: Fit gain autoencoder
def fit_ae_gain(train_dl, val_dl, mod_tv, cids, num_tv, num_units, data_gd, ntents=2, num_latents=1):
    mod_ae_gain = models.SharedGain(num_tv,
                num_units=num_units,
                cids=cids,
                num_latent=num_latents,
                num_tents=ntents,
                latent_noise=False,
                include_tv=True,
                include_gain=True,
                include_offset=False,
                tents_as_input=False,
                output_nonlinearity='Identity',
                tv_act_func='lin',
                tv_reg_vals={'l2': 1},
                reg_vals={'l2': .001},
                act_func='lin')

    if ntents > 1:
        mod_ae_gain.drift.weight.data = mod_tv.drift.weight.data[:,cids].clone()
        mod_ae_gain.bias.requires_grad = False
    else:
        mod_ae_gain.bias.requires_grad = True

    mod_ae_gain.tv.weight.data = mod_tv.tv.weight.data[:,cids].clone()
    mod_ae_gain.bias.data = mod_tv.bias.data[cids].clone()
    mod_ae_gain.tv.weight.requires_grad = False
    mod_ae_gain.readout_gain.weight_scale = 1.0
    mod_ae_gain.latent_gain.weight_scale = 1.0
    mod_ae_gain.readout_gain.weight.data[:] = 1

    mod_ae_gain.prepare_regularization()

    print("Fitting gain autoencoder...", end='')
    fit_autoencoder(mod_ae_gain, train_dl, val_dl, fit_sigmas=False, min_iter=0, max_iter=10)
    print("Done")

    return mod_ae_gain

# Step 3c: Fit offset autoencoder
def fit_ae_offset(train_dl, val_dl, mod_tv, cids, num_tv, num_units, data_gd, ntents=2, num_latents=1):
    mod_ae_offset = models.SharedGain(num_tv,
                num_units=num_units,
                cids=cids,
                num_latent=num_latents,
                num_tents=ntents,
                latent_noise=False,
                include_tv=True,
                include_gain=False,
                include_offset=True,
                tents_as_input=False,
                output_nonlinearity='Identity',
                tv_act_func='lin',
                tv_reg_vals={'l2': 1},
                reg_vals={'l2': .001},
                act_func='lin')

    if ntents > 1:
        mod_ae_offset.drift.weight.data = mod_tv.drift.weight.data[:,cids].clone()
        mod_ae_offset.bias.requires_grad = False
    else:
        mod_ae_offset.bias.requires_grad = True

    mod_ae_offset.tv.weight.data = mod_tv.tv.weight.data[:,cids].clone()
    mod_ae_offset.bias.data = mod_tv.bias.data[cids].clone()
    mod_ae_offset.tv.weight.requires_grad = False
    mod_ae_offset.readout_offset.weight_scale = 1.0
    mod_ae_offset.latent_offset.weight_scale = 1.0
    mod_ae_offset.readout_offset.weight.data[:] = 1

    mod_ae_offset.prepare_regularization()

    print("Fitting offset autoencoder...")
    fit_autoencoder(mod_ae_offset, train_dl, val_dl, fit_sigmas=False, min_iter=0, max_iter=10)
    print("Done")
    
    return mod_ae_offset

# Step 3d: Fit affine autoencoder
def fit_ae_affine(train_dl, val_dl, test_dl, mod_tv, mod_ae_gain, mod_ae_offset, cids, num_tv, num_units, data_gd, device, ntents=5, num_latents=1):
    mod_ae_affine = models.SharedGain(num_tv,
                num_units=num_units,
                cids=cids,
                num_latent=num_latents,
                num_tents=ntents,
                latent_noise=False,
                include_tv=True,
                include_gain=True,
                include_offset=True,
                tents_as_input=False,
                output_nonlinearity='Identity',
                tv_act_func='lin',
                tv_reg_vals={'l2': 1},
                reg_vals={'l2': .1},
                act_func='lin')
    
    # INITIALIZE WEIGHTS
    # initialize with baseline model weights
    if ntents > 1:
        mod_ae_affine.drift.weight.data = mod_tv.drift.weight.data[:,cids].clone()
        mod_ae_affine.drift.weight.requires_grad = True
        mod_ae_affine.bias.requires_grad = False
    else:
        mod_ae_affine.bias.requires_grad = True
    
    # initialize neuron-tv weights with tv model weights
    mod_ae_affine.tv.weight.data = mod_tv.tv.weight.data[:,cids].clone()
    mod_ae_affine.bias.data = mod_tv.bias.data[cids].clone()
    mod_ae_affine.tv.weight.requires_grad = False
    
    # intialize coupling weights with gain and offset only ae models
    mod_ae_affine.readout_gain.weight.data[:] = mod_ae_gain.readout_gain.weight.data.detach().clone() # .shape >> torch.Size([3, 173])
    mod_ae_affine.readout_offset.weight.data[:] = mod_ae_offset.readout_offset.weight.data.detach().clone()
    
    mod_ae_affine.latent_gain.weight.data[:] = mod_ae_gain.latent_gain.weight.data.detach().clone() # .shape >> torch.Size([173, 3])
    mod_ae_affine.latent_offset.weight.data[:] = mod_ae_offset.latent_offset.weight.data.detach().clone()
    
    mod_ae_affine.prepare_regularization()

    # fit_autoencoder: initialize latents by only fitting latents, then refit task var and refit latents
    print("Fitting affine autoencoder...", end='')
    fit_autoencoder(mod_ae_affine, train_dl, val_dl, fit_sigmas=False, min_iter=0, max_iter=10)
    print("Done")

    mod_ae_affine.to(device)
    r2 = model_rsquared(mod_ae_affine, val_dl.dataset[:])
    # print(f"{torch.mean(r2)}, before r2<0 -> 0")

    # mod_ae_affine.readout_gain.weight.data[:,r2<0] = 0  # minor detriment to r2 ([-0.01, -0.1])
    # mod_ae_affine.readout_offset.weight.data[:,r2<0] = 0 # horrible detriment to r2 ([-0.8 (no rand init of latent weights), -0.4 (rand)])
    # r2 = model_rsquared(mod_ae_affine, val_dl.dataset[:])
    print('Autoencoder iter %d, val r2: %.4f' %(0, r2.mean().item()))

    res_ae_affine = eval_model(mod_ae_affine, data_gd, test_dl.dataset)
    print('confirming model r2 = %.4f' %res_ae_affine['r2test'].mean().item())

    return mod_ae_affine, r2

# Step 3e: Convert ae to lvm
def ae2lvm(train_dl, val_dl, mod_ae_offset, mod_ae_gain, mod_ae_affine, cids, num_tv, num_units, data_gd, ntents=5, num_latents=1):
    # convert ae to lvm
    mod_ae_offset = fit_gain_model(num_tv, mod_ae_offset,
        num_units=num_units, num_trials=len(data_gd),
        num_latent=num_latents,
        max_iter=0,
        cids=cids, ntents=ntents,
        train_dl=train_dl, val_dl=val_dl,
        verbose=0,
        l2s=[0.01],
        d2ts=[0.01],
        include_gain=False,
        include_offset=True)

    # convert ae to lvm
    mod_ae_gain = fit_gain_model(num_tv, mod_ae_gain,
        num_units=num_units, num_trials=len(data_gd),
        num_latent=num_latents,
        max_iter=0,
        cids=cids, ntents=ntents,
        train_dl=train_dl, val_dl=val_dl,
        verbose=0,
        l2s=[0.01],
        d2ts=[0.01],
        include_gain=True,
        include_offset=False)
    
    # convert ae to lvm
    mod_ae_affine = fit_gain_model(num_tv, mod_ae_affine,
        num_units=num_units, num_trials=len(data_gd),
        num_latent=num_latents,
        max_iter=0,
        cids=cids, ntents=ntents,
        train_dl=train_dl, val_dl=val_dl,
        verbose=0,
        l2s=[0.01],
        d2ts=[0.01],
        include_gain=True,
        include_offset=True)
    
    return mod_ae_offset, mod_ae_gain, mod_ae_affine

# Step 4: Fit affine model
def fit_affine(train_dl, val_dl, mod_tv, cids, num_tv, num_units, data_gd, ntents=5, num_latents=1):
    print("Fitting Affine Model...", end='')
    mod_affine = fit_gain_model(num_tv, mod_tv,
        num_units=num_units, num_trials=len(data_gd),
        num_latent=num_latents,
        cids=cids, ntents=ntents,
        train_dl=train_dl, val_dl=val_dl,
        verbose=0,
        l2s=[0.001, 0.01, .1, 1],
        d2ts=[0.00001], #[0.0001, 0.001, .01, .1, 1],
        include_gain=True,
        include_offset=True)
    print("Done")

    return mod_affine

# Step 5a: Fit gain only
def fit_gain(mod_ae_gain, mod_affine, train_dl, val_dl, cids, num_tv, num_units, data_gd, ntents=2, num_latents=1):
    print("Fitting Gain Model...", end='')
    mod_gain = fit_gain_model(num_tv, mod_ae_gain, num_units=num_units, num_trials=len(data_gd),
        num_latent=num_latents,
        cids=cids, ntents=ntents,
        train_dl=train_dl, val_dl=val_dl,
        verbose=0,
        l2s=[mod_affine.readout_offset.reg.vals['l2']],
        d2ts=[mod_affine.gain_mu.reg.vals['d2t']],
        include_gain=True,
        include_offset=False)
    print("Done")
    return mod_gain

# Step 5b: Fit offset only
def fit_offset(train_dl, val_dl, mod_ae_affine, mod_affine, cids, num_tv, num_units, data_gd, ntents=2, num_latents=1):
    print("Fitting Offset Model...", end='')
    mod_offset =  fit_gain_model(num_tv, mod_ae_affine, num_units=num_units, num_trials=len(data_gd),
        num_latent=num_latents,
        cids=cids, ntents=ntents,
        train_dl=train_dl, val_dl=val_dl,
        verbose=0,
        l2s=[mod_affine.readout_offset.reg.vals['l2']],
        d2ts=[mod_affine.offset_mu.reg.vals['d2t']],
        include_gain=False,
        include_offset=True)
    print("Done")
    return mod_offset

# EVALUATION
from copy import deepcopy
def get_cvpca_metrics(das, data_gd, cids, train_dl, test_dl, rank):
            data = data_gd.covariates['robs'][:,cids]
            Mtrain = train_dl.dataset[:]['dfs'][:,cids]>0
            Mtest = test_dl.dataset[:]['dfs'][:,cids]>0

            U, Vt, tre, te = cv_pca(data, rank=rank, Mtrain=Mtrain, Mtest=Mtest)
            
            resid = U@Vt - data
            mu = torch.sum(data*Mtrain, dim=0)/torch.sum(Mtrain, dim=0)

            total_err = data - mu

            tre = 1 - torch.sum(resid**2*Mtrain, dim=0) / torch.sum(total_err**2*Mtrain, dim=0)
            te = 1 - torch.sum(resid**2*Mtest, dim=0) / torch.sum(total_err**2*Mtest, dim=0)

            das['cvpca'].append({'rank': rank, 'U': U.cpu().numpy(), 'Vt': Vt.cpu().numpy(), 'r2train': tre.cpu().numpy(), 'r2test': te.cpu().numpy()})
            das['cvpca_train_err'].append((rank, tre.mean().item()))
            das['cvpca_test_err'].append((rank, te.mean().item()))

def eval_models(das, test_dl, mod_baseline, mod_tv, mod_ae_offset, mod_ae_gain, mod_ae_affine, mod_gain, mod_offset, mod_affine, cids, data_gd):

    '''Evaluate Models'''
    # baseline
    mod_baseline.cids = cids
    mod_baseline.bias.data = mod_baseline.bias.data[cids]
    mod_baseline.drift.weight.data = mod_baseline.drift.weight.data[:,cids]
    mod_baseline.drift.bias.data = mod_baseline.drift.bias.data[cids]
    
    moddict_baseline = eval_model(mod_baseline, data_gd, test_dl.dataset)
    moddict_baseline['model'] = mod_baseline
    das['drift'] = moddict_baseline

    # task variables
    mod_tv.cids = cids
    mod_tv.bias.data = mod_tv.bias.data[cids]
    mod_tv.drift.weight.data = mod_tv.drift.weight.data[:,cids]
    mod_tv.tv.weight.data = mod_tv.tv.weight.data[:,cids]
    mod_tv.drift.bias.data = mod_tv.drift.bias.data[cids]
    mod_tv.tv.bias.data = mod_tv.tv.bias.data[cids]

    moddict_tv = eval_model(mod_tv, data_gd, test_dl.dataset)
    moddict_tv['model'] = mod_tv
    das['tv'] = moddict_tv
    
    # autoencoder offset
    moddict_ae_offset = eval_model(mod_ae_offset, data_gd, test_dl.dataset)
    moddict_ae_offset['model'] = mod_ae_offset
    das['offsetae'] = moddict_ae_offset

    # autoencoder gain
    moddict_ae_gain = eval_model(mod_ae_gain, data_gd, test_dl.dataset)
    moddict_ae_gain['model'] = mod_ae_gain
    das['gainae'] = moddict_ae_gain

    # autoencoder affine
    moddict_ae_affine = eval_model(mod_ae_affine, data_gd, test_dl.dataset)
    moddict_ae_affine['model'] = mod_ae_affine
    das['affineae'] = moddict_ae_affine

    # affine
    moddict_affine = eval_model(mod_affine, data_gd, test_dl.dataset)
    moddict_affine['model'] = mod_affine
    das['affine'] = moddict_affine

    # set gain readout weights to zero
    mod_affineng = deepcopy(mod_affine)
    mod_affineng.readout_gain.weight.data[:] = 0

    moddict_affineng = eval_model(mod_affineng, data_gd, test_dl.dataset)
    moddict_affineng['model'] = mod_affineng
    das['affine_nogain'] = moddict_affineng

    # set offset readout weights to zero
    mod_affineno = deepcopy(mod_affine)
    mod_affineno.readout_offset.weight.data[:] = 0

    moddict_affineno = eval_model(mod_affineno, data_gd, test_dl.dataset)
    moddict_affineno['model'] = mod_affineno
    das['affine_nooffset'] = moddict_affineno

    ''' Gain model'''
    moddict_gain = eval_model(mod_gain, data_gd, test_dl.dataset)
    moddict_gain['model'] = mod_gain
    das['gain'] = moddict_gain

    ''' Offset model'''
    moddict_offset = eval_model(mod_offset, data_gd, test_dl.dataset)
    moddict_offset['model'] = mod_offset
    das['offset'] = moddict_offset

    return das

def get_das(trial_data, regions, sample, train_inds, val_inds, test_inds, train_dl, test_dl, mod_baseline, mod_tv, mod_ae_offset, mod_ae_gain, mod_ae_affine, mod_affine, mod_gain, mod_offset, cids, data_gd, apath="vars/", aname="placeholder.pkl", do_save=True, do_plot=True):
    # init das
    das = dict()
    das['data'] = {
        'strategy': trial_data['is_mb'],
        'rewarded': trial_data['rewarded'],
        'regions': regions,
        'robs': sample['robs'].detach().cpu().numpy(),
        'reg_keys': sample['reg_keys'].detach().cpu().numpy(), 
        'dfs': sample['dfs'].detach().cpu().numpy(),
        'train_inds': train_inds,
        'val_inds': val_inds,
        'test_inds': test_inds}

    # evaluate models
    das = eval_models(das, test_dl, mod_baseline, mod_tv, mod_ae_offset, mod_ae_gain, mod_ae_affine, mod_gain, mod_offset, mod_affine, cids, data_gd)
    
    # redo PCA on fit neurons
    print("Fitting CV PCA")
    das['cvpca'] = das['cvpca_train_err'] = das['cvpca_test_err'] = []
    for rank in range(1, 25): get_cvpca_metrics(das, data_gd, cids, train_dl, test_dl, rank)
    
    return das
        
    # if do_save:
    #     print("Saving...", end="")
    #     with open(os.path.join(apath, aname), 'wb') as f:
    #         pickle.dump(das, f)
    #     print("Done")

    if do_plot:
        plot_summary(das, aname)

    return das

# PLOTTING
def plot_summary(das, subj_idx, sess_idx, save=False, fpath=None):
    model = das['affineae']['model']
    for latent_idx in range(model.gain_mu.get_weights().shape[1]):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7,7))

        # R_obs
        zgain = model.gain_mu.get_weights()[:,latent_idx]
        zweight = model.readout_gain.get_weights()[latent_idx]
        if np.mean(np.sign(zweight)) < 0: # flip sign if both are negative
            zgain *= -1
            zweight *= -1

        zoffset = model.offset_mu.get_weights()[:,latent_idx]
        zoffweight = model.readout_offset.get_weights()[latent_idx]
        if np.mean(np.sign(zoffweight)) < 0: # flip sign if both are negative
            zoffset *= -1
            zoffweight *= -1

        robs = das['data']['robs'][:,das['affineae']['model'].cids]

        ind = np.argsort(zweight)

        axes[0][0].imshow(robs[:,ind].T, aspect='auto', interpolation='none', vmin=0, vmax=10, cmap='magma')
        axes[0][0].set_title("R_obs, sorted by gain weights")
        axes[0][0].set_xlabel("Trials"); axes[0][0].set_ylabel("Neurons")


        # Corr w Strategy
        axes[1][0].plot(np.array(das['data']['strategy']), 'k', label='Strategy')
        axes[1][0].plot(zgain, 'r', label='Gain Weights')
        axes[1][0].plot(zoffset, 'b', label='Offset Weights')
        axes[1][0].set_xlim((0, robs.shape[0]))
        axes[1][0].set_xlabel("Trials"); axes[1][0].set_ylabel("Weights")
        axes[1][0].legend()

        from scipy.stats import spearmanr
        rhog = spearmanr(das['data']['strategy'], zgain)
        rhoo = spearmanr(das['data']['strategy'], zoffset)

        titlestr = 'Corr w/ strategy: gain '
        titlestr += "%0.3f" %rhog[0]

        if rhog[1] < 0.05:
            titlestr += "*"

        titlestr += ", offset "
        titlestr += "%0.3f" %rhoo[0]

        if rhoo[1] < 0.05:
            titlestr += "*"
        axes[1][0].set_title(titlestr)

        # R from TV vs + Gain
        axes[0][1].plot(das['tv']['r2test'], das['affineae']['r2test'], 'o')
        mn = min(min(das['tv']['r2test']), min(das['affineae']['r2test']))
        axes[0][1].plot((mn,1), (mn,1), 'k')
        axes[0][1].set_title(f"R, Task Var ({torch.mean(das['tv']['r2test']):.3f}) vs Affine-AE ({torch.mean(das['affineae']['r2test']):.3f})")
        axes[0][1].set_xlabel('Task Vars, R'); axes[0][1].set_ylabel('Gain, R')

        # Histogram
        x = das['affineae']['r2test']-das['tv']['r2test']
        x = x.numpy()
        axes[1][1].hist(x, bins=np.linspace(-0.7, 0.7, 15))
        axes[1][1].plot(np.median(x), plt.ylim()[1], 'v')
        axes[1][1].set_title('R Increase from Adding Gain')
        axes[1][1].set_xlabel('Gain - Task Vars'); axes[1][1].set_ylabel("Frequency")
        
        fig.suptitle(f"{data.subject_ids[subj_idx]}, {data.session_ids[subj_idx][sess_idx]} - Latent Pair #{latent_idx}")

        plt.tight_layout()
        plt.show()
    
        if save:
            plt.savefig(f"{fpath[:-4]}-latent{latent_idx}{fpath[-4:]}")
    
    return # fig

### LISKA'S CODE

def get_data_model(psths, trial_data, regions, norm=True, num_tents=2, task_vars=['response']):
    data_gd, data_dict = get_dataset_dm(psths, trial_data, regions, norm=norm, num_tents=num_tents, task_vars=task_vars, binwidth_ms=25)
    
    train_dl, val_dl, test_dl, (train_inds, val_inds, test_inds) = get_dataloaders(data_gd, batch_size=264, folds=4, use_dropout=True)
    
    Mtrain = train_dl.dataset[:]['dfs']>0
    Mtest = val_dl.dataset[:]['dfs']>0
    
    sample = data_gd[:]
    num_trials, num_tv = sample['tv'].shape
    num_units = sample['robs'].shape[1]
    print("%d Trials, %d Neurons" % (num_trials, num_units))
    
    return data_gd, train_dl, val_dl, test_dl, train_inds, val_inds, test_inds, Mtrain, Mtest, sample, num_trials, num_tv, num_units

def get_dataloaders(data_gd, folds=5, batch_size=64, use_dropout=True, seed=1234):
    np.random.seed(seed)

    num_trials = len(data_gd)

    if use_dropout: # training and test set are the same, but differ in their data filters
        from copy import deepcopy
        train_ds = deepcopy(data_gd)
        val_ds = deepcopy(data_gd)
        test_ds = deepcopy(data_gd)
        
        num_trials, num_units = data_gd.covariates['robs'].shape
        p_holdout = 1/folds # the percent that is held out

        train_mask = np.random.rand(num_trials, num_units) > p_holdout # randomly select 1-p_holdout % of trials + unit pairs
        
        # randomly assign half of the held out unit + trial pairs to be val & the other half to be test
        i,j = np.where(~train_mask) # indices of the held out unit + trial pairs
        ival = np.random.rand(len(i)) < .5
        
        val_mask = np.zeros(train_mask.shape, dtype=bool)
        test_mask = np.zeros(train_mask.shape, dtype=bool)
        
        val_mask[i[ival], j[ival]] = True
        test_mask[i[~ival], j[~ival]] = True

        # apply the masks and create the train, val, and test sets
        # note that dfs is a mask too, so we are incorporating the tvt set mask logic with the og dfs logic
        train_ds.covariates['dfs'] = torch.tensor(np.logical_and(train_ds.covariates['dfs'].cpu().numpy(), train_mask), dtype=torch.float32, device=data_gd.device)
        val_ds.covariates['dfs'] = torch.tensor(np.logical_and(val_ds.covariates['dfs'].cpu().numpy(), val_mask), dtype=torch.float32, device=data_gd.device)
        test_ds.covariates['dfs'] = torch.tensor(np.logical_and(test_ds.covariates['dfs'].cpu().numpy(), test_mask), dtype=torch.float32, device=data_gd.device)
        
        # build latent datafilters (so you always use the training set to get the latent)
        train_ds.covariates['latentdfs'] = train_ds.covariates['dfs'].clone()
        val_ds.covariates['latentdfs'] = train_ds.covariates['dfs'].clone()
        test_ds.covariates['latentdfs'] = train_ds.covariates['dfs'].clone()
        data_gd.covariates['latentdfs'] = train_ds.covariates['dfs'].clone() # og 
        data_gd.requested_covariates = list(data_gd.covariates.keys())
        data_gd.cov_list = list(data_gd.covariates.keys()) # update the instance's 'keys' attr after adding latentdfs

        train_inds = train_mask
        val_inds = val_mask
        test_inds = test_mask
    
    else: # conventional training and test set
        n_val = num_trials//folds # num values per fold
        val_inds = np.random.choice(range(num_trials), size=n_val, replace=False) # select enough vals for one fold
        train_inds = np.setdiff1d(range(num_trials), val_inds) # set difference (i.e., the vals not selected in val_inds)
        ival = np.random.rand(len(val_inds)) < .5 # set half of the val_inds to be test and the other half to be val
        test_inds = val_inds[~ival]
        val_inds = val_inds[ival]

        train_ds = Subset(data_gd, train_inds.tolist())
        val_ds = Subset(data_gd, val_inds.tolist())
        test_ds = Subset(data_gd, test_inds.tolist())

    # create the dls
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    return train_dl, val_dl, test_dl, (train_inds, val_inds, test_inds)
        
def rsquared(y, yhat, dfs=None):
    if dfs is None:
        dfs = torch.ones(y.shape, device=y.device)
    ybar = (y * dfs).sum(dim=0) / dfs.sum(dim=0) # the average y value
    resids = y - yhat # the difference between observed and predicted
    residnull = y - ybar # the difference between observed and observed avg
    sstot = torch.sum( residnull**2*dfs, dim=0) # denom
    ssres = torch.sum( resids**2*dfs, dim=0) # num
    r2 = 1 - ssres/sstot

    return r2.detach().cpu()

def censored_lstsq(A, B, M):
    """Solves least squares problem with missing data in B
    Note: uses a broadcasted solve for speed.
    Args
    ----
    A (ndarray) : m x r matrix
    B (ndarray) : m x n matrix
    M (ndarray) : m x n binary matrix (zeros indicate missing values)
    Returns
    -------
    X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))

    based off code from: http://alexhwilliams.info/itsneuronalblog/2018/02/26/crossval/
    """

    if A.ndim == 1:
        A = A[:,None]

    # else solve via tensor representation
    rhs = (A.T@(M * B)).T[:,:,None] # n x r x 1 tensor
    T = torch.matmul(A.T[None,:,:], M.T[:,:,None] * A[None,:,:]) # n x r x r tensor
    try:
        # transpose to get r x n
        return torch.squeeze(torch.linalg.solve(T, rhs), dim=-1).T
    except:
        r = T.shape[1]
        T[:,torch.arange(r),torch.arange(r)] += 1e-6
        return torch.squeeze(torch.linalg.solve(T, rhs), dim=-1).T

def pca_train(data, rank, Mtrain, max_iter):
    
    # choose solver for alternating minimization
    solver = censored_lstsq

    # initialize U randomly
    U = torch.randn(data.shape[0], rank, device=data.device, dtype=torch.float32)

    # return U, data, rank, Mtrain
    Vt = solver(U, data, Mtrain)
    resid = U@Vt - data
    mse0 = torch.mean(resid**2)
    tol = 1e-3
    # fit pca/nmf
    for itr in range(max_iter):
        Vt = solver(U, data, Mtrain)
        U = solver(Vt.T, data.T, Mtrain.T).T
        resid = U@Vt - data
        mse = torch.mean(resid[Mtrain]**2)
        # print('%d) %.3f' %(itr, mse))
        if mse > (mse0 - tol):
            break
        mse0 = mse

    return mse, U, Vt

def cv_pca(data, rank, Mtrain=None, Mtest=None, p_holdout=0.2, max_iter=10, replicates=5):
    """Fit PCA while holding out a fraction of the dataset.
    """

    # create masking matrix
    if Mtrain is None:
        Mtrain = torch.rand(*data.shape, device=data.device) > p_holdout
    
    if Mtest is None:
        Mtest = ~Mtrain

    Mtrain = Mtrain.to(data.device)
    Mtest = Mtest.to(data.device)

    mses = []
    Us = []
    Vts = []

    for r in range(replicates):
        mse, U, Vt = pca_train(data, rank, Mtrain, max_iter)
        mses.append(mse.item())
        Us.append(U)
        Vts.append(Vt)
    
    id = np.argmin(np.asarray(mses))
    U = Us[id]
    Vt = Vts[id]

    # return result and test/train error
    resid = U@Vt - data
    total_err = data - torch.mean(data, dim=0)
    train_err = 1 - torch.sum(resid[Mtrain]**2) / torch.sum(total_err[Mtrain]**2)
    test_err = 1 - torch.sum(resid[Mtest]**2) / torch.sum(total_err[Mtest]**2)
    return U, Vt, train_err, test_err

def fit_model(model, train_dl, val_dl,
    lr=1e-3, max_epochs=5,
    wd=0.01,
    max_iter=10000,
    use_lbfgs=False,
    verbose=0,
    early_stopping_patience=10,
    use_warmup=True,
    seed=None,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR
    from external.NDNT.training import Trainer, EarlyStopping, LBFGSTrainer
    model.prepare_regularization()

    if use_lbfgs:
        optimizer = torch.optim.LBFGS(model.parameters(),
                    history_size=100,
                    max_iter=max_iter,
                    tolerance_change=1e-9,
                    line_search_fn='strong_wolfe',
                    tolerance_grad=1e-5)

        trainer = LBFGSTrainer(
            optimizer=optimizer,
            device=device,
            accumulate_grad_batches=len(train_dl),
            max_epochs=1,
            optimize_graph=True,
            log_activations=False,
            set_grad_to_none=False,
            verbose=verbose)

        trainer.fit(model, train_dl.dataset[:], seed=seed)

    else:
        earlystopping = EarlyStopping(patience=early_stopping_patience, verbose=False)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
        if use_warmup:
            optimizer.param_groups[0]['lr'] = 1e-5
            warmup_epochs = 5
            trainer = Trainer(model, optimizer,
                device=device,
                optimize_graph=True,
                early_stopping=earlystopping,
                max_epochs=warmup_epochs,
                log_activations=True,
                verbose=verbose)

            trainer.fit(model, train_dl, val_dl, seed=seed)
            trainer.optimizer.param_groups[0]['lr'] = lr
            trainer.max_epochs = max_epochs
            
            trainer.fit(model, train_dl, val_dl, seed=seed)
        else:
            earlystopping = EarlyStopping(patience=early_stopping_patience, verbose=False)
            trainer = Trainer(model, optimizer,
                    device=device,
                    optimize_graph=True,
                    max_epochs=max_epochs,
                    early_stopping=earlystopping,
                    log_activations=True,
                    scheduler_after='batch',
                    scheduler_metric=None,
                    verbose=verbose)

            trainer.fit(model, train_dl, val_dl, seed=seed)
    
    return trainer

def get_data(fpath, num_tents=10,
        normalize_robs=False,
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    # from datasets.utils import tent_basis_generate

    dat = loadmat(fpath)

    trial_ix = np.where(~np.isnan(dat['gdirection']))[0]

    ##### STIMULUS INFO
    directions = np.unique(dat['gdirection'][trial_ix])
    freqs = np.unique(dat['gfreq'][trial_ix])
    direction = dat['gdirection'][trial_ix, ...]
    freq = dat['gfreq'][trial_ix, ...]

    dironehot = direction==directions
    freqonehot = freq==freqs

    ndir = dironehot.shape[1]
    nfreq = freqonehot.shape[1]

    nstim = ndir*nfreq

    # stimulus is a vector of nDirections x nFreqs
    stim = np.reshape(freqonehot[...,None] * dironehot[:,None,...], [-1, nstim])
    # stim = np.reshape(dironehot[...,None] * freqonehot[:,None,...], [-1, nstim])
    # stim = np.reshape( np.expand_dims(dironehot, -1)*np.expand_dims(freqonehot, 1), [-1, nstim])

    num_trials = len(freq)
    xs = np.linspace(0, num_trials-1, num_tents)
    # tents = tent_basis_generate(xs)

    robs = dat['robs'][trial_ix,:].astype(np.float32)
    
    from scipy.ndimage import uniform_filter
    robs_smooth = uniform_filter(robs, size=(50, 1), mode='reflect')
    mx = np.max(robs_smooth, axis=0)
    mn = np.min(robs_smooth, axis=0)

    # good = mx > 1 # max firing rate in sliding window > 1 spike per trial on average
    # pthresh = np.percentile(robs, (0, 99), axis=0)
    # dfs = np.logical_and(robs >= pthresh[0], robs < pthresh[1])
    mu = np.mean(robs, axis=0)
    adiff = np.abs(robs - mu)
    mad = np.median(adiff)
    dfs = (adiff / mad) < 8

    # good = np.mean(dfs, axis=0) > .9 
    good = np.mean(dfs, axis=0) > .8
    # good = np.logical_and(good, mx > 0)
    print("good units %d" %np.sum(good))
    robs = robs[:,good]
    dfs = dfs[:,good]
    
    s = np.std(robs, axis=0)
    mu = np.mean(robs, axis=0)

    mn = robs.min(axis=0)
    mx = robs.max(axis=0)

    if normalize_robs==2:
        # robs = robs / mu
        robs = (robs - mn) / (mx - mn)

    elif normalize_robs==3:
        mu = np.mean(robs, axis=0)
        mad = np.median(np.abs(robs - mu))
        robs = (robs - mu) / mad

    elif normalize_robs==1:
        robs = ( (robs-mu) / s)

    data = {'runningspeed': torch.tensor(dat['runningspeed'][trial_ix], dtype=torch.float32),
        'pupilarea': torch.tensor(dat['pupilarea'][trial_ix], dtype=torch.float32),
        'robs': torch.tensor(robs),
        'dfs': torch.tensor(dfs, dtype=torch.float32),
        'stim': torch.tensor(stim, dtype=torch.float32),
        # 'tents': torch.tensor(tents, dtype=torch.float32),
        'indices': torch.tensor(np.arange(len(trial_ix)), dtype=torch.int64)}

    ds = GenericDataset(data, device=device)

    return ds, dat

def initialize_from_model(model, mod1, train_dl, fit_sigmas=False):
    '''
    fit latent variable model given an initialization
    model: the model to be fit
    mod1: the initialization model. If it is an autoencoder, use the autoencoder parameters for the initial condition
    '''
    
    # used for initialization
    data = train_dl.dataset[:]
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from copy import deepcopy

    model.to(device)
    

    if mod1 is not None:
        robs = data['robs'][:,mod1.cids]
        if 'latentdfs' in data:
            robs = robs * data['latentdfs'][:,mod1.cids]

        robs = robs.to(device)
        
        mod1.to(device)
    
        if hasattr(model, 'gain_mu'):
            
            if hasattr(mod1, 'latent_gain'):
                zg = mod1.latent_gain(robs).detach().clone().to(device) # # trials x # latents
                zgwts = mod1.readout_gain.weight.detach().clone().cpu().to(device)

                v = zg.var(dim=0)
                # v = 1

                s = (zgwts > 0 ).sum() / zgwts.shape[1]
                if s < .5:
                    zg *= -1
                    zgwts *=-1

                model.gain_mu.weight.data[:] = zg/v
                model.readout_gain.weight.data[:] = (zgwts.T*v).T

            else: # random initialization
                model.gain_mu.weight.data[:] = 0#torch.rand(model.gain_mu.weight.shape)
                model.readout_gain.weight.data[:] = 1
            
            model.logvar_g.data[:] = 1
            model.logvar_g.requires_grad = fit_sigmas

        if hasattr(model, 'offset_mu'):
            if hasattr(mod1, 'latent_offset'):
                zh = mod1.latent_offset(robs).detach().clone()
                zhwts = mod1.readout_offset.weight.detach().cpu().clone().to(device)
                v = zh.var(dim=0)
                # v = 1
                
                s = (zhwts > 0 ).sum() / zhwts.shape[1]
                if s < .5:
                    zh *= -1
                    zhwts *=-1
                
                model.offset_mu.weight.data[:] = zh.clone().to(device)/v
                model.readout_offset.weight.data[:] = (zhwts.T*v).T 
            else:
                model.offset_mu.weight.data[:] = 0#torch.randn(model.offset_mu.weight.shape)
                model.readout_offset.weight.data[:] = 1
            
            model.logvar_h.data[:] = 1
            model.logvar_h.requires_grad = fit_sigmas

        if model.drift is not None:
            if len(mod1.cids) == len(model.cids):
                model.drift.weight.data = mod1.drift.weight.data.clone()
            else:
                model.drift.weight.data = mod1.drift.weight.data[:,model.cids].clone()
            model.drift.weight.requires_grad = False
            model.bias.requires_grad = True
        else:
            if len(mod1.cids) == len(model.cids):
                model.bias.data[:] = mod1.bias.data.clone()
            else:
                model.bias.data[:] = mod1.bias.data[model.cids].clone()
            model.bias.requires_grad = True

        if len(mod1.cids) == len(model.cids):
            model.tv.weight.data = mod1.tv.weight.data.clone()
            model.bias.data[:] = mod1.bias.data.clone()
        else:
            model.tv.weight.data = mod1.tv.weight.data[:,model.cids].clone()
            model.bias.data[:] = mod1.bias.data[model.cids].clone()

    model.tv.weight.requires_grad = False
    
    return model

def fit_autoencoder(model, train_dl, val_dl, fit_sigmas=False, min_iter=-1, max_iter=10, seed=None):
    '''
    fit latent variable model given an initialization
    model: the model to be fit
    mod1: the initialization model. If it is an autoencoder, use the autoencoder parameters for the initial condition
    '''
    
    # data used for validation (sets stopping rule)
    vdata = val_dl.dataset[:]
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for dsub in vdata:
        if vdata[dsub].device != device:
            vdata[dsub] = vdata[dsub].to(device)
        
    
    from copy import deepcopy
    
    tol = 1e-9
    model.training = False
    r2 = model_rsquared(model.to(device), vdata)
    l0 = r2.mean().item()

    # l0 = model.validation_step(vdata)['loss'].item()
    model0 = deepcopy(model)
    
    print("Initial: %.4f" %l0)

    if max_iter == 0:
        return l0, model

    # initialize fit by fixing stim, readout; fit gain / offset latents
    model.tv.weight.requires_grad = False

    if hasattr(model, 'latent_gain'):
        model.latent_gain.weight.requires_grad = True
        model.readout_gain.weight.requires_grad = True

    if hasattr(model, 'latent_offset'):
        model.latent_offset.weight.requires_grad = True
        model.readout_offset.weight.requires_grad = True
    
    fit_model(model, train_dl, train_dl, use_lbfgs=True, verbose=0, seed=seed)

    r2 = model_rsquared(model, vdata)
    l1 = r2.mean().item()
    
    print('Fit latents: %.4f, %.4f' % (l0, l1))
    
    l0 = l1

    # fit iteratively
    for itr in range(max_iter):
        
        if itr > min_iter:
            fit_sigmas = True
            

        # fit task vars
        if hasattr(model, 'latent_gain'):
            model.latent_gain.weight.requires_grad = False
            model.readout_gain.weight.requires_grad = False
        
        if hasattr(model, 'latent_offset'):
            model.latent_offset.weight.requires_grad = False
            model.readout_offset.weight.requires_grad = False
        
        model.tv.weight.requires_grad = True
        fit_model(model, train_dl, train_dl, use_lbfgs=True, verbose=0, seed=seed)
        r2 = model_rsquared(model, vdata)
        l1 = r2.mean().item()

        print('%d) fit task vars: %.4f, %.4f' % (itr, l0, l1))
        
        if itr > min_iter and (l1 - l0) < tol:
            print("breaking because tolerance was hit")
            break
        else:
            l0 = l1
            model0 = deepcopy(model)


        # refit latents
        model.tv.weight.requires_grad = False
        
        if hasattr(model, 'latent_gain'):
            model.latent_gain.weight.requires_grad = True
            model.readout_gain.weight.requires_grad = True

        if hasattr(model, 'latent_offset'):
            model.latent_offset.weight.requires_grad = True
            model.readout_offset.weight.requires_grad = True
        
        fit_model(model, train_dl, train_dl, use_lbfgs=True, verbose=0, seed=seed)
        r2 = model_rsquared(model, vdata)
        l1 = r2.mean().item()

        print('%d) fit latents: %.4f, %.4f' % (itr, l0, l1))

        if itr > min_iter and (l1 - l0) < tol:
            print("breaking because tolerance was hit")
            break
        else:
            l0 = l1
            model0 = deepcopy(model)
    
    return l0, model0

def fit_latents(model, train_dl, val_dl, fit_sigmas=False, min_iter=-1, max_iter=10, seed=None, fix_readout_weights=False):
    '''
    fit latent variable model given an initialization
    model: the model to be fit
    mod1: the initialization model. If it is an autoencoder, use the autoencoder parameters for the initial condition
    '''
    
    # used for initialization
    data = train_dl.dataset[:]
    vdata = val_dl.dataset[:]
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for dsub in vdata:
        if vdata[dsub].device != device:
            vdata[dsub] = vdata[dsub].to(device)
    

    
    
    from copy import deepcopy
    
    tol = 1e-9
    model.training = False
    r2 = model_rsquared(model.to(device), vdata)
    l0 = r2.mean().item()

    # l0 = model.validation_step(vdata)['loss'].item()
    model0 = deepcopy(model)

    # initialization value saved until end in case fitting latents does worse
    l00 = deepcopy(l0)
    model00 = deepcopy(model)
    
    print("Initial: %.4f" %l0)

    if max_iter == 0:
        return l0, model

    # initialize fit by fixing stim, readout, fit gain / offset latents

    model.tv.weight.requires_grad = False

    if hasattr(model, 'gain_mu'):
        model.logvar_g.data[:] = 1
        model.logvar_g.requires_grad = fit_sigmas
        model.gain_mu.weight.requires_grad = True
        model.readout_gain.weight.requires_grad = True

    fit_model(model, train_dl, train_dl, use_lbfgs=True, verbose=0, seed=seed)

    if hasattr(model, 'offset_mu'):
        model.logvar_h.data[:] = 1
        model.logvar_h.requires_grad = fit_sigmas
        model.offset_mu.weight.requires_grad = True
        model.readout_offset.weight.requires_grad = True
    
    fit_model(model, train_dl, train_dl, use_lbfgs=True, verbose=0, seed=seed)
    r2 = model_rsquared(model, vdata)
    l1 = r2.mean().item()
    
    print('Fit latents: %.4f, %.4f' % (l0, l1))

    # fit iteratively
    for itr in range(max_iter):
        
        if itr > min_iter:
            fit_sigmas = True

        # fit stimulus
        if hasattr(model, 'gain_mu'):
            model.gain_mu.weight.requires_grad = False
            model.readout_gain.weight.requires_grad = False
            model.logvar_g.data[:] = 0
            model.logvar_g.requires_grad = False
        
        if hasattr(model, 'offset_mu'):
            model.offset_mu.weight.requires_grad = False
            model.readout_offset.weight.requires_grad = False
            model.logvar_h.data[:] = 0
            model.logvar_h.requires_grad = False
        
        model.tv.weight.requires_grad = True
        fit_model(model, train_dl, train_dl, use_lbfgs=True, verbose=0, seed=seed)
        r2 = model_rsquared(model, vdata)
        l1 = r2.mean().item()

        print('%d) fit stim: %.4f, %.4f' % (itr, l0, l1))
        
        if itr > min_iter and (l1 - l0) < tol:
            print("breaking because tolerance was hit")
            break
        else:
            l0 = l1
            model0 = deepcopy(model)


        # refit latents
        model.tv.weight.requires_grad = False
        
        if hasattr(model, 'gain_mu'):
            model.logvar_g.data[:] = 1
            model.logvar_g.requires_grad = fit_sigmas
            model.gain_mu.weight.requires_grad = True
            model.readout_gain.weight.requires_grad = True

        if hasattr(model, 'offset_mu'):
            model.logvar_h.data[:] = 1
            model.logvar_h.requires_grad = fit_sigmas
            model.offset_mu.weight.requires_grad = True
            model.readout_offset.weight.requires_grad = True
        
        fit_model(model, train_dl, train_dl, use_lbfgs=True, verbose=0, seed=seed)
        r2 = model_rsquared(model, vdata)
        l1 = r2.mean().item()

        print('%d) fit latents: %.4f, %.4f' % (itr, l0, l1))

        if itr > min_iter and (l1 - l0) < tol:
            print("breaking because tolerance was hit")
            break
        else:
            l0 = l1
            model0 = deepcopy(model)
    
    # if l0 > l00:
    #     print("LVM worse than autoencoder. reverting")
    #     model0 = model00
    #     l0 = l00
    
    return l0, model0

def eval_model(mod2, ds, val_ds, cids=None, do_plot=False, save=False, fpath=None):
    sample = ds[:]
    mod2 = mod2.to(ds.device)
    rhat = mod2(sample).detach().cpu().numpy()

    # initialize gains and offsets to zero
    zgav = torch.zeros(sample['robs'].shape[0], 1, device='cpu')
    zg = torch.zeros(sample['robs'].shape[0], 1, device='cpu')
    zhav = torch.zeros(sample['robs'].shape[0], 1, device='cpu')
    zh = torch.zeros(sample['robs'].shape[0], 1, device='cpu')

    if hasattr(mod2, 'tents_as_input'):
        if mod2.tents_as_input:
            latent_input = sample['tents']
        else:
            print("as expected")
            latent_input = sample['robs'][:,mod2.cids] * sample['latentdfs'][:,mod2.cids]

        if hasattr(mod2, 'latent_gain'):
            print("checkpoint 1")
            zg = mod2.latent_gain(latent_input)
            
            zgav = mod2.readout_gain(zg).detach().cpu()
            zg = zg.detach().cpu()
            
        if hasattr(mod2, 'latent_offset'):
            print("checkpoint 2")
            zh = mod2.latent_offset(latent_input)
            zhav = mod2.readout_offset(zh).detach().cpu()
            zh = zh.detach().cpu()
            
    else:
        if hasattr(mod2, 'gain_mu'):
            zg = mod2.gain_mu.weight.detach()
            zgav = mod2.readout_gain(zg).detach().cpu()
            zg = zg.cpu().clone()
        
        if hasattr(mod2, 'offset_mu'):
            zh = mod2.offset_mu.weight.detach()
            zhav = mod2.readout_offset(zh).detach().cpu()
            zh = zh.cpu().clone()
        
    sample = val_ds[:]
    robs_ = sample['robs'].detach().cpu()
    rhat_ = mod2(sample).detach().cpu()

    r2test = rsquared(robs_[:,mod2.cids], rhat_, sample['dfs'][:,mod2.cids].detach().cpu())

    if do_plot:
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(5,3))

        axes[0].plot(robs_[:,mod2.cids][:,np.argmax(r2test)], color='#666666', alpha=0.5, label="Observed")
        axes[0].plot(rhat_[:,np.argmax(r2test)], alpha=0.5, color="#5C2392", label="Predicted")
        axes[0].set_title(f"Best Fit: {max(r2test):.3f}"); axes[0].set_xlabel("Trials"); axes[0].set_ylabel("Spike Counts (sqrt)")
        
        axes[1].plot(robs_[:,mod2.cids][:,np.argmin(r2test)], color='#666666', alpha=0.5, label="Observed")
        axes[1].plot(rhat_[:,np.argmin(r2test)], alpha=0.5, color="#5C2392", label="Predicted")
        axes[1].set_title(f"Worst Fit: {min(r2test):.3f}");  axes[1].set_xlabel("Trials"); axes[1].set_ylabel("Spike Counts (sqrt)")
        plt.legend()
        if save:
            plt.savefig(fpath)
        fig.tight_layout()
        fig.show()
        
        

    return {'rhat': rhat, 'zgain': zg, 'zoffset': zh, 'zgainav': zgav, 'zoffsetav': zhav, 'r2test': r2test}

def model_rsquared(model, vdata):
    
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    model.to(device)
    
    for dsub in vdata:
        if vdata[dsub].device != device:
            vdata[dsub] = vdata[dsub].to(device)

    robs_ = vdata['robs'].detach()
    rhat_ = model(vdata).detach()
    model.to('cpu')
    r2 = rsquared(robs_[:,model.cids], rhat_, vdata['dfs'][:,model.cids].detach().cpu())
    return r2

def fit_gain_model(tv_dims, mod1, num_units=None, num_trials=None,
    cids=None, ntents=None, train_dl=None, val_dl=None,
    include_gain=True,
    include_offset=True,
    max_iter=10,
    num_latent=1,
    verbose=0,
    l2s = [.1],
    d2ts = [0, .001, 0.01, 1]): 
    
    from copy import deepcopy    
    
    if include_gain:
        d2tgs = deepcopy(d2ts)
    else:
        d2tgs = [0]
    
    if include_offset:
        d2ths = deepcopy(d2ts)
    else:
        d2ths = [0]

    losses = []
    models = []
    for l2 in l2s:
        for d2th in d2ths:
            for d2tg in d2tgs:
                mod3 = SharedLatentGain(tv_dims,
                            num_units=num_units,
                            num_trials=num_trials,
                            cids=cids,
                            num_latent=num_latent,
                            num_tents=ntents,
                            include_tv=True,
                            include_gain=include_gain,
                            include_offset=include_offset,
                            tents_as_input=False,
                            output_nonlinearity='Identity',
                            tv_act_func='lin',
                            tv_reg_vals={'l2':1},
                            gain_reg_vals={'d2t': d2tg, 'BC': {'d2t': 0}},
                            offset_reg_vals={'d2t': d2th, 'BC': {'d2t': 0}},
                            readout_reg_vals={'l2':l2})

                model = initialize_from_model(mod3, mod1, train_dl, fit_sigmas=False)

                loss, model = fit_latents(model, train_dl, val_dl, fit_sigmas=False, min_iter=-1, max_iter=0)
                
                if max_iter == 0:
                    loss = model_rsquared(model, val_dl.dataset[:]).mean().item()
                    losses.append(loss)
                    models.append(model)
                    print('Fit run %.3f,%.3f: %.4f' % (d2tg, d2th, loss))
                    continue

                model.tv.weight.requires_grad = False
                if include_gain:
                    model.gain_mu.weight.requires_grad = True
                    model.readout_gain.weight.requires_grad = True
                    model.logvar_g.data[:] = 1
                    model.logvar_g.requires_grad = False

                if include_offset:
                    model.offset_mu.weight.requires_grad = True
                    model.readout_offset.weight.requires_grad = True
                    model.logvar_h.data[:] = 1
                    model.logvar_h.requires_grad = False

                if ntents > 1:
                    model.drift.weight.requires_grad = False         
                    model.bias.requires_grad = True
                else:
                    model.bias.requires_grad = True

                loss, model = fit_latents(model, train_dl, val_dl, fit_sigmas=False, min_iter=0, max_iter=max_iter)

                loss = model_rsquared(model, val_dl.dataset[:]).mean().item()
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model.to(device)
                train_loss = model.training_step(train_dl.dataset[:])['loss'].item()
                # print(model.gain_mu.reg.vals)
                losses.append(loss)
                models.append(model)
                print('Fit run %.3f,%.3f: %.4f, train loss = %.4f' % (d2tg, d2th, loss, train_loss))

    id = np.argmax(np.asarray(losses))
    mod2 = deepcopy(models[id])
    return mod2

def fit_session(fpath,
        apath,
        fname,
        aname,
        ntents=5,
        seed=1234):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    figs = [] # initialize list of figures to save into single pdf
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    ds, dat = get_data(fpath, fname, num_tents=ntents, normalize_robs=1)

    trial_ix = np.where(~np.isnan(dat['gdirection']))[0]
    direction = dat['gdirection'][trial_ix]
    directions = np.unique(direction)
    freq = dat['gfreq'][trial_ix]
    freqs = np.unique(freq)
    nfreq = len(freqs)
    ndir = len(directions)
    
    sample = ds[:]
    nstim = sample['stim'].shape[1]
    num_units = sample['robs'].shape[1]
    sx = int(np.ceil(np.sqrt(num_units)))
    sy = int(np.round(np.sqrt(num_units)))

    TC = sample['stim'].T @ sample['robs']
    TC = (TC.T / sample['stim'].sum(dim=0)).T

    figs.append(plt.figure(figsize=(10,10)))
    for cc in range(num_units):
        plt.subplot(sx, sy, cc+1)
        tc = TC[:,cc].detach().cpu().numpy()
        f = plt.plot(directions, np.reshape(tc, [nfreq, ndir]).T, '-o')

    '''
    Build Train / Test sets
    '''
    sample = ds[:]
    num_trials, nstim = sample['stim'].shape
    num_units = sample['robs'].shape[1]
    print("%d Trials n % d Neurons" % (num_trials, num_units))

    # # try to overfit data and throw out outliers
    # mod1 = SharedGain(nstim,
    #             num_units=num_units,
    #             cids=None,
    #             num_latent=1,
    #             num_tents=ntents,
    #             include_stim=True,
    #             include_gain=False,
    #             include_offset=False,
    #             tents_as_input=False,
    #             output_nonlinearity='Identity',
    #             stim_act_func='elu',
    #             stim_reg_vals={'l2':1},
    #             reg_vals={'l2':0.01},
    #             act_func='lin')

    # from torch.utils.data import DataLoader
    # dl = DataLoader(ds, batch_size=64)
    # mod1.bias.requires_grad = False

    # t0 = fit_model(mod1, dl, dl, use_lbfgs=True, verbose=0)

    # mod1.to(device)
    # rhat = mod1(sample)
    # dfs = (rhat - sample['robs']).detach().cpu().abs() < 20
    # ds.covariates['dfs'] = torch.tensor(dfs.numpy(), dtype=torch.float32).to(device)

    train_dl, val_dl, test_dl, indices = get_dataloaders(ds, batch_size=264, folds=4, use_dropout=True)

    train_inds = indices[0]
    val_inds = indices[1]
    test_inds = indices[2]


    '''
    Step 0: check that the dataset has stable low-dimensional structure at >=4 dimensions
    
    '''
    rnk = 1
    data = ds.covariates['robs']
    Mtrain = train_dl.dataset[:]['dfs']>0
    Mtest = val_dl.dataset[:]['dfs']>0
    U, Vt, tre, te = cv_pca(data, rank=rnk, Mtrain=Mtrain, Mtest=Mtest)
    resid = U@Vt - data
    mu = torch.sum(data*Mtrain, dim=0)/torch.sum(Mtrain, dim=0)

    total_err = data - mu
    te = 1 - torch.sum(resid**2*Mtest, dim=0) / torch.sum(total_err**2*Mtest, dim=0)

    cids0 = np.where(te.detach().cpu()>0)[0]
    print("Found %d /%d units with stable low-dimensional structure at rank %d" %(len(cids0), num_units, rnk))

    '''
    Baseline model: has no stimulus, can capture slow drift in firing rate for each unit using b0-splines
    '''
    mod0 = SharedGain(nstim,
            num_units=num_units,
            cids=None,
            num_latent=1,
            num_tents=ntents,
            include_stim=False,
            include_gain=False,
            include_offset=False,
            tents_as_input=False,
            output_nonlinearity='Identity',
            latent_noise=True,
            stim_act_func='lin',
            stim_reg_vals={'l2':0.01},
            reg_vals={'l2':0.001},
            act_func='lin')

    print("Fitting baseline model")
    fit_model(mod0, train_dl, val_dl, use_lbfgs=True, verbose=0)
    print("Done")

    '''
    Model with stimulus and slow drift. Use this to fine which units are driven by the stimulus
    '''
    
    mod1 = SharedGain(nstim,
            num_units=num_units,
            cids=None,
            num_latent=1,
            num_tents=ntents,
            include_stim=True,
            include_gain=False,
            include_offset=False,
            tents_as_input=False,
            output_nonlinearity='Identity',
            stim_act_func='lin',
            stim_reg_vals={'l2':1},
            reg_vals={'l2':0.001},
            act_func='lin')

    mod1.drift.weight.data = mod0.drift.weight.data.clone()

    print("Fitting Stimulus Model")
    fit_model(mod1, train_dl, val_dl, verbose=0, use_lbfgs=True)
    print("Done")

    res0 = eval_model(mod0, ds, test_dl.dataset)
    res1 = eval_model(mod1, ds, test_dl.dataset)

    figs.append(plt.figure())
    plt.plot(res0['r2test'], 'o', label='Baseline')
    plt.plot(res1['r2test'], 'o', label='Stimulus')
    plt.axhline(0, color='k')
    plt.ylabel('$r^2$')
    plt.xlabel("Unit ID")
    plt.ylim([-0.1,1])
    plt.legend()
    

    '''
    Affine model
    '''
    # cids = np.where(np.logical_and(res1['r2test'] > res0['r2test'], res1['r2test'] > 0))[0]
    cids = np.where(res1['r2test'] > 0)[0]
    print("Found %d /%d units with significant stimulus + drift model" %(len(cids), len(res1['r2test'])))
    cids = np.union1d(cids0, cids)
    print("Using %d total units for modeling" %len(cids))
    
    print('Fitting Autoencoder version')
    # seed = 1234
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)

    """ Fit Offset Autoencoder"""
    mod200 = SharedGain(nstim,
                num_units=num_units,
                cids=cids,
                num_latent=1,
                num_tents=ntents,
                latent_noise=False,
                include_stim=True,
                include_gain=False,
                include_offset=True,
                tents_as_input=False,
                output_nonlinearity='Identity',
                stim_act_func='lin',
                stim_reg_vals={'l2': 1},
                reg_vals={'l2': .001},
                act_func='lin')


    if ntents > 1:
        mod200.drift.weight.data = mod1.drift.weight.data[:,cids].clone()
        mod200.bias.requires_grad = False
    else:
        mod200.bias.requires_grad = True

    mod200.tv.weight.data = mod1.tv.weight.data[:,cids].clone()
    mod200.bias.data = mod1.bias.data[cids].clone()
    mod200.tv.weight.requires_grad = False
    mod200.readout_offset.weight_scale = 1.0
    mod200.latent_offset.weight_scale = 1.0
    mod200.readout_offset.weight.data[:] = 1

    mod200.prepare_regularization()

    fit_autoencoder(mod200, train_dl, val_dl, fit_sigmas=False, min_iter=0, max_iter=10)

    """ Fit Gain Autoencoder"""
    mod201 = SharedGain(nstim,
                num_units=num_units,
                cids=cids,
                num_latent=1,
                num_tents=ntents,
                latent_noise=False,
                include_stim=True,
                include_gain=True,
                include_offset=False,
                tents_as_input=False,
                output_nonlinearity='Identity',
                stim_act_func='lin',
                stim_reg_vals={'l2': 1},
                reg_vals={'l2': .001},
                act_func='lin')


    if ntents > 1:
        mod201.drift.weight.data = mod1.drift.weight.data[:,cids].clone()
        mod201.bias.requires_grad = False
    else:
        mod201.bias.requires_grad = True

    mod201.tv.weight.data = mod1.tv.weight.data[:,cids].clone()
    mod201.bias.data = mod1.bias.data[cids].clone()
    mod201.tv.weight.requires_grad = False
    mod201.readout_gain.weight_scale = 1.0
    mod201.latent_gain.weight_scale = 1.0
    mod201.readout_gain.weight.data[:] = 1

    mod201.prepare_regularization()

    fit_autoencoder(mod201, train_dl, val_dl, fit_sigmas=False, min_iter=0, max_iter=10)
    

    """ Fit Affine Autoencoder"""
    mod20 = SharedGain(nstim,
                num_units=num_units,
                cids=cids,
                num_latent=1,
                num_tents=ntents,
                latent_noise=False,
                include_stim=True,
                include_gain=True,
                include_offset=True,
                tents_as_input=False,
                output_nonlinearity='Identity',
                stim_act_func='lin',
                stim_reg_vals={'l2': 1},
                reg_vals={'l2': .1},
                act_func='lin')

    if ntents > 1:
        mod20.drift.weight.data = mod1.drift.weight.data[:,cids].clone()
        mod20.drift.weight.requires_grad = True
        mod20.bias.requires_grad = False
    else:
        mod20.bias.requires_grad = True
    mod20.tv.weight.data = mod1.tv.weight.data[:,cids].clone()
    mod20.bias.data = mod1.bias.data[cids].clone()
    mod20.tv.weight.requires_grad = False
    mod20.readout_gain.weight.data[:] = mod201.readout_gain.weight.data.detach().clone()
    mod20.readout_offset.weight.data[:] = mod200.readout_offset.weight.data.detach().clone()
    mod20.latent_gain.weight.data[:] = mod201.latent_gain.weight.data.detach().clone()
    mod20.latent_offset.weight.data[:] = mod200.latent_offset.weight.data.detach().clone()
    
    mod20.prepare_regularization()

    # fit_model(mod20, train_dl, val_dl, use_lbfgs=True, verbose=0, use_warmup=True)
    fit_autoencoder(mod20, train_dl, val_dl, fit_sigmas=False, min_iter=0, max_iter=10)

    mod20.to(device)
    r2 = model_rsquared(mod20, val_dl.dataset[:])
    mod20.readout_gain.weight.data[:,r2<0] = 0
    mod20.readout_offset.weight.data[:,r2<0] = 0
    r2 = model_rsquared(mod20, val_dl.dataset[:])
    l00 = r2.mean().item()
    print('Autoencoder iter %d, val r2: %.4f' %(0, l00))
        # ctr +=1
        # if ctr > 4:
        #     from copy import deepcopy
        #     mod20 = deepcopy(mod1)
        #     break
    res2 = eval_model(mod20, ds, test_dl.dataset)
    print('confirming model r2 = %.4f' %res2['r2test'].mean().item())

    print("Fitting Affine Model")
    mod2 = fit_gain_model(nstim, mod1,
        num_units=num_units, num_trials=len(ds),
        num_latent=1,
        cids=cids, ntents=ntents,
        train_dl=train_dl, val_dl=val_dl,
        verbose=0,
        l2s=[0.001, 0.01, .1, 1],
        d2ts=[0.00001], #[0.0001, 0.001, .01, .1, 1],
        include_gain=True,
        include_offset=True)

    print("Done")

    ''' 
    Gain Only
    '''
    print("Fitting Gain Model")
    mod3 = fit_gain_model(nstim, mod201, num_units=num_units, num_trials=len(ds),
        num_latent=1,
        cids=cids, ntents=ntents,
        train_dl=train_dl, val_dl=val_dl,
        verbose=0,
        l2s=[mod2.readout_offset.reg.vals['l2']],
        d2ts=[mod2.gain_mu.reg.vals['d2t']],
        include_gain=True,
        include_offset=False)

    print("Done")

    ''' 
    Offset Only
    '''
    
    print("Fitting Offset Model")
    mod4 =  fit_gain_model(nstim, mod20, num_units=num_units, num_trials=len(ds),
        num_latent=1,
        cids=cids, ntents=ntents,
        train_dl=train_dl, val_dl=val_dl,
        verbose=0,
        l2s=[mod2.readout_offset.reg.vals['l2']],
        d2ts=[mod2.offset_mu.reg.vals['d2t']],
        include_gain=False,
        include_offset=True)

    print("convert autoencoder to LVM")
    mod20 = fit_gain_model(nstim, mod20,
        num_units=num_units, num_trials=len(ds),
        num_latent=1,
        max_iter=0,
        cids=cids, ntents=ntents,
        train_dl=train_dl, val_dl=val_dl,
        verbose=0,
        l2s=[0.01],
        d2ts=[0.01],
        include_gain=True,
        include_offset=True)
    
    mod200 = fit_gain_model(nstim, mod200,
        num_units=num_units, num_trials=len(ds),
        num_latent=1,
        max_iter=0,
        cids=cids, ntents=ntents,
        train_dl=train_dl, val_dl=val_dl,
        verbose=0,
        l2s=[0.01],
        d2ts=[0.01],
        include_gain=False,
        include_offset=True)
    
    mod201 = fit_gain_model(nstim, mod201,
        num_units=num_units, num_trials=len(ds),
        num_latent=1,
        max_iter=0,
        cids=cids, ntents=ntents,
        train_dl=train_dl, val_dl=val_dl,
        verbose=0,
        l2s=[0.01],
        d2ts=[0.01],
        include_gain=True,
        include_offset=False)

    print("Done")

    sample = ds[:]

    pupil = sample['pupilarea'].detach().cpu().numpy()
    running = sample['runningspeed'].detach().cpu().numpy()
    robs = sample['robs'].detach().cpu().numpy()
    dfs = sample['dfs'].detach().cpu().numpy()

    das = dict()
    das['data'] = {'direction': direction,
        'frequency': freq,
        'robs': robs,
        'dfs': dfs,
        'pupil': pupil, 'running': running,
        'train_inds': train_inds,
        'val_inds': val_inds,
        'test_inds': test_inds}
    

    '''Evaluate Models'''
    # eval model 0 (Baseline)
    mod0.cids = cids
    mod0.bias.data = mod0.bias.data[cids]
    mod0.drift.weight.data = mod0.drift.weight.data[:,cids]
    mod0.drift.bias.data = mod0.drift.bias.data[cids]
    moddict0 = eval_model(mod0, ds, test_dl.dataset)
    moddict0['model'] = mod0
    das['drift'] = moddict0

    # eval model 1 (Stimulus)
    mod1.cids = cids
    mod1.bias.data = mod1.bias.data[cids]
    mod1.drift.weight.data = mod1.drift.weight.data[:,cids]
    mod1.tv.weight.data = mod1.tv.weight.data[:,cids]
    mod1.drift.bias.data = mod1.drift.bias.data[cids]
    mod1.tv.bias.data = mod1.tv.bias.data[cids]
    moddict1 = eval_model(mod1, ds, test_dl.dataset)
    moddict1['model'] = mod1
    das['stimdrift'] = moddict1
    
    ''' Autoencoder Offset'''
    moddict200 = eval_model(mod200, ds, test_dl.dataset)
    moddict200['model'] = mod200
    das['offsetae'] = moddict200

    ''' Autoencoder Gain'''
    moddict201 = eval_model(mod201, ds, test_dl.dataset)
    moddict201['model'] = mod201
    das['gainae'] = moddict201

    ''' Autoencoder Affine'''
    moddict20 = eval_model(mod20, ds, test_dl.dataset)
    moddict20['model'] = mod20
    das['affineae'] = moddict20

    '''Affine model'''
    moddict2 = eval_model(mod2, ds, test_dl.dataset)
    moddict2['model'] = mod2
    das['affine'] = moddict2

    # set gain readout weights to zero
    mod2ng = deepcopy(mod2)
    mod2ng.readout_gain.weight.data[:] = 0
    moddict2ng = eval_model(mod2ng, ds, test_dl.dataset)
    moddict2ng['model'] = mod2ng
    das['affine_nogain'] = moddict2ng

    # set offset readout weights to zero
    mod2no = deepcopy(mod2)
    mod2no.readout_offset.weight.data[:] = 0
    moddict2no = eval_model(mod2no, ds, test_dl.dataset)
    moddict2no['model'] = mod2no
    das['affine_nogain'] = moddict2no

    ''' Gain model'''
    moddict3 = eval_model(mod3, ds, test_dl.dataset)
    moddict3['model'] = mod3
    das['gain'] = moddict3

    ''' Offset model'''
    moddict4 = eval_model(mod4, ds, test_dl.dataset)
    moddict4['model'] = mod4
    das['offset'] = moddict4

    print("Fitting CV PCA")
    ##%% redo PCA on fit neurons
    das['cvpca'] = []
    train_err= []
    test_err = []
    ranks = range(1, 25)
    for rnk in ranks:
        data = ds.covariates['robs'][:,cids]
        Mtrain = train_dl.dataset[:]['dfs'][:,cids]>0
        Mtest = test_dl.dataset[:]['dfs'][:,cids]>0

        U, Vt, tre, te = cv_pca(data, rank=rnk, Mtrain=Mtrain, Mtest=Mtest)
        
        resid = U@Vt - data
        mu = torch.sum(data*Mtrain, dim=0)/torch.sum(Mtrain, dim=0)

        total_err = data - mu

        tre = 1 - torch.sum(resid**2*Mtrain, dim=0) / torch.sum(total_err**2*Mtrain, dim=0)
        te = 1 - torch.sum(resid**2*Mtest, dim=0) / torch.sum(total_err**2*Mtest, dim=0)

        das['cvpca'].append({'rank': rnk, 'U': U.cpu().numpy(), 'Vt': Vt.cpu().numpy(), 'r2train': tre.cpu().numpy(), 'r2test': te.cpu().numpy()})
        train_err.append((rnk, tre.mean().item()))
        test_err.append((rnk, te.mean().item()))
        
    print("Saving...")
    with open(os.path.join(apath, aname), 'wb') as f:
        pickle.dump(das, f)
    print("Done")

    plot_summary(das, aname)

    return das

# def plot_summary(das, aname):
#     import matplotlib.gridspec as gridspec
#     import matplotlib
#     gridspec.GridSpec(2,3)

#     model = das['affine']['model']
#     zgain = model.gain_mu.get_weights()
#     zweight = model.readout_gain.get_weights()
#     if np.mean(np.sign(zweight)) < 0: # flip sign if both are negative
#         zgain *= -1
#         zweight *= -1
    
#     zoffset = model.offset_mu.get_weights()
#     zoffweight = model.readout_offset.get_weights()
#     if np.mean(np.sign(zoffweight)) < 0: # flip sign if both are negative
#         zoffset *= -1
#         zoffweight *= -1

#     robs = das['data']['robs'][:,das['affine']['model'].cids]

#     ind = np.argsort(zweight)
#     ax0 = plt.subplot2grid((2,3), (0,0), colspan=2)

#     plt.imshow(robs[:,ind].T, aspect='auto', interpolation='none', cmap='jet')
#     plt.title(aname.replace('.pkl', ''))
#     ax0.set_xticklabels([])
    

#     ax = plt.subplot2grid((2,3), (1,0), colspan=2)        
#     plt.plot(das['data']['strategy'], 'k', label='Strategy')
#     ax.set_xlim((0, robs.shape[0]))
#     ax = plt.subplot(2,1,2)

#     ax2 = ax.twinx()
#     plt.plot(zgain, 'r', label='Gain')
#     plt.plot(zoffset, 'b', label='Offset')
#     ax2.set_xlim((0, robs.shape[0]))
#     ax2.legend()
#     plt.xlabel("Trial")
    

#     from scipy.stats import spearmanr
#     rhog = spearmanr(das['data']['strategy'], zgain)
#     rhoo = spearmanr(das['data']['strategy'], zoffset)

#     titlestr = 'Corr w/ running: gain '
#     titlestr += "%0.3f" %rhog[0]

#     if rhog[1] < 0.05:
#         titlestr += "*"

#     titlestr += ", offset "
#     titlestr += "%0.3f" %rhoo[0]

#     if rhoo[1] < 0.05:
#         titlestr += "*"
#     plt.title(titlestr)

#     plt.subplot2grid((2,3), (0,2), colspan=1)
#     plt.plot(das['tv']['r2test'], das['affine']['r2test'], 'o')
#     plt.plot((0,1), (0,1), 'k')
#     plt.xlabel('tv')
#     plt.ylabel('affine')

#     plt.subplot2grid((2,3), (1,2), colspan=1)
#     x = das['affine']['r2test']-das['tv']['r2test']
#     x = x.numpy()
#     plt.hist(x)
#     plt.plot(np.median(x), plt.ylim()[1], 'v')
#     plt.xlabel('affine - tv')

# if main
if __name__ == '__main__':

    fit_session(**vars)

