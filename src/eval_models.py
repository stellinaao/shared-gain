import numpy as np
import torch
import matplotlib.pyplot as plt

from lib import data

def get_num_latents(das, subj_idx, sess_idx, is_msess=True, ae=True, do_plot=False):
    das_ = das[subj_idx][sess_idx] if is_msess else das
    model_str = 'affineae' if ae else 'affine'

    r2s_tv = [np.mean([torch.mean(das_[latent_idx]['tv']['r2test']) for latent_idx in range(len(das_))])]
    r2s_affine = [torch.mean(das_[latent_idx][model_str]['r2test']) for latent_idx in range(len(das_))]

    d = np.diff(np.concatenate((r2s_tv,r2s_affine)), n=1)
    
    if do_plot:
        plot_r2_latents_diff(das, subj_idx, sess_idx, is_msess, ae)
    
    return np.where(d<0.01)[0][0]

def plot_r2_latents_summary(das, subj_idx, ae=True):
    model_str = 'affineae' if ae else 'affine'
    
    r2s_tv = [[np.mean([torch.mean(das_sess[latent_idx]['tv']['r2test']) for latent_idx in range(len(das_sess))])] for das_sess in das]
    r2s_affine = [[torch.mean(das_sess[latent_idx][model_str]['r2test']) for latent_idx in range(len(das_sess))] for das_sess in das]
    
    r2s = np.concatenate((r2s_tv, r2s_affine), axis=1)
    r2s_avg = np.mean(r2s, axis=0)
    r2s_std = np.std(r2s, axis=0)

    fig, ax = plt.subplots()
    ax.plot(range(0,9), r2s_avg)
    ax.fill_between(range(0,9), r2s_avg-r2s_std, r2s_avg+r2s_std, alpha=0.4)
    ax.set_xlabel("Number of Latents"); ax.set_ylabel("R2 of Affine Model")
    fig.suptitle(f"R2 Across Latents, {data.subject_ids[subj_idx]}")
    fig.tight_layout()
    fig.show()
    
def plot_r2_latents(das, subj_idx, sess_idx, is_msess=True, ae=True):
    das_ = das[subj_idx][sess_idx] if is_msess else das
    model_str = 'affineae' if ae else 'affine'
    
    r2s_tv = [np.mean([torch.mean(das_[latent_idx]['tv']['r2test']) for latent_idx in range(len(das_))])]
    r2s_affine = [torch.mean(das_[latent_idx][model_str]['r2test']) for latent_idx in range(len(das_))]

    fig, ax = plt.subplots()
    ax.plot(range(0,9), np.concatenate((r2s_tv,r2s_affine)))
    ax.set_xlabel("Number of Latents"); ax.set_ylabel("R2 of Affine Model")
    fig.suptitle(f"{data.subject_ids[subj_idx]}, {data.session_ids[subj_idx][sess_idx]}")
    fig.tight_layout()
    fig.show()
    
def plot_r2_latents_diff(das, subj_idx, sess_idx, is_msess=True, ae=True, thresh=0.01):
    das_ = das[subj_idx][sess_idx] if is_msess else das
    model_str = 'affineae' if ae else 'affine'
    
    r2s_tv = [np.mean([torch.mean(das_[latent_idx]['tv']['r2test']) for latent_idx in range(len(das_))])]
    r2s_affine = [torch.mean(das_[latent_idx][model_str]['r2test']) for latent_idx in range(len(das_))]
    d = np.diff(np.concatenate((r2s_tv,r2s_affine)), n=1)
    first_min = np.where(d<thresh)[0][0]
    
    fig, ax = plt.subplots()
    ax.plot(range(0,8), d)
    ax.scatter(first_min, d[first_min], marker="*", color="#FFD343", zorder=2)
    ax.axhline(y=0, color='#333333', linestyle='-')
    ax.axhline(y=0.01, color='#777777', linestyle='--')
    ax.set_xlabel("Number of Latents"); ax.set_ylabel("R2 of Affine Model")
    fig.suptitle(f"{data.subject_ids[subj_idx]}, {data.session_ids[subj_idx][sess_idx]}")
    fig.tight_layout()
    fig.show()
    
def plot_latents_all(das, num_latents=8):
    for latents in range(1, num_latents+1): 
        plot_latents(das[latents-1], num_latents=latents, ae=True)
    
def plot_latents(das, num_latents, ae=True, mult=True):
    plt.figure()
    model = das['affineae'] if ae else das['affine']
    weights = model['model'].gain_mu.get_weights() if mult else model['model'].offset_mu.get_weights() 
    for ax in range(num_latents):
        plt.plot(weights[:,ax], alpha=0.5, label=f"Latent {ax+1}")
    plt.title(f"Total # Latents: {num_latents}")
    plt.legend()
    
def plot_cweights_regs_sess(das, subj_idx, sess_idx, num_latents=8, is_mult=True, is_msess=False, ae=True, abort=True, do_save=False, do_show=True):
    for latents in range(num_latents):
        plot_cweights_regs_latent(das, latents, subj_idx, sess_idx, is_mult, is_msess, ae, abort, do_save, do_show)
    
def plot_cweights_regs_latent(das, num_latents, subj_idx, sess_idx, is_mult=True, is_msess=False, ae=True, abort=True, do_save=False, do_show=True):
    for ax0 in range(num_latents):
        for ax1 in range(num_latents):
            if ax0 < ax1: plot_cweight_regs(das, ax0, ax1, num_latents, subj_idx, sess_idx, is_mult, is_msess, ae, abort, do_save, do_show)
    

def plot_cweight_regs(das, ax0, ax1, num_latents, subj_idx, sess_idx, is_mult=True, is_msess=False, ae=True, abort=True, do_save=False, do_show=True):
    das_ = das[subj_idx][sess_idx][num_latents] if is_msess else das[num_latents]
    model_str = 'affineae' if ae else 'affine'

    cids = das_[model_str]['model'].cids
    coupling = das_[model_str]['model'].readout_gain.weight.data[:].T if is_mult else das['affine']['model'].readout_offset.weight.data[:].T

    regs = das_['data']['regions']
    reg_keys = das_['data']['reg_keys'][cids]
    
    if abort and all([all(coupling[np.where(reg_keys==i)[0],ax0] == coupling[np.where(reg_keys==i)[0],ax1]) for i in reg_keys]):
        print(f"Latent {ax0+1} and Latent {ax1+1} are equal, aborting")
        return

    fig, ax = plt.subplots(figsize=(3,3))

    for i, reg in enumerate(regs):
        idxs = np.where(reg_keys==i)[0]
        coupling_reg = coupling[idxs]
        ax.plot(coupling_reg[:,ax0], coupling_reg[:,ax1], data.markers_region[reg], color=data.colors_region[reg], label=reg)
        #ax.axhline(torch.mean(coupling_reg), color=colors[i], linewidth=0.3, linestyle='--')

    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_xlabel(f'Latent {ax0+1}'); ax.set_ylabel(f"Latent {ax1+1}")
    fig.suptitle(f"{data.subject_ids[subj_idx]}, {data.session_ids[subj_idx][sess_idx]}; Total # Latents: {num_latents}")
    
    # ax.set_ylim([-0.5,0.5])
    # ax.set_xlim([-0.5,0.5])

    ax.legend()
    fig.tight_layout()
    if do_save: fig.savefig(f"figs/cweights/{data.subject_ids[subj_idx]}-{data.session_ids[subj_idx][sess_idx]}_{num_latents}latents-ax{ax0+1}-ax{ax1+1}.png")
    if do_show: fig.show()
    return