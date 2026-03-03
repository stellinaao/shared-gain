import numpy as np
import matplotlib.pyplot as plt
import torch

from lib import data

def plot_r2s_sess(das):
    for subj_idx in range(len(data.subject_ids)):
        r2_baseline = np.array([torch.mean(das[subj_idx][sess_idx]['drift']['r2test']) for sess_idx in range(len(data.session_ids[subj_idx]))])
        r2_tv = np.array([torch.mean(das[subj_idx][sess_idx]['tv']['r2test']) for sess_idx in range(len(data.session_ids[subj_idx]))])
        r2_affine = np.array([torch.mean(das[subj_idx][sess_idx]['affine']['r2test']) for sess_idx in range(len(data.session_ids[subj_idx]))])
        
        r2_baseline_std = np.array([torch.std(das[subj_idx][sess_idx]['drift']['r2test']) for sess_idx in range(len(data.session_ids[subj_idx]))])
        r2_tv_std = np.array([torch.std(das[subj_idx][sess_idx]['tv']['r2test']) for sess_idx in range(len(data.session_ids[subj_idx]))])
        r2_affine_std = np.array([torch.std(das[subj_idx][sess_idx]['affine']['r2test']) for sess_idx in range(len(data.session_ids[subj_idx]))])
        
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(r2_baseline)), r2_baseline, color="#666666", label='Baseline')
        ax.fill_between(np.arange(len(r2_baseline)), r2_baseline-r2_baseline_std, r2_baseline+r2_baseline_std, color="#666666", alpha=0.2)
        ax.plot(np.arange(len(r2_tv)), r2_tv, color="#E5A400", label="Task Variables")
        ax.fill_between(np.arange(len(r2_tv)), r2_tv-r2_tv_std, r2_tv+r2_tv_std, color="#E5A400", alpha=0.2)
        ax.plot(np.arange(len(r2_affine)), r2_affine, color="#1A4D99", label="Shared Gain")
        ax.fill_between(np.arange(len(r2_affine)), r2_affine-r2_affine_std, r2_affine+r2_affine_std, color="#1A4D99", alpha=0.2)
        
        ax.spines["left"].set_position(("outward", 5))
        ax.spines["bottom"].set_position(("outward", 5))
        
        ax.set_xticks(np.arange(len(r2_affine)))
        ax.set_xticklabels([data.session_ids[subj_idx][sess_idx][4:8] for sess_idx in range(len(data.session_ids[subj_idx]))])
        ax.set_xlabel("Session"); ax.set_ylabel('$r^2$')
        fig.suptitle(f"{data.subject_ids[subj_idx]}, Model Performance (R2)")
        ax.legend()
        fig.tight_layout()
        fig.show()
        
def plot_r2(das, model="affine", save=False, fpath=None):
    markers = ['v', '^', 'x', '*']
    colors = ["#140C6A", "#7166E9", "#845910", "#F7E164"]
    
    cids = das[model]['model'].cids
    r2s = das[model]['r2test']
    
    regs = das['data']['regions']
    reg_keys = das['data']['reg_keys'][cids]
    
    fig, ax = plt.subplots(figsize=(3,3))
    
    for i, reg in enumerate(regs):
        idxs = np.where(reg_keys==i)[0]
        r2s_reg = r2s[idxs]
        ax.plot(idxs, r2s_reg, markers[i], color=colors[i], label=reg)
        ax.axhline(torch.mean(r2s_reg), color=colors[i], linewidth=1, linestyle='--')
    
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_ylabel('$r^2$'); ax.set_xlabel("Unit ID")
    ax.set_title(f"R2, {model}: {torch.mean(r2s):.3f}")
    ax.set_ylim([-1,1])
    
    ax.legend()
    fig.tight_layout()
    fig.show()
        
    
    if save:
        plt.savefig(fpath)

def plot_coupling(das, model="affine", save=False, fpath=None):
    markers = ['v', '^', 'x', '*']
    colors = ["#140C6A", "#7166E9", "#845910", "#F7E164"]
    
    cids = das[model]['model'].cids
    coupling = das[model]['model'].readout_gain.weight.data[:].T
    
    regs = das['data']['regions']
    reg_keys = das['data']['reg_keys'][cids]
    
    fig, ax = plt.subplots(figsize=(3,3))
    
    for i, reg in enumerate(regs):
        idxs = np.where(reg_keys==i)[0]
        coupling_reg = coupling[idxs]
        ax.plot(idxs, coupling_reg, markers[i], color=colors[i], label=reg)
        ax.axhline(torch.mean(coupling_reg), color=colors[i], linewidth=1, linestyle='--')
    
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Coupling Weight$'); ax.set_xlabel("Unit ID")
    ax.set_title(f"Mean Coupling Weight, {model}: {torch.mean(coupling_reg):.3f}")
    ax.set_ylim([-0.75,0.75])
    
    ax.legend()
    fig.tight_layout()
    fig.show()
        
    
    if save:
        plt.savefig(fpath)
