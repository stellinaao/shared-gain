import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from utils import spearmanr_vec

from sg import data


def plot_summary(family, model, potato=None, mode="offset", metric="spearman"):
    do_gain = mode == "gain" or mode == "affine"
    do_offset = mode == "offset" or mode == "affine"
    n_latents = (
        model.offset_mu.get_weights().shape[1]
        if mode == "offset"
        else model.gain_mu.get_weights().shape[1]
    )

    fig, axes = plt.subplots(
        ncols=1, nrows=n_latents, figsize=(3, 1.5 * n_latents), squeeze=False
    )

    for latent_idx, ax in enumerate(axes.flat):
        ax.plot(np.array(potato), "#9C9C9C", linewidth=2, label="Strategy")
        if do_gain:
            zgain = model.gain_mu.get_weights()[:, latent_idx]
            zweight = (
                model.readout_gain.get_weights()[latent_idx]
                if n_latents > 1
                else model.readout_gain.get_weights()
            )
            if np.mean(np.sign(zweight)) < 0:  # flip sign if both are negative
                zgain *= -1
                zweight *= -1

            ax.plot(zgain, "#9E2D2D", label="Gain Weights")

        if do_offset:
            zoffset = model.offset_mu.get_weights()[:, latent_idx]
            zoffweight = (
                model.readout_offset.get_weights()[latent_idx]
                if n_latents > 1
                else model.readout_offset.get_weights()
            )
            if np.mean(np.sign(zoffweight)) < 0:  # flip sign if both are negative
                zoffset *= -1
                zoffweight *= -1

            ax.plot(zoffset, "#463C8B", label="Offset Weights")

        ax.set_xlim((0, family.num_trials))
        ax.set_xlabel("Trials")
        ax.set_ylabel("Weights")
        ax.legend()

        title = ""
        if do_gain:
            if metric == "spearman":
                corr_g = spearmanr(potato, zgain).statistic
                title += rf"Gain $r$ = {corr_g:.3f}"
            elif metric == "pearsonr":
                corr_g = spearmanr(potato, zgain).statistic
                title += rf"Gain $\rho$ = {corr_g:.3f}"
        if do_offset:
            if metric == "spearman":
                corr_o = spearmanr(potato, zoffset).statistic
                title += (
                    rf" Offset $r$ = {corr_o:.3f}"
                    if do_gain
                    else rf"Offset $r$ = {corr_o:.3f}"
                )
            elif metric == "pearsonr":
                corr_g = spearmanr(potato, zoffset).statistic
                title += (
                    rf" Offset $\rho$ = {corr_o:.3f}"
                    if do_gain
                    else rf"Offset $rbo$ = {corr_o:.3f}"
                )

        ax.set_title(title)

    fig.tight_layout()


def get_num_latents(das, subj_idx, sess_idx, is_msess=True, ae=True, do_plot=False):
    das_ = das[subj_idx][sess_idx] if is_msess else das
    model_str = "affineae" if ae else "affine"

    r2s_tv = [
        np.mean(
            [
                torch.mean(das_[latent_idx]["tv"]["r2test"])
                for latent_idx in range(len(das_))
            ]
        )
    ]
    r2s_affine = [
        torch.mean(das_[latent_idx][model_str]["r2test"])
        for latent_idx in range(len(das_))
    ]

    d = np.diff(np.concatenate((r2s_tv, r2s_affine)), n=1)

    if do_plot:
        plot_r2_latents_diff(das, subj_idx, sess_idx, is_msess, ae)

    return np.where(d < 0.01)[0][0]


def plot_r2_latents_summary(das, subj_idx, ae=True):
    model_str = "affineae" if ae else "affine"

    r2s_tv = [
        [
            np.mean(
                [
                    torch.mean(das_sess[latent_idx]["tv"]["r2test"])
                    for latent_idx in range(len(das_sess))
                ]
            )
        ]
        for das_sess in das
    ]
    r2s_affine = [
        [
            torch.mean(das_sess[latent_idx][model_str]["r2test"])
            for latent_idx in range(len(das_sess))
        ]
        for das_sess in das
    ]

    r2s = np.concatenate((r2s_tv, r2s_affine), axis=1)
    r2s_avg = np.mean(r2s, axis=0)
    r2s_std = np.std(r2s, axis=0)

    fig, ax = plt.subplots()
    ax.plot(range(0, 9), r2s_avg)
    ax.fill_between(range(0, 9), r2s_avg - r2s_std, r2s_avg + r2s_std, alpha=0.4)
    ax.set_xlabel("Number of Latents")
    ax.set_ylabel("R2 of Affine Model")
    fig.suptitle(f"R2 Across Latents, {data.subject_ids[subj_idx]}")
    fig.tight_layout()
    fig.show()


def plot_r2_latents(das, subj_idx, sess_idx, is_msess=True, ae=True):
    das_ = das[subj_idx][sess_idx] if is_msess else das
    model_str = "affineae" if ae else "affine"

    r2s_tv = [
        np.mean(
            [
                torch.mean(das_[latent_idx]["tv"]["r2test"])
                for latent_idx in range(len(das_))
            ]
        )
    ]
    r2s_affine = [
        torch.mean(das_[latent_idx][model_str]["r2test"])
        for latent_idx in range(len(das_))
    ]

    fig, ax = plt.subplots()
    ax.plot(range(0, 9), np.concatenate((r2s_tv, r2s_affine)))
    ax.set_xlabel("Number of Latents")
    ax.set_ylabel("R2 of Affine Model")
    fig.suptitle(
        f"{data.subject_ids[subj_idx]}, {data.session_ids[subj_idx][sess_idx]}"
    )
    fig.tight_layout()
    fig.show()


def plot_r2_latents_diff(das, subj_idx, sess_idx, is_msess=True, ae=True, thresh=0.01):
    das_ = das[subj_idx][sess_idx] if is_msess else das
    model_str = "affineae" if ae else "affine"

    r2s_tv = [
        np.mean(
            [
                torch.mean(das_[latent_idx]["tv"]["r2test"])
                for latent_idx in range(len(das_))
            ]
        )
    ]
    r2s_affine = [
        torch.mean(das_[latent_idx][model_str]["r2test"])
        for latent_idx in range(len(das_))
    ]
    d = np.diff(np.concatenate((r2s_tv, r2s_affine)), n=1)
    first_min = np.where(d < thresh)[0][0]

    fig, ax = plt.subplots()
    ax.plot(range(0, 8), d)
    ax.scatter(first_min, d[first_min], marker="*", color="#FFD343", zorder=2)
    ax.axhline(y=0, color="#333333", linestyle="-")
    ax.axhline(y=0.01, color="#777777", linestyle="--")
    ax.set_xlabel("Number of Latents")
    ax.set_ylabel("R2 of Affine Model")
    fig.suptitle(
        f"{data.subject_ids[subj_idx]}, {data.session_ids[subj_idx][sess_idx]}"
    )
    fig.tight_layout()
    fig.show()


def plot_latents_all(das, num_latents=8):
    for latents in range(1, num_latents + 1):
        plot_latents(das[latents - 1], num_latents=latents, ae=True)


def plot_latents(das, num_latents, ae=True, mult=True):
    plt.figure()
    model = das["affineae"] if ae else das["affine"]
    weights = (
        model["model"].gain_mu.get_weights()
        if mult
        else model["model"].offset_mu.get_weights()
    )
    for ax in range(num_latents):
        plt.plot(weights[:, ax], alpha=0.5, label=f"Latent {ax + 1}")
    plt.title(f"Total # Latents: {num_latents}")
    plt.legend()


def plot_cweights_reg_hist(family, model, n_latents, mode="offset"):
    coupling = (
        model.readout_gain.weight.data[:].T
        if mode == "gain"
        else model.readout_offset.weight.data[:].T
    )

    regs = family.regions
    reg_keys = family.sample["reg_keys"]

    fig, axes = plt.subplots(
        ncols=n_latents, nrows=1, figsize=(1.5 * n_latents, 1.5), squeeze=False
    )

    for latent_idx, ax in enumerate(axes.flat):
        for region_idx, reg in enumerate(regs):
            idxs = np.where(reg_keys == region_idx)[0]
            coupling_reg = coupling[idxs][latent_idx]
            ax.hist(
                coupling_reg,
                color=data.colors_region[reg],
                bins=np.linspace(-2.5, 2.5, 21),
                alpha=0.5,
                label=reg,
            )
        ax.legend()
        ax.set_xlabel("Coupling Weight")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Latent {latent_idx + 1}")
    fig.tight_layout()


def plot_cweight_regs(
    family,
    model,
    ax0,
    ax1,
    num_latents,
    mode="gain",
    is_msess=False,
    use_das=False,
    ae=True,
    abort=True,
    do_save=False,
    do_show=True,
):
    coupling = (
        model.readout_gain.weight.data[:].T
        if mode == "gain"
        else model.readout_offset.weight.data[:].T
    )

    regs = family.regions
    reg_keys = family.sample["reg_keys"]

    if abort and all(
        [
            all(
                coupling[np.where(reg_keys == i)[0], ax0]
                == coupling[np.where(reg_keys == i)[0], ax1]
            )
            for i in reg_keys
        ]
    ):
        print(f"Latent {ax0 + 1} and Latent {ax1 + 1} are equal, aborting")
        return

    fig, ax = plt.subplots(figsize=(3, 3))

    for i, reg in enumerate(regs):
        idxs = np.where(reg_keys == i)[0]
        coupling_reg = coupling[idxs]
        ax.plot(
            coupling_reg[:, ax0],
            coupling_reg[:, ax1],
            data.markers_region[reg],
            color=data.colors_region[reg],
            label=reg,
        )
        # ax.axhline(torch.mean(coupling_reg), color=colors[i], linewidth=0.3, linestyle='--')

    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel(f"Latent {ax0 + 1}")
    ax.set_ylabel(f"Latent {ax1 + 1}")
    # fig.suptitle(f"{data.subject_ids[subj_idx]}, {data.session_ids[subj_idx][sess_idx]}; Total # Latents: {num_latents}")

    # ax.set_ylim([-0.5,0.5])
    # ax.set_xlim([-0.5,0.5])

    ax.legend()
    fig.tight_layout()
    if do_save:
        raise NotImplementedError("Check the filepath.")
        # fig.savefig(
        #     f"figs/cweights/{data.subject_ids[subj_idx]}-{data.session_ids[subj_idx][sess_idx]}_{num_latents}latents-ax{ax0 + 1}-ax{ax1 + 1}.png"
        # )
    if do_show:
        fig.show()


def plot_cweights_regs_sess(
    das,
    subj_idx,
    sess_idx,
    num_latents=8,
    is_mult=True,
    is_msess=False,
    ae=True,
    abort=True,
    do_save=False,
    do_show=True,
):
    for latents in range(num_latents):
        plot_cweights_regs_latent(
            das,
            latents,
            subj_idx,
            sess_idx,
            is_mult,
            is_msess,
            ae,
            abort,
            do_save,
            do_show,
        )


def plot_cweights_regs_latent(
    das,
    num_latents,
    subj_idx,
    sess_idx,
    is_mult=True,
    is_msess=False,
    use_das=False,
    ae=True,
    abort=True,
    do_save=False,
    do_show=True,
):
    for ax0 in range(num_latents):
        for ax1 in range(num_latents):
            if ax0 < ax1:
                plot_cweight_regs(
                    das,
                    ax0,
                    ax1,
                    num_latents,
                    subj_idx,
                    sess_idx,
                    is_mult,
                    is_msess,
                    use_das,
                    ae,
                    abort,
                    do_save,
                    do_show,
                )


def res_taskvar_corr(family, mode="taskvar", plot_res=True, plot_r2_dist=True):
    # family.eval()

    robs = family.val_dl.dataset[:]["robs"]
    rhat = (
        family.res_taskvar["rhat"] if mode == "taskvar" else family.res_offset["rhat"]
    )
    res = robs - rhat

    block_side = family.block_side
    choice = family.response
    reward = family.rewarded

    # rob_block_corr = utils.spearmanr_vec(block_side, robs)
    # rob_choice_corr = utils.spearmanr_vec(choice, robs)
    # rob_reward_corr = utils.spearmanr_vec(reward, robs)
    res_block_corr = spearmanr_vec(block_side, res)
    res_choice_corr = spearmanr_vec(choice, res)
    res_reward_corr = spearmanr_vec(reward, res)

    if plot_res:
        plt.figure()
        plt.imshow(
            res.T,
            aspect="auto",
            interpolation="none",
            cmap="coolwarm",
            vmin=-0.2,
            vmax=0.2,
        )
        plt.colorbar(label="Residuals")
        plt.xlabel("Trials")
        plt.ylabel("Neurons")
        plt.tight_layout()
        plt.show()
    if plot_r2_dist:
        plt.figure()
        plt.hist(res_block_corr, bins=np.arange(-0.45, 0.45, 0.1))
        plt.xlabel(r"$\rho$")
        plt.ylabel("Frequency")
        plt.title("Block")
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.hist(res_choice_corr, bins=np.arange(-0.45, 0.45, 0.1))
        plt.xlabel(r"$\rho$")
        plt.ylabel("Frequency")
        plt.title("Choice")
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.hist(res_reward_corr, bins=np.arange(-0.45, 0.45, 0.1))
        plt.xlabel(r"$\rho$")
        plt.ylabel("Frequency")
        plt.title("Reward")
        plt.tight_layout()
        plt.show()


"""def plot_cweight_regs(
    das,
    ax0,
    ax1,
    num_latents,
    subj_idx,
    sess_idx,
    is_mult=True,
    is_msess=False,
    use_das=False,
    ae=True,
    abort=True,
    do_save=False,
    do_show=True,
):
    if use_das:
        das_ = das
    else:
        das_ = das[subj_idx][sess_idx][num_latents] if is_msess else das[num_latents]
    model_str = "affineae" if ae else "affine"

    cids = das_[model_str]["model"].cids
    coupling = (
        das_[model_str]["model"].readout_gain.weight.data[:].T
        if is_mult
        else das["affine"]["model"].readout_offset.weight.data[:].T
    )

    regs = das_["data"]["regions"]
    reg_keys = das_["data"]["reg_keys"][cids]

    if abort and all(
        [
            all(
                coupling[np.where(reg_keys == i)[0], ax0]
                == coupling[np.where(reg_keys == i)[0], ax1]
            )
            for i in reg_keys
        ]
    ):
        print(f"Latent {ax0 + 1} and Latent {ax1 + 1} are equal, aborting")
        return

    fig, ax = plt.subplots(figsize=(3, 3))

    for i, reg in enumerate(regs):
        idxs = np.where(reg_keys == i)[0]
        coupling_reg = coupling[idxs]
        ax.plot(
            coupling_reg[:, ax0],
            coupling_reg[:, ax1],
            data.markers_region[reg],
            color=data.colors_region[reg],
            label=reg,
        )
        # ax.axhline(torch.mean(coupling_reg), color=colors[i], linewidth=0.3, linestyle='--')

    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel(f"Latent {ax0 + 1}")
    ax.set_ylabel(f"Latent {ax1 + 1}")
    fig.suptitle(
        f"{data.subject_ids[subj_idx]}, {data.session_ids[subj_idx][sess_idx]}; Total # Latents: {num_latents}"
    )

    # ax.set_ylim([-0.5,0.5])
    # ax.set_xlim([-0.5,0.5])

    ax.legend()
    fig.tight_layout()
    if do_save:
        fig.savefig(
            f"figs/cweights/{data.subject_ids[subj_idx]}-{data.session_ids[subj_idx][sess_idx]}_{num_latents}latents-ax{ax0 + 1}-ax{ax1 + 1}.png"
        )
    if do_show:
        fig.show()
    return
"""
