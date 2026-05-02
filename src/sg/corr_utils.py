import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr, zscore
from itertools import product

from utils.paths import FIGURES_DIR


# get scorr between two units
def get_scorr_pair(psths_a, psths_b, do_plot=False):
    psth_a = zscore(np.mean(psths_a, axis=0))  # should be (# tbins, )... and it is
    psth_b = zscore(np.mean(psths_b, axis=0))
    r = pearsonr(psth_a, psth_b).statistic

    if do_plot:
        fig, ax = plt.subplots()
        ax.scatter(psth_a, psth_b)
        fig.suptitle(f"r2: {np.round(r, 3)}")

    return r


# get scorr for all regions in a session
def get_scorr(psths, regions):
    scorr = {}
    for region_a, region_b in product(regions, regions):
        scorr[f"{region_a}-{region_b}"] = np.zeros(
            (psths[region_a].shape[0], psths[region_b].shape[0])
        )

        for i, psths_a in enumerate(psths[region_a]):
            for j, psths_b in enumerate(psths[region_b]):
                scorr[f"{region_a}-{region_b}"][i, j] = get_scorr_pair(psths_a, psths_b)

    return scorr


# NOISE CORRELATION
def get_ncorr_pair(psths_a, psths_b, trial_data, do_plot=False):
    lc_idx = np.where((trial_data["response"] == 1) & (trial_data["rewarded"] == 1))
    rc_idx = np.where((trial_data["response"] == -1) & (trial_data["rewarded"] == 1))
    li_idx = np.where((trial_data["response"] == 1) & (trial_data["rewarded"] == 0))
    ri_idx = np.where((trial_data["response"] == -1) & (trial_data["rewarded"] == 0))

    psth_a_lc = np.mean(psths_a[lc_idx], axis=0)  # should be (# tbins)...all good
    psth_a_rc = np.mean(psths_a[rc_idx], axis=0)
    psth_a_li = np.mean(psths_a[li_idx], axis=0)
    psth_a_ri = np.mean(psths_a[ri_idx], axis=0)

    psth_b_lc = np.mean(psths_b[lc_idx], axis=0)  # should be (# tbins)...all good
    psth_b_rc = np.mean(psths_b[rc_idx], axis=0)
    psth_b_li = np.mean(psths_b[li_idx], axis=0)
    psth_b_ri = np.mean(psths_b[ri_idx], axis=0)

    psths_noise_a = zscore(
        np.concatenate(
            (
                np.ravel([psth_lc - psth_a_lc for psth_lc in psths_a[lc_idx]]),
                np.ravel([psth_rc - psth_a_rc for psth_rc in psths_a[rc_idx]]),
                np.ravel([psth_li - psth_a_li for psth_li in psths_a[li_idx]]),
                np.ravel([psth_ri - psth_a_ri for psth_ri in psths_a[ri_idx]]),
            )
        )
    )
    psths_noise_b = zscore(
        np.concatenate(
            (
                np.ravel([psth_lc - psth_b_lc for psth_lc in psths_b[lc_idx]]),
                np.ravel([psth_rc - psth_b_rc for psth_rc in psths_b[rc_idx]]),
                np.ravel([psth_li - psth_b_li for psth_li in psths_b[li_idx]]),
                np.ravel([psth_ri - psth_b_ri for psth_ri in psths_b[ri_idx]]),
            )
        )
    )

    r = pearsonr(psths_noise_a, psths_noise_b).statistic

    if do_plot:
        fig, ax = plt.subplots()
        ax.scatter(psths_noise_a, psths_noise_b)
        fig.suptitle(f"r: {np.round(r, 3)}")
    return r


def get_ncorr(psths, regions, trial_data):
    ncorr = {}
    for region_a, region_b in product(regions, regions):
        ncorr[f"{region_a}-{region_b}"] = np.zeros(
            (psths[region_a].shape[0], psths[region_b].shape[0])
        )

        for i, psths_a in enumerate(psths[region_a]):
            for j, psths_b in enumerate(psths[region_b]):
                ncorr[f"{region_a}-{region_b}"][i, j] = get_ncorr_pair(
                    psths_a, psths_b, trial_data
                )

    return ncorr


def plot_hmaps_all(
    hmaps,
    regions,
    title=None,
    vmax=None,
    cmap="coolwarm",
    subtitle=None,
    do_save=False,
    format="pdf",
):
    n_units = [
        np.shape(hmaps[reg])[1]
        for i, reg in enumerate(hmaps.keys())
        if i < np.sqrt(len(hmaps.keys()))
    ]  # get the no. units per region
    gs_units = [int(r / 2) for r in n_units]

    fig = plt.figure(figsize=(8, 8), constrained_layout=True)
    gs = GridSpec(nrows=sum(gs_units), ncols=sum(gs_units), figure=fig)

    axes = []
    for i, r_units in enumerate(gs_units):
        for j, c_units in enumerate(gs_units):
            # display heatmap
            ax = fig.add_subplot(
                gs[
                    sum(gs_units[:i]) : sum(gs_units[:i]) + r_units,
                    sum(gs_units[:j]) : sum(gs_units[:j]) + c_units,
                ]
            )
            if vmax is None:
                im = ax.imshow(
                    hmaps[f"{regions[i]}-{regions[j]}"], aspect="auto", cmap=cmap
                )
            else:
                im = ax.imshow(
                    hmaps[f"{regions[i]}-{regions[j]}"],
                    aspect="auto",
                    vmin=-vmax,
                    vmax=vmax,
                    cmap=cmap,
                )
            axes.append(ax)

            # show y-axis ticks + label on left column only
            if j == 0:
                ax.set_ylabel(f"{regions[i]}", fontsize=8)
            else:
                ax.set_yticks([])
                ax.set_ylabel("")

            # show x-axis ticks + label on bottom row only
            if i == len(n_units) - 1:
                ax.set_xlabel(f"{regions[j]}", fontsize=8)
            else:
                ax.set_xticks([])
                ax.set_xlabel("")
    if title is None:
        title = ""
    if subtitle is not None:
        title += f" - {subtitle}"
    fig.suptitle(title)

    fig.colorbar(im, ax=axes)

    return fig


def plot_scorr_all(
    scorr, regions, subj_id, sess_id, do_save=False, format="png", **kwargs
):
    fig = plot_hmaps_all(
        scorr, regions, title=f"Signal Correlation ({subj_id}, {sess_id})", **kwargs
    )

    if do_save:
        save_dir = FIGURES_DIR / "scorr" / subj_id / sess_id / f"scorr.{format}"
        save_dir.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir, dpi=300, bbox_inches="tight")


def plot_ncorr_all(
    ncorr, regions, subj_id, sess_id, do_save=False, format="png", **kwargs
):
    fig = plot_hmaps_all(
        ncorr, regions, title=f"Noise Correlation ({subj_id}, {sess_id})", **kwargs
    )

    if do_save:
        save_dir = FIGURES_DIR / "ncorr" / subj_id / sess_id / f"ncorr.{format}"
        save_dir.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir, dpi=300, bbox_inches="tight")
    fig.close()
