# save scatter compiled across sessions, no errorbars (mess)
import numpy as np
import matplotlib.pyplot as plt
from core.data import subject_ids, session_ids
from sg.models import Encoder, StrategyEncoder, BootstrapperShuffle as BSS
from sg.models import make_tre
from core.viz import plot_scatter
from utils.viz_utils import save_fig
from utils.paths import FIGURES_DIR
from core.utils import b_regr_tex
from core.viz import plot_kdes
from utils.colors import colors_region_epoch
from utils.colors import colors_region, colors_strategy

subj_id = "MR95"
sess_id = "20260909_180217"

encoder = Encoder(subj_id, sess_id, norm=True)
encoder.verify()

encoder_mb = StrategyEncoder(subj_id, sess_id, norm=True, strategy_filter="mb")
encoder_mf = StrategyEncoder(subj_id, sess_id, norm=True, strategy_filter="mf")

encoder_mb.verify()
encoder_mb.fit_encoder()

encoder_mf.verify()
encoder_mf.fit_encoder()

# bweights_master = {
#     regr: {reg: {strat: [] for strat in ["mb", "mf"]} for reg in encoder.regions}
#     for regr in encoder.tv_keys
# }

# bweights_master_ns = {
#     regr: {reg: {strat: [] for strat in ["mb", "mf"]} for reg in encoder.regions}
#     for regr in encoder.tv_keys
# }

# alphas = {
#     regr: {reg: {strat: [] for strat in ["mb", "mf"]} for reg in encoder.regions}
#     for regr in encoder.tv_keys
# }

# alphas_ns = {
#     regr: {reg: {strat: [] for strat in ["mb", "mf"]} for reg in encoder.regions}
#     for regr in encoder.tv_keys
# }


# for sess_id in session_ids[np.where(subject_ids == subj_id)[0][0]]:
#     print(sess_id)

#     # get encoder weights
#     encoder_ = Encoder(subj_id, sess_id, norm=True)
#     encoder_mb_ = StrategyEncoder(subj_id, sess_id, norm=True, strategy_filter="mb")
#     encoder_mf_ = StrategyEncoder(subj_id, sess_id, norm=True, strategy_filter="mf")

#     try:
#         encoder_mb_.fit_encoder()
#         encoder_mf_.fit_encoder()
#         encoder_.fit_encoder()

#         assert not np.all(encoder_mb_.encoder_weights==0), "zeros in the mb"
#         assert not np.all(encoder_mf_.encoder_weights==0), "zeros in the mf"
#         assert not np.all(encoder_.encoder_weights==0), "zeros in the full"
#     except ValueError:
#         continue

#     # get cids
#     bss_mb_ = BSS(
#         subj_id,
#         sess_id,
#         StrategyEncoder,
#         strategy_filter="mb",
#         n=10,
#         norm=True,
#     )
#     bss_mb_.get_ci_idxs()

#     bss_mf_ = BSS(
#         subj_id,
#         sess_id,
#         StrategyEncoder,
#         strategy_filter="mf",
#         n=10,
#         norm=True,
#     )
#     bss_mf_.get_ci_idxs()

#     cids_ = {
#         regr: {
#             reg: np.sort(
#                 np.unique(
#                     np.union1d(
#                         bss_mb_.ci_idxs_reg[regr][reg], bss_mf_.ci_idxs_reg[regr][reg]
#                     )
#                 )
#             )
#             for reg in encoder_.regions
#         }
#         for regr in encoder_.tv_keys
#     }

#     cids_ns_ = {
#         regr: {
#             reg: np.sort(
#                 np.unique(np.setdiff1d(encoder_.reg_idxs[reg], cids_[regr][reg]))
#             )
#             for reg in encoder_.regions
#         }
#         for regr in encoder_.tv_keys
#     }

#     # add to dict
#     for regr in encoder_.tv_keys:
#         for reg in encoder_.regions:
#             bweights_master[regr][reg]["mb"].append(
#                 encoder_mb_.encoder_weights[cids_[regr][reg], encoder_.tv_idxs[regr]]
#             )
#             bweights_master[regr][reg]["mf"].append(
#                 encoder_mf_.encoder_weights[cids_[regr][reg], encoder_.tv_idxs[regr]]
#             )
#             alphas[regr][reg]["mb"].append(encoder_mb_.encoder.alpha_[cids_[regr][reg]])
#             alphas[regr][reg]["mf"].append(encoder_mf_.encoder.alpha_[cids_[regr][reg]])

#             bweights_master_ns[regr][reg]["mb"].append(
#                 encoder_mb_.encoder_weights[cids_ns_[regr][reg], encoder_.tv_idxs[regr]]
#             )
#             bweights_master_ns[regr][reg]["mf"].append(
#                 encoder_mf_.encoder_weights[cids_ns_[regr][reg], encoder_.tv_idxs[regr]]
#             )
#             alphas_ns[regr][reg]["mb"].append(
#                 encoder_mb_.encoder.alpha_[cids_ns_[regr][reg]]
#             )
#             alphas_ns[regr][reg]["mf"].append(
#                 encoder_mf_.encoder.alpha_[cids_ns_[regr][reg]]
#             )

# mn = np.min(
#     [
#         np.min([np.min(a) for a in bweights_master[regr][reg][strat]])
#         for regr in encoder.tv_keys
#         for reg in encoder.regions
#         for strat in ["mb", "mf"]
#     ]
# )
# mx = np.max(
#     [
#         np.max([np.max(a) for a in bweights_master[regr][reg][strat]])
#         for regr in encoder.tv_keys
#         for reg in encoder.regions
#         for strat in ["mb", "mf"]
#     ]
# )

# from core.viz import plot_scatter
# from utils.viz_utils import save_fig
# from utils.paths import FIGURES_DIR

# for regr in encoder.tv_keys:
#     fig, axes = plt.subplots(
#         ncols=len(encoder.regions), figsize=(2.25*len(encoder.regions), 2.5), sharey=True, tight_layout=True
#     )

#     for i, (ax, reg) in enumerate(zip(axes.flat, encoder.regions)):
#         # plot significant first
#         bw_mb = np.concatenate(bweights_master[regr][reg]["mb"])
#         bw_mf = np.concatenate(bweights_master[regr][reg]["mf"])
#         ax = plot_scatter(
#             x=bw_mb,
#             y=bw_mf,
#             xlabel="mb",
#             ylabel="mf",
#             add_unity=True,
#             add_lr=True,
#             mn=mn,
#             mx=mx,
#             title=reg,
#             ax=ax,
#         )

#     regr_tex = regr.replace("_", r"\_")
#     fig.suptitle(rf"$\beta_{{\mathrm{{{regr_tex}}}}}$")

#     for fext in ["svg", "png"]:
#         save_fig(
#             fig,
#             FIGURES_DIR / "bweight" / "strategy_scatter" / subj_id,
#             f"{regr}-bweight_strategy_scatter-{subj_id}_sesscomp.{fext}",
#         )

# # plot significant and not significant in same scatter
# from core.viz import plot_scatter
# from utils.viz_utils import save_fig
# from utils.paths import FIGURES_DIR

# for regr in encoder.tv_keys:
#     fig, axes = plt.subplots(
#         ncols=len(encoder.regions), figsize=(2.25*len(encoder.regions), 2.5), sharey=True, tight_layout=True
#     )

#     for i, (ax, reg) in enumerate(zip(axes.flat, encoder.regions)):
#         # plot significant first
#         bw_mb = np.concatenate(bweights_master[regr][reg]["mb"])
#         bw_mf = np.concatenate(bweights_master[regr][reg]["mf"])
#         ax = plot_scatter(
#             x=bw_mb,
#             y=bw_mf,
#             color="#f0bb71",
#             xlabel="mb",
#             ylabel="mf",
#             add_unity=True,
#             add_lr=True,
#             lr_color="#f26704",
#             mn=mn,
#             mx=mx,
#             title=reg,
#             ax=ax,
#         )

#         # then plot non significant
#         bw_mb_ns = np.concatenate(bweights_master_ns[regr][reg]["mb"])
#         bw_mf_ns = np.concatenate(bweights_master_ns[regr][reg]["mf"])
#         ax = plot_scatter(
#             x=bw_mb_ns,
#             y=bw_mf_ns,
#             color="#58606A",
#             xlabel="mb",
#             ylabel="mf",
#             add_title=False,
#             add_unity=True,
#             add_lr=True,
#             lr_color="#48556D",
#             mn=mn,
#             mx=mx,
#             title=reg,
#             ax=ax,
#         )
#     regr_tex = regr.replace("_", r"\_")
#     fig.suptitle(rf"$\beta_{{\mathrm{{{regr_tex}}}}}$")

#     for fext in ["svg", "png"]:
#         save_fig(
#             fig,
#             FIGURES_DIR / "bweight" / "strategy_scatter" / subj_id ,
#             f"{regr}-bweight_strategy_scatter-highlighted-{subj_id}_sig_n_notsig_sesscomp.{fext}",
#         )

# # ssi sem across sessions

# from core.viz import plot_kdes
# from core.utils import b_regr_tex
# from utils.colors import colors_region, colors_strategy


# def get_ssi(bweight_mb, bweight_mf, abs=False):
#     if abs:
#         return np.abs((bweight_mb - bweight_mf) / (bweight_mb + bweight_mf))
#     else:
#         return (bweight_mb - bweight_mf) / (bweight_mb + bweight_mf)


# for regr in encoder.tv_keys:
#     fig, ax = plot_kdes(
#         data={
#             reg: [
#                 get_ssi(
#                     bweights_sess_mb,
#                     bweights_sess_mf,
#                     abs=False,
#                 )
#                 for (bweights_sess_mb, bweights_sess_mf) in zip(
#                     bweights_master[regr][reg]["mb"], bweights_master[regr][reg]["mf"]
#                 )
#             ]
#             for reg in encoder.regions
#         },
#         do_sem=True,
#         bw_method=0.1,
#         label="ssi",
#         ylabel="density",
#         xlim=[-5, 5],
#         ynorm=False,
#         add_means=False,
#         line_kwargs={reg: {"color": colors_region[reg]} for reg in encoder.regions},
#     )
#     ax.set_title(b_regr_tex(regr))
#     ax.axvline(x=0, color="#555555", linewidth=0.75, linestyle="--", zorder=-1)
#     ax.axvline(
#         x=-1, color=colors_strategy["mf"], linewidth=0.75, linestyle="--", zorder=-1
#     )
#     ax.axvline(
#         x=1, color=colors_strategy["mb"], linewidth=0.75, linestyle="--", zorder=-1
#     )

#     for fext in ["svg", "png"]:
#         save_fig(
#             fig,
#             FIGURES_DIR / "bweight" / "ssi" / "distros" / subj_id,
#             fname=f"{regr}-bweight_ssi-{subj_id}_sessavg.{fext}",
#         )

# # alpha values between mb and mf (significant cells only)
# from core.utils import b_regr_tex
# from utils.viz_utils import save_fig
# from utils.paths import FIGURES_DIR

# reg_exp = encoder.max_reg
# alpha_range = np.logspace(-reg_exp, reg_exp, 2 * reg_exp + 1)

# for regr in encoder.tv_keys:
#     fig, axes = plt.subplots(ncols=len(encoder.regions), figsize=(2.5*len(encoder.regions), 2), tight_layout=True)
#     for i, reg in enumerate(encoder.regions):
#         ax = axes[i]

#         a_mb = np.concatenate(alphas[regr][reg]["mb"])
#         a_mf = np.concatenate(alphas[regr][reg]["mf"])

#         alpha_counts = np.array(
#             [
#                 [len(np.where((a_mb == a1) & (a_mf == a2))[0]) for a2 in alpha_range]
#                 for a1 in alpha_range
#             ]
#         )

#         im = ax.imshow(alpha_counts, cmap="Blues")

#         ax.set_xlabel(r"mb $\alpha$")
#         ax.set_ylabel(r"mf $\alpha$")
#         ax.set_title(rf"{reg}, $\alpha$ count comparison", fontsize=7)

#         ax.set_xticks(
#             np.arange(len(alpha_range)),
#             [f"{a:.0e}" for a in alpha_range],
#             rotation=45,
#             ha="right",
#             fontsize=5,
#         )
#         ax.set_yticks(
#             np.arange(len(alpha_range)), [f"{a:.0e}" for a in alpha_range], fontsize=5
#         )

#         fig.colorbar(im, ax=ax, shrink=0.67, label="count")
#     fig.suptitle(rf"{b_regr_tex(regr)}, significant units")
#     for fext in ["svg", "png"]:
#         save_fig(
#             fig,
#             FIGURES_DIR / "alphas" / subj_id,
#             fname=f"{regr}-alphas_strategy_comp-{subj_id}_sesscomp.{fext}",
#         )

# del bweights_master
# del bweights_master_ns
# del alphas
# del alphas_ns

print("tre")

encoder_t = make_tre(Encoder, tr_type="dme")(
    subj_id,
    sess_id,
    norm=True,
    stepsize_s=0.5,
)
encoder_t.verify()

# save scatter compiled across sessions, no errorbars (mess)

# subj_id = "MR82"
# subj_id = "MR83"
subj_id = "MR95"

bweights_master = {
    regr: {reg: {strat: [] for strat in ["mb", "mf"]} for reg in encoder.regions}
    for regr in encoder_t.tv_idxs.keys()
}

bweights_master_ns = {
    regr: {reg: {strat: [] for strat in ["mb", "mf"]} for reg in encoder.regions}
    for regr in encoder_t.tv_idxs.keys()
}

alphas = {
    regr: {reg: {strat: [] for strat in ["mb", "mf"]} for reg in encoder.regions}
    for regr in encoder_t.tv_idxs.keys()
}

alphas_ns = {
    regr: {reg: {strat: [] for strat in ["mb", "mf"]} for reg in encoder.regions}
    for regr in encoder_t.tv_idxs.keys()
}


for sess_id in session_ids[np.where(subject_ids == subj_id)[0][0]]:
    print(sess_id)

    # get encoder weights
    encoder_ = make_tre(Encoder)(subj_id, sess_id, norm=True, stepsize_s=0.5)
    encoder_mb_ = make_tre(StrategyEncoder)(
        subj_id, sess_id, norm=True, stepsize_s=0.5, strategy_filter="mb"
    )
    encoder_mf_ = make_tre(StrategyEncoder)(
        subj_id, sess_id, norm=True, stepsize_s=0.5, strategy_filter="mf"
    )

    try:
        encoder_mb_.fit_encoder()
        encoder_mf_.fit_encoder()
        encoder_.fit_encoder()
    except ValueError:
        continue

    # get cids
    bss_mb_ = BSS(
        subj_id,
        sess_id,
        make_tre(StrategyEncoder),
        strategy_filter="mb",
        n=8,
        norm=True,
        stepsize_s=0.5,
    )
    bss_mb_.get_ci_idxs()

    bss_mf_ = BSS(
        subj_id,
        sess_id,
        make_tre(StrategyEncoder),
        strategy_filter="mf",
        n=8,
        norm=True,
        stepsize_s=0.5,
    )
    bss_mf_.get_ci_idxs()

    cids_ = {
        regr: {
            reg: np.sort(
                np.unique(
                    np.union1d(
                        bss_mb_.ci_idxs_reg[regr][reg], bss_mf_.ci_idxs_reg[regr][reg]
                    )
                )
            )
            for reg in encoder_.regions
        }
        for regr in encoder_.tv_idxs.keys()
    }

    cids_ns_ = {
        regr: {
            reg: np.sort(
                np.unique(np.setdiff1d(encoder_.reg_idxs[reg], cids_[regr][reg]))
            )
            for reg in encoder_.regions
        }
        for regr in encoder_.tv_idxs.keys()
    }

    # add to dict
    for regr in encoder_.tv_idxs.keys():
        for reg in encoder_.regions:
            bweights_master[regr][reg]["mb"].append(
                encoder_mb_.encoder_weights[cids_[regr][reg], encoder_.tv_idxs[regr]]
            )
            bweights_master[regr][reg]["mf"].append(
                encoder_mf_.encoder_weights[cids_[regr][reg], encoder_.tv_idxs[regr]]
            )
            alphas[regr][reg]["mb"].append(encoder_mb_.encoder.alpha_[cids_[regr][reg]])
            alphas[regr][reg]["mf"].append(encoder_mf_.encoder.alpha_[cids_[regr][reg]])

            bweights_master_ns[regr][reg]["mb"].append(
                encoder_mb_.encoder_weights[cids_ns_[regr][reg], encoder_.tv_idxs[regr]]
            )
            bweights_master_ns[regr][reg]["mf"].append(
                encoder_mf_.encoder_weights[cids_ns_[regr][reg], encoder_.tv_idxs[regr]]
            )
            alphas_ns[regr][reg]["mb"].append(
                encoder_mb_.encoder.alpha_[cids_ns_[regr][reg]]
            )
            alphas_ns[regr][reg]["mf"].append(
                encoder_mf_.encoder.alpha_[cids_ns_[regr][reg]]
            )

mn = np.min(
    [
        np.min([np.min(a) for a in bweights_master[regr][reg][strat]])
        for regr in encoder_t.tv_idxs.keys()
        for reg in encoder_t.regions
        for strat in ["mb", "mf"]
    ]
)
mx = np.max(
    [
        np.max([np.max(a) for a in bweights_master[regr][reg][strat]])
        for regr in encoder_t.tv_idxs.keys()
        for reg in encoder_t.regions
        for strat in ["mb", "mf"]
    ]
)


# plot significant cells in strategy scatter

for regr in encoder.tv_keys:
    fig, axes = plt.subplots(
        ncols=len(encoder.regions),
        nrows=encoder_t.num_bins,
        figsize=(2.25 * len(encoder.regions), 2.5 * encoder_t.num_bins),
        sharey=True,
        sharex=True,
        tight_layout=True,
    )

    for i in range(encoder_t.num_bins):
        for j, reg in enumerate(encoder_t.regions):
            ax = axes[i][j]
            regr_t = f"{regr}_{i}"

            bw_mb = np.concatenate(bweights_master[regr_t][reg]["mb"])
            bw_mf = np.concatenate(bweights_master[regr_t][reg]["mf"])
            ax = plot_scatter(
                x=bw_mb,
                y=bw_mf,
                xlabel="mb",
                ylabel="mf",
                add_unity=True,
                add_lr=True,
                mn=mn,
                mx=mx,
                title=reg,
                ax=ax,
            )
    regr_tex = regr.replace("_", r"\_")
    fig.suptitle(rf"$\beta_{{\mathrm{{{regr_tex}}}}}$")

    for fext in ["svg", "png"]:
        save_fig(
            fig,
            FIGURES_DIR / "bweight" / "strategy_scatter" / subj_id / "time_resolved",
            f"{regr}-bweight_strategy_scatter-time_resolved-{subj_id}_sesscomp.{fext}",
        )

# plot significant and not significant in same scatter

for regr in encoder.tv_keys:
    fig, axes = plt.subplots(
        ncols=len(encoder.regions),
        nrows=encoder_t.num_bins,
        figsize=(2.25 * len(encoder.regions), 2.5 * encoder_t.num_bins),
        sharey=True,
        tight_layout=True,
    )

    for i in range(encoder_t.num_bins):
        for j, reg in enumerate(encoder.regions):
            ax = axes[i][j]
            regr_t = f"{regr}_{i}"

            # plot significant first
            bw_mb = np.concatenate(bweights_master[regr_t][reg]["mb"])
            bw_mf = np.concatenate(bweights_master[regr_t][reg]["mf"])
            ax = plot_scatter(
                x=bw_mb,
                y=bw_mf,
                color="#f0bb71",
                xlabel="mb",
                ylabel="mf",
                add_unity=True,
                add_lr=True,
                lr_color="#f26704",
                mn=mn,
                mx=mx,
                title=reg,
                ax=ax,
            )

            # then plot non significant
            bw_mb_ns = np.concatenate(bweights_master_ns[regr_t][reg]["mb"])
            bw_mf_ns = np.concatenate(bweights_master_ns[regr_t][reg]["mf"])
            ax = plot_scatter(
                x=bw_mb_ns,
                y=bw_mf_ns,
                color="#58606A",
                xlabel="mb",
                ylabel="mf",
                add_title=False,
                add_unity=True,
                add_lr=True,
                lr_color="#48556D",
                mn=mn,
                mx=mx,
                title=reg,
                ax=ax,
            )
    regr_tex = regr.replace("_", r"\_")
    fig.suptitle(rf"$\beta_{{\mathrm{{{regr_tex}}}}}$")

    for fext in ["svg", "png"]:
        save_fig(
            fig,
            FIGURES_DIR / "bweight" / "strategy_scatter" / subj_id / "time_resolved",
            f"{regr}-bweight_strategy_scatter-highlighted-time_resolved-{subj_id}_sig_n_notsig_sesscomp.{fext}",
        )

# sem - region comp


def get_ssi(bweight_mb, bweight_mf, abs=False):
    if abs:
        return np.abs((bweight_mb - bweight_mf) / (bweight_mb + bweight_mf))
    else:
        return (bweight_mb - bweight_mf) / (bweight_mb + bweight_mf)


for regr in encoder.tv_keys:
    fig, axes = plt.subplots(
        ncols=encoder_t.num_bins,
        figsize=(2.25 * encoder_t.num_bins, 2.5),
        sharey=True,
        tight_layout=True,
    )

    for i in range(encoder_t.num_bins):
        ax = axes[i]
        regr_t = f"{regr}_{i}"
        _, ax = plot_kdes(
            data={
                reg: [
                    get_ssi(
                        bweights_sess_mb,
                        bweights_sess_mf,
                        abs=False,
                    )
                    for (bweights_sess_mb, bweights_sess_mf) in zip(
                        bweights_master[regr_t][reg]["mb"],
                        bweights_master[regr_t][reg]["mf"],
                    )
                ]
                for reg in encoder.regions
            },
            do_sem=True,
            bw_method=0.1,
            label="ssi",
            ylabel="density",
            xlim=[-5, 5],
            ynorm=False,
            add_means=False,
            line_kwargs={reg: {"color": colors_region[reg]} for reg in encoder.regions},
            ax=ax,
        )
        ax.set_title(encoder_t.epoch_keys[i])
        ax.axvline(x=0, color="#555555", linewidth=0.75, linestyle="--", zorder=-1)
        ax.axvline(
            x=-1, color=colors_strategy["mf"], linewidth=0.75, linestyle="--", zorder=-1
        )
        ax.axvline(
            x=1, color=colors_strategy["mb"], linewidth=0.75, linestyle="--", zorder=-1
        )
    fig.suptitle(b_regr_tex(regr))

    for fext in ["svg", "png"]:
        save_fig(
            fig,
            FIGURES_DIR
            / "bweight"
            / "ssi"
            / "distros"
            / "time_resolved"
            / subj_id
            / "region_comp",
            fname=f"{regr}-bweight_ssi-time_resolved-region_comp-{subj_id}_sessavg.{fext}",
        )

# sem - time comp

for regr in encoder.tv_keys:
    fig, axes = plt.subplots(
        ncols=len(encoder_t.regions),
        figsize=(2.25 * len(encoder_t.regions), 2.5),
        sharey=True,
        tight_layout=True,
    )

    for i, reg in enumerate(encoder_t.regions):
        ax = axes[i]

        _, ax = plot_kdes(
            data={
                encoder_t.epoch_keys_str[i]: [
                    get_ssi(
                        bweights_sess_mb,
                        bweights_sess_mf,
                        abs=False,
                    )
                    for (bweights_sess_mb, bweights_sess_mf) in zip(
                        bweights_master[f"{regr}_{i}"][reg]["mb"],
                        bweights_master[f"{regr}_{i}"][reg]["mf"],
                    )
                ]
                for i in range(encoder_t.num_bins)
            },
            do_sem=True,
            bw_method=0.1,
            label="ssi",
            ylabel="density",
            xlim=[-5, 5],
            ynorm=False,
            add_means=False,
            line_kwargs={
                encoder_t.epoch_keys_str[i]: {"color": colors_region_epoch[reg][i]}
                for i in range(encoder_t.num_bins)
            },
            ax=ax,
        )
        ax.set_title(b_regr_tex(regr))
        ax.axvline(x=0, color="#555555", linewidth=0.75, linestyle="--", zorder=-1)
        ax.axvline(
            x=-1, color=colors_strategy["mf"], linewidth=0.75, linestyle="--", zorder=-1
        )
        ax.axvline(
            x=1, color=colors_strategy["mb"], linewidth=0.75, linestyle="--", zorder=-1
        )

    for fext in ["svg", "png"]:
        save_fig(
            fig,
            FIGURES_DIR
            / "bweight"
            / "ssi"
            / "distros"
            / "time_resolved"
            / subj_id
            / "epoch_comp",
            fname=f"{regr}-bweight_ssi-time_resolved-epoch_comp-{subj_id}_sessavg.{fext}",
        )


# ssi, all in different subplots
def get_ssi(bweight_mb, bweight_mf, abs=False):
    if abs:
        return np.abs((bweight_mb - bweight_mf) / (bweight_mb + bweight_mf))
    else:
        return (bweight_mb - bweight_mf) / (bweight_mb + bweight_mf)


for regr in encoder.tv_keys:
    fig, axes = plt.subplots(
        nrows=len(encoder_t.regions),
        ncols=encoder_t.num_bins,
        figsize=(2.25 * encoder_t.num_bins, 2.5 * len(encoder_t.regions)),
        sharey=True,
        tight_layout=True,
    )

    for i, reg in enumerate(encoder_t.regions):
        for j in range(encoder_t.num_bins):
            ax = axes[i][j]
            regr_t = f"{regr}_{j}"
            _, ax = plot_kdes(
                data={
                    reg: [
                        get_ssi(
                            bweights_sess_mb,
                            bweights_sess_mf,
                            abs=False,
                        )
                        for (bweights_sess_mb, bweights_sess_mf) in zip(
                            bweights_master[regr_t][reg]["mb"],
                            bweights_master[regr_t][reg]["mf"],
                        )
                    ]
                },
                do_sem=True,
                bw_method=0.1,
                label="ssi",
                ylabel="density",
                xlim=[-5, 5],
                ynorm=False,
                add_means=False,
                line_kwargs={reg: {"color": colors_region[reg]}},
                ax=ax,
            )
            ax.axvline(x=0, color="#555555", linewidth=0.75, linestyle="--", zorder=-1)
            ax.axvline(
                x=-1,
                color=colors_strategy["mf"],
                linewidth=0.75,
                linestyle="--",
                zorder=-1,
            )
            ax.axvline(
                x=1,
                color=colors_strategy["mb"],
                linewidth=0.75,
                linestyle="--",
                zorder=-1,
            )
            if i == 0:
                ax.set_title(encoder_t.epoch_keys_str[j])

    fig.suptitle(b_regr_tex(regr))

    for fext in ["svg", "png"]:
        save_fig(
            fig,
            FIGURES_DIR / "bweight" / "ssi" / "distros" / "time_resolved" / subj_id,
            fname=f"{regr}-bweight_ssi-time_resolved-{subj_id}_sessavg.{fext}",
        )

# alpha values between mb and mf (significant cells only)

reg_exp = encoder.max_reg
alpha_range = np.logspace(-reg_exp, reg_exp, 2 * reg_exp + 1)

for regr in encoder.tv_keys:
    fig, axes = plt.subplots(
        ncols=len(encoder.regions),
        figsize=(2.5 * len(encoder.regions), 2),
        sharey=True,
        tight_layout=True,
    )
    for i, reg in enumerate(encoder.regions):
        ax = axes[i]
        regr_t = f"{regr}_0"

        a_mb = np.concatenate(alphas[regr_t][reg]["mb"])
        a_mf = np.concatenate(alphas[regr_t][reg]["mf"])

        alpha_counts = np.array(
            [
                [len(np.where((a_mb == a1) & (a_mf == a2))[0]) for a2 in alpha_range]
                for a1 in alpha_range
            ]
        )

        im = ax.imshow(alpha_counts, cmap="Blues")

        ax.set_xlabel(r"mb $\alpha$")
        ax.set_ylabel(r"mf $\alpha$")
        if i == 0:
            ax.set_title(rf"{reg}, $\alpha$ count comparison", fontsize=7)

        ax.set_xticks(
            np.arange(len(alpha_range)),
            [f"{a:.0e}" for a in alpha_range],
            rotation=45,
            ha="right",
            fontsize=5,
        )
        ax.set_yticks(
            np.arange(len(alpha_range)), [f"{a:.0e}" for a in alpha_range], fontsize=5
        )

        fig.colorbar(im, ax=ax, shrink=0.67, label="count")
    fig.suptitle(rf"{b_regr_tex(regr)}, significant units")
    for fext in ["svg", "png"]:
        save_fig(
            fig,
            FIGURES_DIR / "alphas" / subj_id / "time_resolved",
            fname=f"{regr}-alphas_strategy_comp-time_resolved-{subj_id}_sesscomp.{fext}",
        )
