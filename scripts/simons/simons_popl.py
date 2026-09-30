"""
simons_nb.ipynb

figures for simons meeting

Author: Stellina X. Ao
Created: 2026-09-21
Last Modified: 2026-09-21
Python Version: 3.11.14
"""

import scienceplots  # noqa: F401
import shutup
import matplotlib.pyplot as plt
import numpy as np
from core.data import subject_ids, session_ids
from scipy.stats import wilcoxon, false_discovery_control
from utils.colors import colors_region
from utils.viz_utils import save_fig
from utils.paths import FIGURES_DIR
from sg.models import BootstrapperShuffle as BSS
from core.viz import plot_grouped_bar_h, plot_kdes

# snares, harmonica (tinny), bass, strings,
# stereo width/angle of drum kit
# syncopated instruments that accomplish the same things (bass and kick, string and snare)
# logic pro, hosken
# randomish eight beat accents

from sg.models import Encoder, StrategyEncoder, ShuffledEncoder

# pretty plots
plt.style.use(["nature"])
plt.rcParams["figure.dpi"] = 200

# suppress warnings :-)
shutup.please()

# subj_id
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

encoders = {k: [] for k in ["full", "mb", "mf"]}

for sess_id in session_ids[np.where(subject_ids == subj_id)[0][0]]:
    print(f">{sess_id}")
    e = Encoder(subj_id, sess_id, norm=True)
    e_mb = StrategyEncoder(subj_id, sess_id, norm=True, strategy_filter="mb")
    e_mf = StrategyEncoder(subj_id, sess_id, norm=True, strategy_filter="mf")

    try:
        e_mb.fit_encoder()
        e_mf.fit_encoder()

        e_mb.get_r2()
        e_mf.get_r2()

        e.fit_encoder()
        e.get_r2()
    except ValueError:
        print("not enough trials...")
        continue

    encoders["full"].append(e)
    encoders["mb"].append(e_mb)
    encoders["mf"].append(e_mf)

# r2
# non-parametric significance test between mb/mf explained variance, bonferroni for region

alpha = 0.05

ps = [
    wilcoxon(
        x=encoder_mb.scores["encoder"][encoder_mb.reg_idxs[reg]],
        y=encoder_mf.scores["encoder"][encoder_mf.reg_idxs[reg]],
    ).pvalue
    for reg in encoder.regions
]
ps_corr = false_discovery_control(ps)
ps_corr = {reg: ps_corr[i] for i, reg in enumerate(encoder.regions)}

print("sig diff r2 between mb and mf?")
for reg in encoder.regions:
    print(reg, ps_corr[reg] < alpha, f"({ps_corr[reg]:.3f})")

# distro figs

fig_full, ax_full = plot_kdes(
    data={
        reg: encoder.scores["encoder"][encoder.reg_idxs[reg]] for reg in encoder.regions
    },
    line_kwargs={reg: {"color": c} for reg, c in colors_region.items()},
    xlim=[-0.2, 1.0],
    label=r"$r^2$, full",
)
ax_full.axvline(x=0, color="#333333", linestyle="--")

fig_strat, ax_strat = plot_kdes(
    data={
        reg: encoder_mb.scores["encoder"][encoder_mb.reg_idxs[reg]]
        - encoder_mf.scores["encoder"][encoder_mf.reg_idxs[reg]]
        for reg in encoder.regions
    },
    line_kwargs={reg: {"color": c} for reg, c in colors_region.items()},
    xlim=[-1.0, 1.0],
    label=r"$r^2$, (mb-mf)",
)
ax_strat.axvline(x=0, color="#333333", linestyle="--")
ax_strat.set_title(f"DMS (p={ps_corr['DMS']:.3f}), DLS (p={ps_corr['DLS']:.3f})")

for fext in ["svg", "png"]:
    save_fig(
        fig_full,
        FIGURES_DIR / "r2" / "distros" / subj_id / sess_id,
        fname=f"r2_distro_full-{subj_id}_{sess_id}.{fext}",
    )
    save_fig(
        fig_strat,
        FIGURES_DIR / "r2" / "distros" / subj_id / sess_id,
        fname=f"r2_distro_strat-{subj_id}_{sess_id}.{fext}",
    )

# distro figs
fig_full, ax_full = plot_kdes(
    data={
        reg: [e.scores["encoder"][e.reg_idxs[reg]] for e in encoders["full"]]
        for reg in encoder.regions
    },
    do_sem=True,
    line_kwargs={reg: {"color": c} for reg, c in colors_region.items()},
    xlim=[-0.2, 1.0],
    label=r"$r^2$, full",
)
ax_full.axvline(x=0, color="#333333", linestyle="--")

fig_strat, ax_strat = plot_kdes(
    data={
        reg: [
            e_mb.scores["encoder"][e_mb.reg_idxs[reg]]
            - e_mf.scores["encoder"][e_mf.reg_idxs[reg]]
            for (e_mb, e_mf) in zip(encoders["mb"], encoders["mf"])
        ]
        for reg in encoder.regions
    },
    do_sem=True,
    line_kwargs={reg: {"color": c} for reg, c in colors_region.items()},
    xlim=[-1.0, 1.0],
    label=r"$r^2$, (mb-mf)",
)
ax_strat.axvline(x=0, color="#333333", linestyle="--")
ax_strat.set_title(f"DMS (p={ps_corr['DMS']:.3f}), DLS (p={ps_corr['DLS']:.3f})")

for fext in ["svg", "png"]:
    save_fig(
        fig_full,
        FIGURES_DIR / "r2" / "distros" / subj_id,
        fname=f"r2_distro_full-{subj_id}_sessavg.{fext}",
    )
    save_fig(
        fig_strat,
        FIGURES_DIR / "r2" / "distros" / subj_id,
        fname=f"r2_distro_strat-{subj_id}_sessavg.{fext}",
    )

# beta weight
# single session
for regr in encoder.tv_keys:
    # sig
    ps = [
        wilcoxon(
            x=encoder_mb.encoder_weights[
                encoder_mb.reg_idxs[reg], encoder_mb.tv_idxs[regr]
            ],
            y=encoder_mf.encoder_weights[
                encoder_mf.reg_idxs[reg], encoder_mf.tv_idxs[regr]
            ],
        ).pvalue
        for reg in encoder.regions
    ]
    ps_corr = false_discovery_control(ps)
    ps_corr = {reg: ps_corr[i] for i, reg in enumerate(encoder.regions)}

    regr_tex = regr.replace("_", r"\_")
    fig_full, ax_full = plot_kdes(
        data={
            reg: encoder.encoder_weights[encoder.reg_idxs[reg], encoder.tv_idxs[regr]]
            for reg in encoder.regions
        },
        line_kwargs={reg: {"color": c} for reg, c in colors_region.items()},
        xlim=[-1.0, 1.0],
        label=rf"$\beta_{{\mathrm{{{regr_tex}}}}}$, full",
    )
    ax_full.axvline(x=0, color="#333333", linestyle="--")

    fig_strat, ax_strat = plot_kdes(
        data={
            reg: encoder_mb.encoder_weights[
                encoder_mb.reg_idxs[reg], encoder_mb.tv_idxs[regr]
            ]
            - encoder_mf.encoder_weights[
                encoder_mf.reg_idxs[reg], encoder_mf.tv_idxs[regr]
            ]
            for reg in encoder.regions
        },
        line_kwargs={reg: {"color": c} for reg, c in colors_region.items()},
        xlim=[-1.0, 1.0],
        label=rf"$\beta_{{\mathrm{{{regr_tex}}}}}$, (mb-mf)",
    )
    ax_strat.axvline(x=0, color="#333333", linestyle="--")
    ax_strat.set_title(f"DMS (p={ps_corr['DMS']:.3f}), DLS (p={ps_corr['DLS']:.3f})")

    for fext in ["svg", "png"]:
        save_fig(
            fig_full,
            FIGURES_DIR / "bweight" / "distros" / subj_id / sess_id,
            fname=f"{regr}-bweight_distro_full-{subj_id}_{sess_id}.{fext}",
        )
        save_fig(
            fig_strat,
            FIGURES_DIR / "bweight" / "distros" / subj_id / sess_id,
            fname=f"{regr}-bweight_distro_strat-{subj_id}_{sess_id}.{fext}",
        )

ps_corr = {regr: {} for regr in encoder.tv_keys}
alpha = 0.05

for regr in encoder.tv_keys:
    ps = [
        wilcoxon(
            x=encoder_mb.encoder_weights[
                encoder_mb.reg_idxs[reg], encoder_mb.tv_idxs[regr]
            ],
            y=encoder_mf.encoder_weights[
                encoder_mf.reg_idxs[reg], encoder_mf.tv_idxs[regr]
            ],
        ).pvalue
        for reg in encoder.regions
    ]
    ps_corr_ = false_discovery_control(ps)
    ps_corr[regr] = {reg: ps_corr_[i] for i, reg in enumerate(encoder.regions)}

print("sig diff bweight between mb and mf?")
for regr in encoder.tv_keys:
    print(regr)
    for reg in encoder.regions:
        print(f"\t {reg}, {ps_corr[regr][reg] < alpha}, ({ps_corr[regr][reg]:.3f})")

# across sessions
for regr in encoder.tv_keys:
    regr_tex = regr.replace("_", r"\_")
    fig_full, ax_full = plot_kdes(
        data={
            reg: [
                e.encoder_weights[e.reg_idxs[reg], e.tv_idxs[regr]]
                for e in encoders["full"]
            ]
            for reg in encoder.regions
        },
        do_sem=True,
        line_kwargs={reg: {"color": c} for reg, c in colors_region.items()},
        xlim=[-1.0, 1.0],
        label=rf"$\beta_{{\mathrm{{{regr_tex}}}}}$, full",
    )
    ax_full.axvline(x=0, color="#333333", linestyle="--")

    fig_strat, ax_strat = plot_kdes(
        data={
            reg: [
                e_mb.encoder_weights[e_mb.reg_idxs[reg], e_mb.tv_idxs[regr]]
                - e_mf.encoder_weights[e_mf.reg_idxs[reg], e_mf.tv_idxs[regr]]
                for (e_mb, e_mf) in zip(encoders["mb"], encoders["mf"])
            ]
            for reg in encoder.regions
        },
        do_sem=True,
        line_kwargs={reg: {"color": c} for reg, c in colors_region.items()},
        xlim=[-1.0, 1.0],
        label=rf"$\beta_{{\mathrm{{{regr_tex}}}}}$, (mb-mf)",
    )
    ax_strat.axvline(x=0, color="#333333", linestyle="--")

    for fext in ["svg", "png"]:
        save_fig(
            fig_full,
            FIGURES_DIR / "bweight" / "distros" / subj_id,
            fname=f"{regr}-bweight_distro_full-{subj_id}_sessavg.{fext}",
        )
        save_fig(
            fig_strat,
            FIGURES_DIR / "bweight" / "distros" / subj_id,
            fname=f"{regr}-bweight_distro_strat-{subj_id}_sessavg.{fext}",
        )

# cv/dr2
# all sessions
ses = {k: [] for k in ["full", "mb", "mf"]}

for sess_id in session_ids[np.where(subject_ids == subj_id)[0][0]]:
    print(f">{sess_id}")

    try:
        se_mb_ = ShuffledEncoder(
            subj_id,
            sess_id,
            enc_class=StrategyEncoder,
            strategy_filter="mb",
            tv_keys=[
                "response",
                "rewarded",
                "response_prev",
                "rewarded_prev",
            ],
        )

        se_mf_ = ShuffledEncoder(
            subj_id,
            sess_id,
            enc_class=StrategyEncoder,
            strategy_filter="mf",
            tv_keys=[
                "response",
                "rewarded",
                "response_prev",
                "rewarded_prev",
            ],
        )

        se_ = ShuffledEncoder(
            subj_id,
            sess_id,
            enc_class=Encoder,
            tv_keys=[
                "response",
                "rewarded",
                "response_prev",
                "rewarded_prev",
            ],
        )

        print(">> mb, cvr2")
        se_mb_.get_cvr2_all(n_iters=8)
        print(">> mb, dr2")
        se_mb_.get_dr2_all(n_iters=8)

        print(">> mf, cvr2")
        se_mf_.get_cvr2_all(n_iters=8)
        print(">> mf, dr2")
        se_mf_.get_dr2_all(n_iters=8)

        print(">> full, cvr2")
        se_.get_cvr2_all(n_iters=8)
        print(">> full, dr2")
        se_.get_dr2_all(n_iters=8)
    except ValueError:
        print("not enough trials...")
        continue

    ses["full"].append(se_)
    ses["mb"].append(se_mb_)
    ses["mf"].append(se_mf_)

# cvr2

# across sessions
for regr in encoder.tv_keys:
    if regr != "block_side":
        regr_tex = regr.replace("_", r"\_")
        fig_full, ax_full = plot_kdes(
            data={
                reg: [
                    se_.cvr2_unit[regr][:, encoders["full"][i].reg_idxs[reg]].mean(
                        axis=0
                    )
                    for i, se_ in enumerate(ses["full"])
                ]
                for reg in encoder.regions
            },
            do_sem=True,
            line_kwargs={reg: {"color": c} for reg, c in colors_region.items()},
            xlim=[-0.2, 0.8],
            label=rf"$\text{{cv }} r^2_{{\mathrm{{{regr_tex}}}}}$, full",
        )
        ax_full.axvline(x=0, color="#333333", linestyle="--")

        fig_strat, ax_strat = plot_kdes(
            data={
                reg: [
                    se_mb_.cvr2_unit[regr][:, encoders["mb"][i].reg_idxs[reg]].mean(
                        axis=0
                    )
                    - se_mf_.cvr2_unit[regr][:, encoders["mf"][i].reg_idxs[reg]].mean(
                        axis=0
                    )
                    for i, (se_mb_, se_mf_) in enumerate(zip(ses["mb"], ses["mf"]))
                ]
                for reg in encoder.regions
            },
            do_sem=True,
            line_kwargs={reg: {"color": c} for reg, c in colors_region.items()},
            xlim=[-0.8, 0.8],
            label=rf"$\text{{cv }} r^2_{{\mathrm{{{regr_tex}}}}}$, (mb-mf)",
        )
        ax_strat.axvline(x=0, color="#333333", linestyle="--")

        for fext in ["svg", "png"]:
            save_fig(
                fig_full,
                FIGURES_DIR / "cv_d_r2" / "cv" / "distros" / subj_id,
                fname=f"{regr}-cvr2_distro_full-{subj_id}_sessavg.{fext}",
            )
            save_fig(
                fig_strat,
                FIGURES_DIR / "cv_d_r2" / "cv" / "distros" / subj_id,
                fname=f"{regr}-cvr2_distro_strat-{subj_id}_sessavg.{fext}",
            )

# delta r2
for regr in encoder.tv_keys:
    if regr != "block_side":
        regr_tex = regr.replace("_", r"\_")
        fig_full, ax_full = plot_kdes(
            data={
                reg: [
                    se_.dr2_unit[regr][:, encoders["full"][i].reg_idxs[reg]].mean(
                        axis=0
                    )
                    for i, se_ in enumerate(ses["full"])
                ]
                for reg in encoder.regions
            },
            do_sem=True,
            line_kwargs={reg: {"color": c} for reg, c in colors_region.items()},
            xlim=[-0.05, 0.1],
            label=rf"$\text{{cv }} r^2_{{\mathrm{{{regr_tex}}}}}$, full",
        )
        ax_full.axvline(x=0, color="#333333", linestyle="--")

        fig_strat, ax_strat = plot_kdes(
            data={
                reg: [
                    se_mb_.dr2_unit[regr][:, encoders["mb"][i].reg_idxs[reg]].mean(
                        axis=0
                    )
                    - se_mf_.dr2_unit[regr][:, encoders["mf"][i].reg_idxs[reg]].mean(
                        axis=0
                    )
                    for i, (se_mb_, se_mf_) in enumerate(zip(ses["mb"], ses["mf"]))
                ]
                for reg in encoder.regions
            },
            do_sem=True,
            line_kwargs={reg: {"color": c} for reg, c in colors_region.items()},
            xlim=[-0.3, 0.3],
            label=rf"$\text{{cv }} r^2_{{\mathrm{{{regr_tex}}}}}$, (mb-mf)",
        )
        ax_strat.axvline(x=0, color="#333333", linestyle="--")

        for fext in ["svg", "png"]:
            save_fig(
                fig_full,
                FIGURES_DIR / "cv_d_r2" / "delta" / "distros" / subj_id,
                fname=f"{regr}-dr2_distro_full-{subj_id}_sessavg.{fext}",
            )
            save_fig(
                fig_strat,
                FIGURES_DIR / "cv_d_r2" / "delta" / "distros" / subj_id,
                fname=f"{regr}-dr2_distro_strat-{subj_id}_sessavg.{fext}",
            )

del ses

# sig pi chart

p_sig_union_sess = {
    regr: {reg: [] for reg in encoder.regions} for regr in encoder.tv_keys
}

sess_idx = 0
for sess_id in session_ids[np.where(subject_ids == subj_id)[0][0]]:
    print(f">{sess_id}")
    try:
        bss_mb_ = BSS(
            subj_id,
            sess_id,
            StrategyEncoder,
            strategy_filter="mb",
            n=8,
            norm=True,
        )
        bss_mb_.get_ci_idxs()

        bss_mf_ = BSS(
            subj_id,
            sess_id,
            StrategyEncoder,
            strategy_filter="mf",
            n=8,
            norm=True,
        )
        bss_mf_.get_ci_idxs()

        bss_ = BSS(
            subj_id,
            sess_id,
            Encoder,
            n=8,
            norm=True,
        )
        bss_.get_ci_idxs()
    except ValueError:
        print("not enough trials")
        continue

    # union
    cids_ = {
        regr: {
            reg: np.sort(
                np.unique(
                    np.union1d(
                        bss_mb_.ci_idxs_reg[regr][reg], bss_mf_.ci_idxs_reg[regr][reg]
                    )
                )
            )
            for reg in encoder.regions
        }
        for regr in encoder.tv_keys
    }

    for regr in encoder.tv_keys:
        for reg in encoder.regions:
            p_sig_union_sess[regr][reg].append(
                len(cids_[regr][reg]) / len(encoders["full"][sess_idx].psths[reg])
            )

    sess_idx += 1


fig_union, ax = plot_grouped_bar_h(
    data=p_sig_union_sess,
    do_sem=True,
    ylabel="p(significant) [union]",
    colors=colors_region,
    legend=False,
)
ax.axvline(x=1.0, color="#333333", linewidth=0.75)

for fext in ["svg", "png"]:
    save_fig(
        fig_union,
        FIGURES_DIR / "bweight" / subj_id / "p_significant",
        f"p_significant_union-{subj_id}_sessavg.{fext}",
    )
