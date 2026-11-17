import pickle
import numpy as np
import matplotlib.pyplot as plt

from sg.models import ShuffledEncoder, StrategyEncoder, make_tre
from core.data import default_sessions, subject_ids, session_ids
from utils.paths import VARS_DIR, FIGURES_DIR
from core.viz import plot_grouped_boxes
from utils.colors import colors_region
from utils.viz_utils import save_fig

for subj_id in ["MR82", "MR83", "MR94", "MR95"]:
    print(subj_id)
    encoder_mb = make_tre(StrategyEncoder, tr_type="dme")(
        subj_id,
        default_sessions[subj_id],
        norm=True,
        stepsize_s=0.5,
        strategy_filter="mb",
        strategy_params="mb",
        add_interaction=True,
        # add_svd=True,
        # num_svd=10,
    )
    encoder_mb.verify()

    encoder_mf = make_tre(StrategyEncoder, tr_type="dme")(
        subj_id,
        default_sessions[subj_id],
        norm=True,
        stepsize_s=0.5,
        strategy_filter="mf",
        strategy_params="mf",
        add_interaction=True,
        # add_svd=True,
        # num_svd=10,
    )
    encoder_mf.verify()

    pivots_groups_mb = {
        regr: [f"{regr}_{i}" for i in range(encoder_mb.num_bins)]
        for regr in np.concatenate(
            (encoder_mb.tv_keys, encoder_mb.tv_names["interaction"])
        )
    }
    pivots_groups_mb["spe"] = [f"spe_{i}" for i in range(encoder_mb.num_bins)]
    # pivots_groups_mb["svd"] = [f"svd_{i}" for i in range(encoder_mb.num_svd)]

    pivots_groups_mf = {
        regr: [f"{regr}_{i}" for i in range(encoder_mf.num_bins)]
        for regr in np.concatenate(
            (encoder_mf.tv_keys, encoder_mf.tv_names["interaction"])
        )
    }
    pivots_groups_mf["rpe"] = [f"rpe_{i}" for i in range(encoder_mf.num_bins)]
    # pivots_groups_mf["svd"] = [f"svd_{i}" for i in range(encoder_mf.num_svd)]

    cvr2s = {
        strat: {
            regr: {reg: [] for reg in encoder_mb.regions}
            for regr in pivots_groups_.keys()
        }
        for strat, pivots_groups_ in {
            "mb": pivots_groups_mb,
            "mf": pivots_groups_mf,
        }.items()
    }
    dr2s = {
        strat: {
            regr: {reg: [] for reg in encoder_mb.regions}
            for regr in pivots_groups_.keys()
        }
        for strat, pivots_groups_ in {
            "mb": pivots_groups_mb,
            "mf": pivots_groups_mf,
        }.items()
    }

    for sess_id in session_ids[np.where(subject_ids == subj_id)[0][0]]:
        print(f"> {sess_id}")
        try:
            se_mb = ShuffledEncoder(
                subj_id,
                sess_id,
                enc_class=make_tre(StrategyEncoder, tr_type="dme"),
                strategy_filter="mb",
                strategy_params="mb",
                add_interaction=True,
                norm=True,
                # add_svd=True,
                # num_svd=10,
                stepsize_s=0.5,
            )

            se_mf = ShuffledEncoder(
                subj_id,
                sess_id,
                enc_class=make_tre(StrategyEncoder, tr_type="dme"),
                strategy_filter="mf",
                strategy_params="mf",
                add_interaction=True,
                norm=True,
                # add_svd=True,
                # num_svd=10,
                stepsize_s=0.5,
            )

            se_mb.get_cvr2_groups(pivots_groups=pivots_groups_mb, n_iters=5)
            se_mb.get_dr2_groups(pivots_groups=pivots_groups_mb, n_iters=5)

            se_mf.get_cvr2_groups(pivots_groups=pivots_groups_mf, n_iters=5)
            se_mf.get_dr2_groups(pivots_groups=pivots_groups_mf, n_iters=5)
        except ValueError as e:
            print(f"{subj_id} {sess_id} skipped: {e}")
            continue

        assert all(se_mb.encoder_full.regions == se_mf.encoder_full.regions), "ruh roh"

        for reg in se_mb.encoder_full.regions:
            for strat, (se, pivots_groups_) in {
                "mb": (se_mb, pivots_groups_mb),
                "mf": (se_mf, pivots_groups_mf),
            }.items():
                for regr in pivots_groups_.keys():
                    cvr2s[strat][regr][reg].append(
                        se.cvr2_unit[regr].mean(axis=0)[se.encoder_full.reg_idxs[reg]]
                    )
                    dr2s[strat][regr][reg].append(
                        se.dr2_unit[regr].mean(axis=0)[se.encoder_full.reg_idxs[reg]]
                    )

        fpath = VARS_DIR / "cv_d_r2" / "grouped" / "no_svd"
        fpath.mkdir(exist_ok=True, parents=True)
        with open(fpath / f"{subj_id}_cvr2s.pkl", "wb") as f:
            pickle.dump(cvr2s, f)
        with open(fpath / f"{subj_id}_dr2s.pkl", "wb") as f:
            pickle.dump(dr2s, f)

    fig, axes = plt.subplots(
        ncols=2, nrows=2, figsize=(5, 5), sharey=True, tight_layout=True
    )

    for i, tv_type in enumerate(["base_regressors", "interaction"]):
        for j, (strategy, (metrics, regressors_)) in enumerate(
            {
                "mb": (cvr2s["mb"], encoder_mb.tv_names[tv_type]),
                "mf": (cvr2s["mf"], encoder_mf.tv_names[tv_type]),
            }.items()
        ):
            _, ax = plot_grouped_boxes(
                data={
                    group: {
                        reg: np.concatenate(metrics[group][reg])
                        for reg in encoder_mb.regions
                    }
                    for group in regressors_
                },
                colors=colors_region,
                ax=axes[i][j],
            )
            ax.legend(loc="upper right")
            ax.set_title(strategy)

    fig.suptitle(rf"{subj_id}, cv $r^2$")
    save_fig(
        fig,
        FIGURES_DIR / "cv_d_r2" / "boxplot_comp" / "grouped" / "no_svd",
        f"{subj_id}_cvr2_comp.png",
    )

    fig, axes = plt.subplots(
        ncols=2, nrows=2, figsize=(5, 5), sharey=True, tight_layout=True
    )
    for i, tv_type in enumerate(["base_regressors", "interaction"]):
        for j, (strategy, (metrics, regressors_)) in enumerate(
            {
                "mb": (dr2s["mb"], encoder_mb.tv_names[tv_type]),
                "mf": (dr2s["mf"], encoder_mf.tv_names[tv_type]),
            }.items()
        ):
            _, ax = plot_grouped_boxes(
                data={
                    group: {
                        reg: np.concatenate(metrics[group][reg])
                        for reg in encoder_mb.regions
                    }
                    for group in regressors_
                },
                colors=colors_region,
                ax=axes[i][j],
            )
            ax.legend(loc="upper right")
            ax.set_title(strategy)

    fig.suptitle(rf"{subj_id}, $\Delta r^2$")
    save_fig(
        fig,
        FIGURES_DIR / "cv_d_r2" / "boxplot_comp" / "grouped" / "no_svd",
        f"{subj_id}_dr2_comp.png",
    )
