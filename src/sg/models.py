import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from typing import Type

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from scipy.stats import zscore

from core.data import (
    load_sess,
    get_strategy_filter_idxs,
    get_psths_ref,
    get_encoder_io,
    get_tavg_sc_cond,
    get_choice_ts,
    get_psths_cond,
    tv_vals,
)

from core.viz import plot_raincloud

from squiggs.neuron_viewer import NeuronViewer
from squiggs.renderers import (
    FitRenderer,
    FitRendererTime,
    PETHWeightRenderer,
    PETHWeightRendererTime,
    PETHRasterRenderer,
)
from utils.paths import FIGURES_DIR


class Encoder:
    def __init__(
        self,
        subj_id: str = None,
        sess_id: str = None,
        **kwargs,
    ):
        self.subj_id = subj_id
        self.sess_id = sess_id

        self.random_state = kwargs.pop("random_state", 1024)

        self.tv_keys = kwargs.pop(
            "tv_keys",
            [
                "response",
                "rewarded",
                "block_side",
                "response_prev",
                "rewarded_prev",
            ],
        )
        self.add_svd = kwargs.pop("add_svd", False)
        if self.add_svd:
            self.num_svd = kwargs.pop("num_svd", 10)
        self.add_licks = kwargs.pop("add_licks", False)

        self.n = kwargs.pop("n", None)
        self.full_trial = kwargs.pop("full_trial", False)

        self.norm = kwargs.pop("norm", False)
        self.max_reg = kwargs.pop("max_reg", 5)

        self.num_tents = kwargs.pop("num_tents", 5)

        self.tpre = kwargs.pop("tpre", 0.5)
        self.tpost = kwargs.pop("tpost", 1)
        self.alignment = kwargs.pop("alignment", "choice")
        self.edge_inclusive = kwargs.pop("edge_inclusive", True)

        self.tpre_ref = kwargs.pop("tpre_ref", 0.5)
        self.tpost_ref = kwargs.pop("tpost_ref", 1)
        self.alignment_ref = kwargs.pop("alignment_ref", "choice")

        self.binwidth_ms = kwargs.pop("binwidth_ms", 25)
        self.thresh = kwargs.pop("thresh", 1)

        if len(kwargs) > 0:
            extra_kwargs = ", ".join('"%s' % k for k in list(kwargs.keys()))
            raise ValueError("Extra arguments %s" % extra_kwargs)

    def seed(self):
        np.random.seed(self.random_state)

    def get_data(self):
        def _edge_inclusive(t):
            return t + 0.001 if self.edge_inclusive else t

        (
            self.spike_times,
            self.trial_data,
            self.psths,
            self.session_data,
            self.regions,
            self.tbin_edges,
        ) = load_sess(
            subj_id=self.subj_id,
            sess_id=self.sess_id,
            tpre=_edge_inclusive(self.tpre),
            tpost=_edge_inclusive(self.tpost),
            alignment=self.alignment,
            tpre_ref=_edge_inclusive(self.tpre_ref),
            tpost_ref=_edge_inclusive(self.tpost_ref),
            alignment_ref=self.alignment_ref,
            binwidth_ms=self.binwidth_ms,
            add_svd=self.add_svd,
            add_licks=self.add_licks,
            full_trial=self.full_trial,
            thresh=self.thresh,
        )

        self.tbin_centers = (self.tbin_edges - (self.binwidth_ms / (2 * 1000)))[1:]
        if self.n is not None:
            self.idxs = np.sort(
                np.random.choice(len(self.trial_data), self.n, replace=False)
            )
            self.trial_data = self.trial_data.iloc[self.idxs]
            self.psths = {reg: self.psths[reg][:, self.idxs, :] for reg in self.regions}

    def build_dm(self):
        if not (hasattr(self, "psths")):
            self.get_data()
        (
            self.tents,
            self.tvs,
            self.dm,
            self.robs,
            self.dm_names,
            self.dm_idxs,
            self.reg_idxs,
        ) = get_encoder_io(
            self.psths,
            self.trial_data,
            self.regions,
            norm=self.norm,
            num_tents=self.num_tents,
            tv_keys=self.tv_keys,
            add_svd=self.add_svd,
            num_svd=self.num_svd if self.add_svd else None,
            add_licks=self.add_licks,
            binwidth_ms=self.binwidth_ms,
        )

        self.num_trials, self.num_tv = self.tvs.shape
        self.num_units = self.robs.shape[1]

    def fit_baseline(self, robs=None):
        if not (hasattr(self, "tents") and hasattr(self, "robs")):
            self.build_dm()

        robs = self.robs if robs is None else robs

        self.baseline_model = RidgeCV(
            alphas=np.logspace(
                -self.max_reg, self.max_reg, 2 * self.max_reg + 1, base=10
            ),
            alpha_per_target=True,
        ).fit(self.tents, robs)

    def baseline_predict(self):
        if not hasattr(self, "robs_predict"):
            self.robs_predict = {}

        if not hasattr(self, "baseline_model"):
            self.fit_baseline()

        self.robs_predict["baseline"] = self.baseline_model.predict(self.tents)

    def fit_encoder(self):
        if not (hasattr(self, "robs")):
            self.build_dm()

        if (
            not hasattr(self, "robs_predict")
            or "baseline" not in self.robs_predict.keys()
        ):
            self.baseline_predict()

        self.encoder = RidgeCV(
            alphas=np.logspace(
                -self.max_reg, self.max_reg, 2 * self.max_reg + 1, base=10
            ),
            alpha_per_target=True,
        ).fit(self.tvs, self.robs - self.robs_predict["baseline"])

        self.encoder_weights = np.hstack(
            (self.baseline_model.coef_, self.encoder.coef_)
        )

    def encoder_predict(self):
        if (
            not hasattr(self, "robs_predict")
            or "baseline" not in self.robs_predict.keys()
        ):
            self.baseline_predict()

        if not hasattr(self, "encoder"):
            self.fit_encoder()

        self.robs_predict["encoder"] = self.robs_predict[
            "baseline"
        ] + self.encoder.predict(self.tvs)

    def _get_predictions(self, is_encoder, **kwargs):
        if is_encoder:
            return self._get_predictions_encoder(**kwargs)
        else:
            return self._get_predictions_baseline(**kwargs)

    def _get_predictions_baseline(self, idxs, io, model=None):
        train_idxs, test_idxs = idxs
        dm, robs = io

        if model is None:
            model = RidgeCV(
                alphas=np.logspace(
                    -self.max_reg, self.max_reg, 2 * self.max_reg + 1, base=10
                ),
                alpha_per_target=True,
            ).fit(dm[train_idxs], robs[train_idxs])
        return model.predict(dm[test_idxs]), model

    def _get_predictions_encoder(
        self, idxs, io, baseline_model=None, baseline_predict=None
    ):
        train_idxs, test_idxs = idxs
        if baseline_predict is not None:
            tvs, robs = io
        else:
            tents, tvs, robs = io

        if baseline_predict is None:
            baseline_predict = baseline_model.predict(tents)

        encoder = RidgeCV(
            alphas=np.logspace(
                -self.max_reg, self.max_reg, 2 * self.max_reg + 1, base=10
            ),
            alpha_per_target=True,
        ).fit(
            tvs[train_idxs],
            robs[train_idxs] - baseline_predict[train_idxs],
        )

        return baseline_predict[test_idxs] + encoder.predict(tvs[test_idxs]), encoder

    def get_r2(self, n_folds=20, p_train=0.8):
        if not hasattr(self, "robs"):
            self.build_dm()

        self.yhats = {
            k: np.zeros((self.num_trials, self.num_units))
            for k in ["baseline", "encoder"]
        }

        for i, (train_idxs, test_idxs) in enumerate(
            KFold(n_splits=n_folds, shuffle=True, random_state=0).split(self.robs)
        ):
            # train, test, save test predictions
            self.yhats["baseline"][test_idxs], baseline_model = self._get_predictions(
                is_encoder=False,
                io=(self.tents, self.robs),
                idxs=(train_idxs, test_idxs),
            )

            self.yhats["encoder"][test_idxs], _ = self._get_predictions(
                is_encoder=True,
                io=(self.tents, self.tvs, self.robs),
                idxs=(train_idxs, test_idxs),
                baseline_model=baseline_model,
            )

        self.scores = {
            k: r2_score(self.robs, yhat, multioutput="raw_values", force_finite=False)
            for k, yhat in self.yhats.items()
        }

    def get_weights(self, regr, val=None):
        if val is not None:
            return self.encoder_weights[:, self.dm_idxs[f"{regr}_{val}"]]
        else:
            if regr in self.tv_keys:
                idxs = [self.dm_idxs[f"{regr}_{val}"] for val in tv_vals[regr]]
            elif regr == "tents":
                idxs = [self.dm_idxs[f"{regr}_{val}"] for val in range(self.num_tents)]
            return self.encoder_weights[:, idxs]

    def verify(self, r2_comp=True, subtract_baseline=True):
        ncols = 6 if self.tv_keys is not None else 2
        _, axes = plt.subplots(
            ncols=ncols, nrows=1, figsize=(ncols * 1.5, 1.5), tight_layout=True
        )

        # baseline vs encoder r2
        if r2_comp:
            self.plot_r2_comp(axes[0])
        else:
            self.plot_r2_distro(axes[0])

        # p(resp)
        # self.plot_p_resp(axes[1]) FLAG: out of service for now
        # from core.viz import plot_kdes

        # scores_no_ps = deepcopy(self.scores)
        # if "ps_baseline" in scores_no_ps.keys():
        #     scores_no_ps.pop("ps_baseline")
        # plot_kdes(scores_no_ps, label=r"$r^2$", ax=axes[1])

        # sctavg vs beta weight
        print("oop")
        if self.tv_keys is not None:
            self.plot_sctavg_weights(
                axes[2:4], cond="response", subtract_baseline=subtract_baseline
            )
            self.plot_sctavg_weights(
                axes[4:6], cond="rewarded", subtract_baseline=subtract_baseline
            )

    def plot_r2_distro(self, ax=None):
        if not hasattr(self, "scores"):
            self.get_r2()

        if ax is None:
            _, ax = plt.subplots(tight_layout=True)

        plot_raincloud(self.scores["encoder"], label=r"$r^2$, encoder", ax=ax)

    def plot_r2_comp(self, ax=None):
        if not hasattr(self, "scores"):
            try:
                self.get_r2()
            except NotImplementedError:
                return

        if ax is None:
            _, ax = plt.subplots(tight_layout=True)

        ax.scatter(self.scores["baseline"], self.scores["encoder"], s=0.5, alpha=0.5)
        ax.plot([-0.5, 1], [-0.5, 1], color="#666666", linestyle="--", linewidth=0.5)
        ax.plot([-0.5, 1], [-0.5, 1], color="#666666", linestyle="--", linewidth=0.5)
        ax.axhline(y=0, color="k", linewidth=0.5)
        ax.axvline(x=0, color="k", linewidth=0.5)

        ax.set_xlabel(r"$r^2$, drift")
        ax.set_ylabel(r"$r^2$, encoder")

    def get_sctavg_weights(self, cond="response", subtract_baseline=True):
        if cond not in ["response", "rewarded"]:
            raise NotImplementedError(f"cond={cond} is not currently supported.")
        if not hasattr(self, "encoder"):
            self.fit_encoder()

        if not hasattr(self, "sc_tavg"):
            self.sc_tavg = {}

        if cond in self.sc_tavg.keys():
            return
        if subtract_baseline:
            # get baseline explained variance
            if (
                not hasattr(self, "robs_predict")
                or "baseline" not in self.robs_predict.keys()
            ):
                self.baseline_predict()
            robs_baseline = self.robs_predict["baseline"]
            self.robs_baseline = robs_baseline

            # get tv explained variance (besides pivot)
            pivot_idxs = [
                self.dm_idxs[regr] - self.num_tents
                for regr in [f"{cond}_{val}" for val in tv_vals[cond]]
            ]
            self.tv_pivot_ko = deepcopy(self.tvs)
            self.tv_pivot_ko[:, pivot_idxs] = 0

            robs_tv = self.encoder.predict(self.tv_pivot_ko)
            self.robs_to_subtract = robs_baseline + robs_tv

        else:
            # robs_to_subtract = None
            pass

        a = get_tavg_sc_cond(
            self.robs,
            self.trial_data,
            cond=cond,
            robs_to_subtract=self.robs_to_subtract,
            subtract_robs=subtract_baseline,
        )
        self.sc_tavg[cond] = a

    def plot_sctavg_weights(self, axes, cond="response", subtract_baseline=True):
        if not hasattr(self, "sc_tavg") or cond not in self.sc_tavg.keys():
            self.get_sctavg_weights(cond=cond, subtract_baseline=subtract_baseline)
        print("ello guvna")
        if axes is None:
            fig, axes = plt.subplots(
                ncols=2, nrows=1, figsize=(4, 2), tight_layout=True
            )

        keys = tv_vals[cond]
        idxs = [self.dm_idxs[regr] for regr in [f"{cond}_{key}" for key in keys]]

        for i in range(2):
            axes[i].scatter(
                self.sc_tavg[cond][keys[i]],
                self.encoder_weights[:, idxs[i]],
                s=0.5,
                alpha=0.5,
                vmin=1e-5,
                vmax=1e5,
                c=self.encoder.alpha_,
                cmap="viridis",
                norm="log",
            )

            mn = min(
                min(self.sc_tavg[cond][keys[i]]), min(self.encoder_weights[:, idxs[i]])
            )
            mx = max(
                max(self.sc_tavg[cond][keys[i]]), max(self.encoder_weights[:, idxs[i]])
            )
            axes[i].plot(
                [1.05 * mn, 1.05 * mx],
                [1.05 * mn, 1.05 * mx],
                color="#666666",
                linewidth=0.7,
                linestyle="--",
                zorder=0,
            )
            axes[i].axhline(y=0, color="k", linewidth=0.5)
            axes[i].axvline(x=0, color="k", linewidth=0.5)

            axes[i].set_xlabel(f"resid spk count, {keys[i]}")
            axes[i].set_ylabel(f"beta weight, {keys[i]}")

    def get_resp_units(self):
        if not hasattr(self, "scores"):
            self.get_r2()

        self.resp_idxs = np.where(
            (self.scores["encoder"] > self.scores["baseline"])
            & (self.scores["encoder"] > 0)
        )[0]
        self.p_resp = len(self.resp_idxs) / self.num_units

    def plot_p_resp(self, ax):
        if ax is None:
            _, ax = plt.figure(tight_layout=True)

        if not hasattr(self, "resp_idxs"):
            self.get_resp_units()

        ax.pie(
            [self.num_units - len(self.resp_idxs), len(self.resp_idxs)],
            colors=["#666666", "#F1AeAe"],
            autopct="%.1f%%",
            startangle=90,
        )

    def view_fits(self, reg="DLS", model="encoder"):
        if reg not in self.regions:
            raise ValueError(f"{reg} must be in {self.regions}")

        if not hasattr(self, "robs_predict") or model not in self.robs_predict.keys():
            if model == "encoder":
                self.encoder_predict()
            elif model == "baseline":
                self.baseline_predict()
            else:
                raise ValueError(
                    f"valid arguments for model are 'encoder' and 'baseline,' not {model}"
                )

        if not hasattr(self, "scores"):
            self.get_r2()

        r = FitRenderer(
            y=self.robs[:, self.reg_idxs[reg]],
            yhat=self.robs_predict[model][:, self.reg_idxs[reg]],
            rsquared=self.scores[model][self.reg_idxs[reg]],
            mode="lite",
        )

        return NeuronViewer(
            num_units=self.psths[reg].shape[0], render_func=r, fig_dir=FIGURES_DIR
        )

    def view_weights(self, reg="DLS", mode="response"):
        if reg not in self.regions:
            raise ValueError(f"{reg} must be in {self.regions}")

        if not hasattr(self, "encoder"):
            self.fit_encoder()

        sc_tavg = get_tavg_sc_cond(
            self.robs[:, self.reg_idxs[reg]], self.trial_data, cond=mode
        )

        r = PETHWeightRenderer(
            weights=self.encoder_weights[self.reg_idxs[reg], :],
            weight_names=self.dm_names,
            robs=self.robs[:, self.reg_idxs[reg]],
            sc_tavg=sc_tavg,
            event_times=get_choice_ts(self.trial_data, mode=mode),
            spike_times=self.spike_times[reg],
            peths=get_psths_cond(self.psths[reg], self.trial_data, mode=mode),
            pres=self.tpre,
            posts=self.tpost,
            binwidth_s=self.binwidth_ms / 1000,
            tbin_edges=self.tbin_centers,
        )

        return NeuronViewer(
            num_units=self.psths[reg].shape[0], render_func=r, fig_dir=FIGURES_DIR
        )

    def view_peths(self, reg="DLS", mode="response", flag=True):
        if reg not in self.regions:
            raise ValueError(f"{reg} must be in {self.regions}")

        if not hasattr(self, "psths"):
            self.get_data()

        r = PETHRasterRenderer(
            event_times=get_choice_ts(self.trial_data, mode=mode),
            spike_times=self.spike_times[reg],
            peths=get_psths_cond(self.psths[reg], self.trial_data, mode=mode),
            pres=self.tpre,
            posts=self.tpost,
            binwidth_s=self.binwidth_ms / 1000,
            tbin_edges=self.tbin_edges,
            s=0.5,
            linewidths=0.5,
        )

        return NeuronViewer(
            num_units=self.reg_idxs[reg].shape[0], render_func=r, fig_dir=FIGURES_DIR
        )


class StrategyEncoder(Encoder):
    def __init__(
        self,
        subj_id,
        sess_id,
        strategy_filter,
        **kwargs,
    ):
        self.subj_id = subj_id
        self.sess_id = sess_id

        self.strategy_filter = strategy_filter
        self.balance_strategy = kwargs.pop("balance_strategy", True)
        if self.balance_strategy:
            self.cond_balance = kwargs.pop("cond_balance", False)
        self.min_num_trials = kwargs.pop("min_num_trials", 20)

        self.idxs = kwargs.pop("idxs", None)

        if not (self.strategy_filter == "mb" or self.strategy_filter == "mf"):
            raise ValueError("strategy_filter must be mb or mf")

        self.encoder_ref = Encoder(subj_id=subj_id, sess_id=sess_id, **kwargs)

        super().__init__(
            subj_id,
            sess_id,
            num_tents=self.encoder_ref.num_tents,
            **kwargs,
        )

    def get_data(self):
        super().get_data()

        if self.idxs is None:
            self.idxs_all = get_strategy_filter_idxs(
                self.trial_data,
                self.strategy_filter,
                balance_strategy=self.balance_strategy,
                cond_balance=self.cond_balance if self.balance_strategy else False,
            )
            self.idxs = self.idxs_all[self.strategy_filter]
            if len(self.idxs) < self.min_num_trials:
                raise ValueError(
                    f"# trials after balancing for trial count (len(self.idxs)) is less than {self.min_num_trials}"
                )

        self.trial_data = self.trial_data.iloc[self.idxs]
        self.psths = {reg: self.psths[reg][:, self.idxs, :] for reg in self.regions}

    def build_dm(self):
        super().build_dm()

        self.encoder_ref.build_dm()
        self.tents = self.encoder_ref.tents[self.idxs]

        if self.norm:
            # want to have normalized with respect to all the trials
            self.robs = self.encoder_ref.robs[self.idxs]

    def fit_baseline(self):
        self.encoder_ref.fit_baseline()
        self.baseline_model = self.encoder_ref.baseline_model

    def get_r2(self, n_folds=20, p_train=0.8, pool=True):
        if not hasattr(self, "robs"):
            self.build_dm()
        if not hasattr(self, "baseline_model"):
            self.fit_baseline()

        yhats = {
            k: np.zeros((self.num_trials, self.num_units))
            for k in ["baseline", "encoder"]
        }

        for i, (train_idxs, test_idxs) in enumerate(
            KFold(n_splits=n_folds, shuffle=True, random_state=0).split(self.robs)
        ):
            yhats["encoder"][test_idxs], _ = self._get_predictions(
                is_encoder=True,
                io=(self.tents, self.tvs, self.robs),
                idxs=(train_idxs, test_idxs),
                baseline_model=self.baseline_model,
            )

        yhats["baseline"] = self.baseline_model.predict(self.tents)

        self.scores = {
            k: r2_score(self.robs, yhat, multioutput="raw_values", force_finite=False)
            for k, yhat in yhats.items()
        }

    def verify(self, subtract_baseline=True):
        super().verify(r2_comp=False)


class ShuffledEncoder:
    def __init__(
        self,
        subj_id,
        sess_id,
        **kwargs,
    ):
        self.subj_id = subj_id
        self.sess_id = sess_id

        self.kwargs = kwargs

        self.encoder_full = Encoder(subj_id, sess_id, **kwargs)
        self.encoder_full.get_r2()

        self.task_vars = self.encoder_full.tv_keys

    def get_cvr2(self, pivot, n_iters=3):
        # TODO: rerun the shuffle several times

        if pivot not in self.task_vars:
            raise ValueError(f"pivot {pivot} is not in {self.task_vars}")

        if not hasattr(self, "cvr2"):
            self.cvr2_unit = {}
            self.cvr2 = {}

        if not hasattr(self, "encoders_cvr2"):
            self.encoders_cvr2 = {}

        self.cvr2_unit[pivot] = np.zeros((n_iters, self.encoder_full.num_units))
        self.cvr2[pivot] = np.zeros((n_iters,))

        for i in range(n_iters):
            encoder_shuffle = Encoder(self.subj_id, self.sess_id, **self.kwargs)
            encoder_shuffle.get_data()

            # shuffle all taskvars besides the pivot
            for tv in encoder_shuffle.tv_keys:
                if not tv == pivot:
                    encoder_shuffle.trial_data[tv] = (
                        encoder_shuffle.trial_data[tv]
                        .sample(frac=1, random_state=i)
                        .to_numpy()
                    )

            encoder_shuffle.get_r2()

            # ...
            self.cvr2_unit[pivot][i] = encoder_shuffle.scores["encoder"]
            self.cvr2[pivot][i] = self.cvr2_unit[pivot][i].mean()

        self.encoders_cvr2[pivot] = encoder_shuffle

    def get_dr2(self, pivot, n_iters=3):
        if pivot not in self.task_vars:
            raise ValueError(f"pivot {pivot} is not in {self.task_vars}")

        if not hasattr(self, "dr2"):
            self.dr2_unit = {}
            self.dr2 = {}

        if not hasattr(self, "encoders_dr2"):
            self.encoders_dr2 = {}

        self.dr2_unit[pivot] = np.zeros((n_iters, self.encoder_full.num_units))
        self.dr2[pivot] = np.zeros((n_iters,))

        for i in range(n_iters):
            encoder_shuffle = Encoder(self.subj_id, self.sess_id, **self.kwargs)
            encoder_shuffle.get_data()

            # shuffle the pivot
            encoder_shuffle.trial_data[pivot] = (
                encoder_shuffle.trial_data[pivot]
                .sample(frac=1, random_state=i)
                .to_numpy()
            )

            encoder_shuffle.get_r2()

            # ...
            self.dr2_unit[pivot][i] = (
                self.encoder_full.scores["encoder"] - encoder_shuffle.scores["encoder"]
            )
            self.dr2[pivot][i] = self.dr2_unit[pivot][i].mean()

        self.encoders_dr2[pivot] = encoder_shuffle

    def get_cvr2_all(self):
        pivots = (
            self.task_vars
            if not hasattr(self, "cvr2")
            else np.setdiff1d(self.task_vars, list(self.cvr2.keys()))
        )
        for pivot in pivots:
            self.get_cvr2(pivot)

    def get_dr2_all(self):
        pivots = (
            self.task_vars
            if not hasattr(self, "dr2")
            else np.setdiff1d(self.task_vars, list(self.dr2.keys()))
        )
        for pivot in pivots:
            self.get_dr2(pivot)

    def plot_cvr2(self):
        if (
            not hasattr(self, "cvr2")
            or len(np.setdiff1d(self.task_vars, list(self.cvr2.keys()))) > 0
        ):
            self.get_cvr2_all()

        cvr2_mean = [self.cvr2[pivot].mean() for pivot in self.task_vars]
        cvr2_std = [self.cvr2[pivot].std() for pivot in self.task_vars]

        fig, ax = plt.subplots(tight_layout=True)
        ax.bar(self.task_vars, cvr2_mean, width=0.5)
        ax.errorbar(
            x=self.task_vars,
            y=cvr2_mean,
            yerr=cvr2_std,
            color="k",
            fmt=".",
            capsize=2,
        )

        if not hasattr(self.encoder_full, "scores"):
            self.encoder_full.get_r2()
        ax.axhline(
            y=self.encoder_full.scores["encoder"].mean(),
            color="#666666",
            linewidth=0.5,
            linestyle="--",
            label=r"full $r^2$",
        )
        ax.legend(loc="upper right")

        ax.set_ylabel(r"cv $r^2$")
        ax.tick_params(axis="x", labelrotation=45)

    def plot_dr2(self):
        if (
            not hasattr(self, "dr2")
            or len(np.setdiff1d(self.task_vars, list(self.dr2.keys()))) > 0
        ):
            self.get_dr2_all()

        dr2_mean = [self.dr2[pivot].mean() for pivot in self.task_vars]
        dr2_std = [self.dr2[pivot].std() for pivot in self.task_vars]

        fig, ax = plt.subplots(tight_layout=True)
        ax.bar(self.task_vars, dr2_mean, width=0.5)
        ax.errorbar(
            x=self.task_vars,
            y=dr2_mean,
            yerr=dr2_std,
            color="k",
            fmt=".",
            capsize=2,
        )

        if not hasattr(self.encoder_full, "scores"):
            self.encoder_full.get_r2()
        ax.axhline(
            y=self.encoder_full.scores["encoder"].mean(),
            color="#666666",
            linewidth=0.5,
            linestyle="--",
            label=r"full $r^2$",
        )
        ax.legend(loc="upper right")

        ax.set_ylabel(r"$\Delta r^2$")
        ax.tick_params(axis="x", labelrotation=45)

    def plot_bound_r2(self, ax=None):
        if (
            not hasattr(self, "cvr2")
            or len(np.setdiff1d(self.task_vars, list(self.cvr2.keys()))) > 0
        ):
            self.get_cvr2_all()
        if (
            not hasattr(self, "dr2")
            or len(np.setdiff1d(self.task_vars, list(self.dr2.keys()))) > 0
        ):
            self.get_dr2_all()

        if ax is None:
            _, ax = plt.subplots(tight_layout=True)

        n = len(self.task_vars)
        x = np.arange(n)

        cvr2_mean = np.array([self.cvr2[pivot].mean() for pivot in self.task_vars])
        cvr2_std = np.array([self.cvr2[pivot].std() for pivot in self.task_vars])
        dr2_mean = np.array([self.dr2[pivot].mean() for pivot in self.task_vars])
        dr2_std = np.array([self.dr2[pivot].std() for pivot in self.task_vars])

        ax.errorbar(
            x=x,
            y=cvr2_mean,
            yerr=cvr2_std,
            fmt=".",
            color="#999999",
            markersize=1,
            capsize=2,
            label="maximal",
        )
        ax.errorbar(
            x=x,
            y=dr2_mean,
            yerr=dr2_std,
            fmt=".",
            color="#333333",
            markersize=1,
            capsize=2,
            label="unique",
        )

        if not hasattr(self.encoder_full, "scores"):
            self.encoder_full.get_r2()
        ax.axhline(
            y=self.encoder_full.scores["encoder"].mean(),
            color="#666666",
            linewidth=0.5,
            linestyle="--",
            label=r"full $r^2$",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(self.task_vars)
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_ylabel(r"$r^2$")

        ax.legend(loc="upper right")

        return ax


def make_tre(enc_class: Type[Encoder] = Encoder, **kwargs):
    class TimeResolvedEncoder(enc_class):
        def __init__(
            self,
            subj_id,
            sess_id,
            stepsize_s=0.1,
            **kwargs,
        ):
            if not (isinstance(enc_class, type) and issubclass(enc_class, Encoder)):
                raise TypeError(
                    f"enc_class must be a subclass of Encoder, got {enc_class}"
                )

            self.subj_id = subj_id
            self.sess_id = sess_id

            self.stepsize_s = stepsize_s

            if self.stepsize_s < 0.001:
                raise ValueError("min stepsize is 1 ms")

            if enc_class is Encoder:
                super().__init__(
                    subj_id,
                    sess_id,
                    separate_drift=True,
                    **kwargs,
                )
            elif enc_class is StrategyEncoder:
                # separate_drift is True already in StrategyEncoder
                super().__init__(
                    subj_id,
                    sess_id,
                    **kwargs,
                )
            else:
                raise ValueError(f"what kind of class did you pass??? ({enc_class})")

            assert self.separate_drift, "ur not separating drift"

            self.binwidth_ms = stepsize_s * 1000

            self.num_bins = int((self.tpre + self.tpost) / self.stepsize_s)

            if not self.num_bins * self.stepsize_s == self.tpre + self.tpost:
                raise ValueError(
                    f"stepsize {stepsize_s} s must evenly divide the spanned epoch [-{self.tpre}, {self.tpost}]"
                )

            self.tbins, step_ = np.linspace(
                -self.tpre * 1000, self.tpost * 1000, self.num_bins + 1, retstep=True
            )
            self.tbins /= 1000
            self.tbins = np.round(
                self.tbins, 3
            )  # only supports precision to the nearest millisecond
            self.tbin_centers = (self.tbins - (self.stepsize_s / 2))[1:]

            assert step_ / 1000 == self.stepsize_s, "stepsize issue"

            if enc_class is Encoder:
                self.t_encoders = {
                    f"[{start:.3f},{stop:.3f}]": enc_class(  # modify this to accommodate strategy encoder too.
                        subj_id, sess_id, separate_drift=True, **kwargs
                    )
                    for (start, stop) in zip(self.tbins, self.tbins[1:])
                }
            elif enc_class is StrategyEncoder:
                self.t_encoders = {
                    f"[{start:.3f},{stop:.3f}]": enc_class(  # modify this to accommodate strategy encoder too.
                        subj_id, sess_id, **kwargs
                    )
                    for (start, stop) in zip(self.tbins, self.tbins[1:])
                }

            assert all([e.separate_drift for e in self.t_encoders.values()]), (
                "ur not separating drift in t_encoders!!"
            )

            assert len(self.t_encoders) == self.num_bins, (
                "not one-to-one encoder per bin"
            )

        def get_data(self):
            print("ignore the next warning hehe")
            super().get_data()

        def build_dm(self):
            print("ignore the next warning hehe")
            super().build_dm()

            # self.tents will not be used to estimate robs on a subtrial basis
            self.tvs = np.array([self.tvs for i in range(self.num_bins)])
            self.dm = np.array([self.dm for i in range(self.num_bins)])

            assert self.tvs.shape == (self.num_bins, self.num_trials, self.num_tv), (
                "tv shape is wrong"
            )
            assert self.dm.shape == (
                self.num_bins,
                self.num_trials,
                self.num_tents + self.num_tv,
            ), "dm shape is wrong"

            def _edge_inclusive(t):
                return t + 0.001 if self.edge_inclusive else t

            self.robs, _, self.t_tbin_edges = get_psths_ref(
                self.spike_times,
                self.encoder_ref.trial_data
                if enc_class is StrategyEncoder
                else self.trial_data,
                self.session_data,
                self.regions,
                tpre=_edge_inclusive(self.tpre),
                tpost=_edge_inclusive(self.tpost),
                binwidth_ms=self.stepsize_s * 1000,
                alignment=self.alignment,
                trial_start_pre=0,
                tpre_ref=_edge_inclusive(self.tpre_ref),
                tpost_ref=_edge_inclusive(self.tpost_ref),
                alignment_ref=self.alignment_ref,
                mode="new",
                thresh=self.thresh,
            )

            if self.edge_inclusive:
                assert np.allclose(self.t_tbin_edges, self.tbins), "alignment issue"
            else:
                assert np.allclose(self.t_tbin_edges, self.tbins[1:-1]), (
                    "alignment issue"
                )

            self.robs = np.array(
                [unit for reg in self.regions for unit in self.robs[reg]]
            ).T

            if self.norm:
                self.robs = zscore(self.robs, axis=(0, 1))
            self.robs_tavg = self.robs.mean(axis=0)  # don't want to subsample robs_tavg
            if enc_class is StrategyEncoder:
                self.robs = self.robs[:, self.idxs, :]

        def fit_baseline(self):
            if not hasattr(self, "robs"):
                self.build_dm()
            print("done-")

            self.baseline_models = {}
            for i, (k, encoder_) in enumerate(self.t_encoders.items()):
                assert encoder_.separate_drift, "t_encoder is not separating drift!"
                encoder_.tents = self.tents
                encoder_.robs = self.robs_tavg
                if enc_class is StrategyEncoder:
                    encoder_.encoder_ref.build_dm()
                    encoder_.encoder_ref.robs = self.robs_tavg
                    assert np.all(encoder_.encoder_ref.robs - self.robs_tavg == 0)

                encoder_.fit_baseline()
                self.baseline_models[k] = encoder_.baseline_model
            print("girl bye")

        def baseline_predict(self, pseudo=False):
            if not hasattr(self, "robs_predict"):
                self.robs_predict = {}

            if not hasattr(self, "baseline_models"):
                self.fit_baseline()

            if pseudo:
                self.robs_predict["ps_baseline"] = np.zeros_like(
                    self.robs, dtype="float32"
                )
            else:
                self.robs_predict["baseline"] = np.zeros_like(
                    self.robs, dtype="float32"
                )

            for i, encoder in enumerate(self.t_encoders.values()):
                encoder.baseline_predict(pseudo=pseudo)
                if pseudo:
                    self.robs_predict["ps_baseline"][i] = encoder.robs_predict[
                        "ps_baseline"
                    ]
                else:
                    self.robs_predict["baseline"][i] = encoder.robs_predict["baseline"]

        def fit_encoder(self):
            if not hasattr(self, "robs"):
                self.build_dm()

            self.encoders = {}
            self.encoder_weights = np.zeros(
                (self.num_bins, self.num_units, self.num_tents + self.num_tv)
            )
            for i, (k, encoder_) in enumerate(self.t_encoders.items()):
                assert encoder_.separate_drift, "not separating drift!"
                # sanity checks needed here
                encoder_.tents = self.tents
                encoder_.tvs = self.tvs[i]
                encoder_.robs = self.robs[i]

                encoder_.fit_encoder()
                self.encoders[k] = encoder_.encoder
                self.encoder_weights[i] = encoder_.encoder_weights

        def encoder_predict(self):
            if not hasattr(self, "robs_predict"):
                self.robs_predict = {}

            if not hasattr(self, "encoders"):
                self.fit_encoder()

            self.robs_predict["encoder"] = np.zeros_like(self.robs, dtype="float32")

            for i, encoder in enumerate(self.t_encoders.values()):
                encoder.encoder_predict()
                self.robs_predict["encoder"][i] = encoder.robs_predict["encoder"]

        def get_r2(self, n_folds=20, p_train=0.8):
            if not hasattr(self, "robs"):
                self.build_dm()

            if (
                not hasattr(self, "robs_predict")
                or "baseline" not in self.robs_predict.keys()
            ):
                self.baseline_predict()

            self.yhats = {
                k: np.zeros((self.num_trials * self.num_bins, self.num_units))
                for k in ["encoder"]
            }

            def _reshape(arr_3d):
                # arr_3d.shape = (num_bins, num_trials, _)
                assert arr_3d.ndim == 3, "not 3d"
                return arr_3d.reshape(self.num_bins * self.num_trials, arr_3d.shape[-1])

            for i, (train_idxs, test_idxs) in enumerate(
                KFold(n_splits=n_folds, shuffle=True, random_state=0).split(
                    _reshape(self.robs)
                )
            ):
                # train, test, save test predictions
                assert self.separate_drift, "no separate drift, must separate drift!!!"

                self.yhats["encoder"][test_idxs], _ = self._get_predictions(
                    is_encoder=True,
                    io=(
                        _reshape(self.tvs),
                        _reshape(self.robs),
                    ),
                    idxs=(train_idxs, test_idxs),
                    baseline_predict=_reshape(self.robs_predict["baseline"]),
                )

            self.scores = {
                k: r2_score(
                    _reshape(self.robs),
                    yhat,
                    multioutput="raw_values",
                    force_finite=False,
                )
                for k, yhat in self.yhats.items()
            }

        def view_fits(self, reg="DLS", model="encoder", mode="time"):
            if not hasattr(self, "regions"):
                self.get_data()
            if reg not in self.regions:
                raise ValueError(f"{reg} must be in {self.regions}")

            if (
                not hasattr(self, "robs_predict")
                or model not in self.robs_predict.keys()
            ):
                if model == "encoder":
                    self.encoder_predict()
                elif model == "baseline":
                    self.baseline_predict()
                else:
                    raise ValueError(
                        f"valid arguments for model are 'encoder' and 'baseline,' not {model}"
                    )

            if not hasattr(self, "scores"):
                self.get_r2()

            def _transform(robs_3d, reg):
                return (
                    robs_3d[:, :, self.reg_idxs[reg]]
                    .T.reshape(self.psths[reg].shape[0], -1)
                    .T
                )

            if mode == "time":
                r = FitRendererTime(
                    x=self.tbin_centers,
                    y=self.robs[:, :, self.reg_idxs[reg]],
                    yhat=self.robs_predict[model][:, :, self.reg_idxs[reg]],
                    rsquared=self.scores[model][self.reg_idxs[reg]],
                )
            elif mode == "full":
                r = FitRenderer(
                    y=_transform(self.robs, reg),
                    yhat=_transform(self.robs_predict[model], reg),
                    rsquared=self.scores[model][self.reg_idxs[reg]],
                    mode="lite",
                )
            else:
                raise ValueError("mode can only be 'time' or 'full'")

            return NeuronViewer(
                num_units=self.psths[reg].shape[0], render_func=r, fig_dir=FIGURES_DIR
            )

        def view_weights(self, reg="DLS", peth_mode="response", mode="trace"):
            if not hasattr(self, "encoder_weights"):
                self.fit_encoder()

            if mode == "trace":
                r = PETHWeightRendererTime(
                    weights=self.encoder_weights[:, self.reg_idxs[reg], :],
                    tv="response",
                    weight_idxs=self.dm_idxs,
                    tv_vals=tv_vals,
                    mode="trace",
                    peths=get_psths_cond(
                        self.psths[reg], self.trial_data, mode=peth_mode
                    ),
                    binwidth_s=0.1,
                    tbin_centers=self.tbin_centers,
                )
            elif mode == "matrix":
                r = PETHWeightRendererTime(
                    weights=self.encoder_weights[:, self.reg_idxs[reg], :],
                    weight_names=self.dm_names,
                    mode="matrix",
                    peths=get_psths_cond(
                        self.psths[reg], self.trial_data, mode=peth_mode
                    ),
                    binwidth_s=0.1,
                    tbin_centers=self.tbin_centers,
                )
            else:
                raise ValueError("valid arguments are 'trace' and 'matrix'")

            return NeuronViewer(
                num_units=self.psths[reg].shape[0], render_func=r, fig_dir=FIGURES_DIR
            )

        def verify(self):
            super().verify(r2_comp=False)

        def get_sctavg_weights(self, cond="response", subtract_baseline=True):
            if cond not in ["response", "rewarded"]:
                raise NotImplementedError(f"cond={cond} is not currently supported.")

            if not hasattr(self, "sc_tavg"):
                self.sc_tavg = {}

            if cond in self.sc_tavg.keys():
                return

            self.sc_tavg[cond] = {val: [] for val in tv_vals[cond]}

            # if not hasattr(next(iter(self.t_encoders.values())), "baseline_model"):
            print("Pump")
            self.baseline_predict()
            for i, e in enumerate(self.t_encoders.values()):
                e.dm_idxs = self.dm_idxs
                e.trial_data = self.trial_data
                e.robs = self.robs[i]
                e.get_sctavg_weights(cond=cond)

                for k in e.sc_tavg[cond].keys():
                    self.sc_tavg[cond][k].extend(e.sc_tavg[cond][k])

            self.sc_tavg[cond] = {
                k: np.array(arr) for k, arr in self.sc_tavg[cond].items()
            }

        def plot_sctavg_weights(
            self, axes=None, cond="response", subtract_baseline=True
        ):
            if not hasattr(self, "sc_tavg") or cond not in self.sc_tavg.keys():
                self.get_sctavg_weights(cond=cond, subtract_baseline=subtract_baseline)
            if axes is None:
                fig, axes = plt.subplots(
                    ncols=2, nrows=1, figsize=(4, 2), tight_layout=True
                )

            keys = tv_vals[cond]
            idxs = [self.dm_idxs[regr] for regr in [f"{cond}_{key}" for key in keys]]

            for i in range(2):
                axes[i].scatter(
                    self.sc_tavg[cond][keys[i]],
                    self.encoder_weights[:, :, idxs[i]].ravel(),
                    s=0.5,
                    alpha=0.5,
                    vmin=1e-5,
                    vmax=1e5,
                    c=np.concatenate(
                        [e.encoder.alpha_ for e in self.t_encoders.values()]
                    ),
                    cmap="viridis",
                    norm="log",
                )

                mn = min(
                    min(self.sc_tavg[cond][keys[i]]),
                    min(self.encoder_weights[:, :, idxs[i]].ravel()),
                )
                mx = max(
                    max(self.sc_tavg[cond][keys[i]]),
                    max(self.encoder_weights[:, :, idxs[i]].ravel()),
                )
                axes[i].plot(
                    [1.05 * mn, 1.05 * mx],
                    [1.05 * mn, 1.05 * mx],
                    color="#666666",
                    linewidth=0.7,
                    linestyle="--",
                    zorder=0,
                )
                axes[i].axhline(y=0, color="k", linewidth=0.5)
                axes[i].axvline(x=0, color="k", linewidth=0.5)

                axes[i].set_xlabel(f"resid spk count, {keys[i]}")
                axes[i].set_ylabel(f"beta weight, {keys[i]}")

        def plot_r2_comp(self):
            raise NotImplementedError("no baseline r2 to compare to")

    TimeResolvedEncoder.__name__ = f"TimeResolved{enc_class.__name__}"
    TimeResolvedEncoder.__qualname__ = TimeResolvedEncoder.__name__
    return TimeResolvedEncoder
