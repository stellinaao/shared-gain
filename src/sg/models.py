import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from core.data import (
    load_sess,
    get_strategy_filter_idxs,
    get_encoder_io,
    get_tavg_sc_cond,
    get_choice_ts,
    get_psths_cond,
)


from squiggs.neuron_viewer import NeuronViewer
from squiggs.renderers import FitRenderer, PETHWeightRenderer
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

        self.norm = kwargs.pop("norm", False)
        self.separate_drift = kwargs.pop("separate_drift", False)

        self.num_tents = kwargs.pop("num_tents", 5)

        self.tpre = kwargs.pop("tpre", 0.5)
        self.tpost = kwargs.pop("tpost", 1)
        self.alignment = kwargs.pop("alignment", "choice")

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
        (
            self.spike_times,
            self.trial_data,
            self.psths,
            self.session_data,
            self.regions,
        ) = load_sess(
            subj_id=self.subj_id,
            sess_id=self.sess_id,
            tpre=self.tpre,
            tpost=self.tpost,
            alignment=self.alignment,
            tpre_ref=self.tpre_ref,
            tpost_ref=self.tpost_ref,
            alignment_ref=self.alignment_ref,
            binwidth_ms=self.binwidth_ms,
            add_svd=self.add_svd,
            thresh=self.thresh,
        )

    def build_dm(self):
        if not (hasattr(self, "psths")):
            self.get_data()
        (
            self.tents,
            self.tvs,
            self.dm,
            self.robs,
            self.dm_names,
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
            binwidth_ms=25,
        )

        self.num_trials, self.num_tv = self.tvs.shape
        self.num_units = self.robs.shape[1]

    def fit_baseline(self, idxs=None):
        if not (hasattr(self, "dm") and hasattr(self, "robs")):
            self.build_dm()

        self.baseline_model = RidgeCV(
            alphas=np.logspace(-5, 5, 11, base=10),
            alpha_per_target=True,
        ).fit(self.tents, self.robs)

    def baseline_predict(self, pseudo=False):
        if not hasattr(self, "robs_predict"):
            self.robs_predict = {}

        if not pseudo:
            if not hasattr(self, "baseline_model"):
                self.fit_baseline()

            self.robs_predict["baseline"] = self.baseline_model.predict(self.tents)
        else:
            if self.separate_drift:
                if (
                    not hasattr(self, "robs_predict")
                    or "baseline" not in self.robs_predict.keys()
                ):
                    self.baseline_predict(pseudo=False)

                self.robs_predict["ps_baseline"] = self.robs_predict["baseline"]
            else:
                if not hasattr(self, "encoder"):
                    self.fit_encoder()

                if not hasattr(self, "dm_tv_ko"):
                    self.dm_tv_ko = deepcopy(self.dm)
                    self.dm_tv_ko[:, self.num_tents :] = 0

                self.robs_predict["ps_baseline"] = self.encoder.predict(self.dm_tv_ko)

    def fit_encoder(self, idxs=None):
        if not (hasattr(self, "dm") and hasattr(self, "robs")):
            self.build_dm()

        if self.separate_drift:
            if not hasattr(self, "baseline_model"):
                self.fit_baseline()
            self.baseline_predict()

            self.encoder = RidgeCV(
                alphas=np.logspace(-5, 5, 11, base=10),
                alpha_per_target=True,
            ).fit(self.tvs, self.robs - self.robs_predict["baseline"])

            self.encoder_weights = np.hstack(
                (self.baseline_model.coef_, self.encoder.coef_)
            )
        else:
            self.encoder = RidgeCV(
                alphas=np.logspace(-5, 5, 11, base=10),
                alpha_per_target=True,
            ).fit(self.dm, self.robs)

            self.encoder_weights = self.encoder.coef_

    def encoder_predict(self):
        if not hasattr(self, "robs_predict"):
            self.robs_predict = {}
        if not hasattr(self, "encoder"):
            self.fit_encoder()

        if self.separate_drift:
            self.robs_predict["encoder"] = self.baseline_model.predict(
                self.tents
            ) + self.encoder.predict(self.tvs)
        else:
            self.robs_predict["encoder"] = self.encoder.predict(self.dm)

    def get_scores(self, is_encoder, **kwargs):
        if self.separate_drift and is_encoder:
            return self._get_scores_sd(**kwargs)
        else:
            return self._get_scores_nosd(**kwargs)

    def _get_scores_nosd(self, io, idxs, model=None):
        train_idxs, test_idxs = idxs
        dm, robs = io

        if model is None:
            model = RidgeCV(
                alphas=np.logspace(-5, 5, 11, base=10), alpha_per_target=True
            ).fit(dm[train_idxs], robs[train_idxs])

        scores = r2_score(
            robs[test_idxs],
            model.predict(dm[test_idxs]),
            multioutput="raw_values",
            force_finite=False,
        )

        return scores, model

    def _get_scores_sd(self, io, idxs, baseline_model):
        train_idxs, test_idxs = idxs
        tents, tvs, robs = io

        encoder = RidgeCV(
            alphas=np.logspace(-5, 5, 11, base=10),
            alpha_per_target=True,
        ).fit(
            tvs[train_idxs],
            robs[train_idxs] - baseline_model.predict(tents[train_idxs]),
        )

        scores = r2_score(
            robs[test_idxs],
            baseline_model.predict(tents[test_idxs]) + encoder.predict(tvs[test_idxs]),
            multioutput="raw_values",
            force_finite=False,
        )

        return scores, encoder

    def get_r2(self, n_folds=20, p_train=0.8):
        if not hasattr(self, "robs"):
            self.build_dm()

        self.scores_cv = {
            "baseline": np.zeros((n_folds, self.num_units)),
            "ps_baseline": np.zeros((n_folds, self.num_units)),
            "encoder": np.zeros((n_folds, self.num_units)),
        }

        for i in range(n_folds):
            np.random.seed(seed=i)

            train_idxs = np.sort(
                np.random.choice(
                    self.num_trials, int(self.num_trials * p_train), replace=False
                )
            )
            test_idxs = np.setdiff1d(np.arange(self.num_trials), train_idxs)

            self.scores_cv["baseline"][i], baseline_model = self.get_scores(
                is_encoder=False,
                io=(self.tents, self.robs),
                idxs=(train_idxs, test_idxs),
            )

            if self.separate_drift:
                """
                encoder = RidgeCV(
                    alphas=np.logspace(-5, 5, 11, base=10),
                    alpha_per_target=True,
                ).fit(
                    self.tvs[train_idxs],
                    self.robs[train_idxs]
                    - baseline_model.predict(self.tents[train_idxs]),
                )

                self.scores_cv["encoder"][i] = r2_score(
                    self.robs[test_idxs],
                    baseline_model.predict(self.tents[test_idxs])
                    + encoder.predict(self.tvs[test_idxs]),
                    multioutput="raw_values",
                    force_finite=False,
                )
                """
                self.scores_cv["encoder"][i], _ = self.get_scores(
                    is_encoder=True,
                    io=(self.tents, self.tvs, self.robs),
                    idxs=(train_idxs, test_idxs),
                    baseline_model=baseline_model,
                )

                self.scores_cv["ps_baseline"][i] = self.scores_cv["baseline"][i]

            else:
                self.scores_cv["encoder"][i], encoder = self.get_scores(
                    is_encoder=True,
                    io=(self.dm, self.robs),
                    idxs=(train_idxs, test_idxs),
                )

                if not hasattr(self, "dm_tv_ko"):
                    self.dm_tv_ko = deepcopy(self.dm)
                    self.dm_tv_ko[:, self.num_tents :] = 0

                self.scores_cv["ps_baseline"][i], _ = self.get_scores(
                    is_encoder=False,
                    io=(self.dm_tv_ko, self.robs),
                    idxs=(train_idxs, test_idxs),
                    model=encoder,
                )

        self.seed()

        self.scores = {
            "baseline": np.median(self.scores_cv["baseline"], axis=0),
            "ps_baseline": np.median(self.scores_cv["ps_baseline"], axis=0),
            "encoder": np.median(self.scores_cv["encoder"], axis=0),
        }

    def verify(self, cond="response", subtract_baseline=True):
        ncols = 6 if self.tv_keys is not None else 2
        _, axes = plt.subplots(
            ncols=ncols, nrows=1, figsize=(ncols * 1.5, 1.5), tight_layout=True
        )

        # baseline vs encoder r2
        self.plot_r2_comp(axes[0])

        # p(resp)
        self.plot_p_resp(axes[1])

        # sctavg vs beta weight
        if self.tv_keys is not None:
            self.plot_sctavg_weights(
                axes[2:4], cond="response", subtract_baseline=subtract_baseline
            )
            self.plot_sctavg_weights(
                axes[4:6], cond="rewarded", subtract_baseline=subtract_baseline
            )

    def plot_r2_comp(self, ax=None):
        if not hasattr(self, "scores"):
            try:
                self.get_r2()
            except NotImplementedError:
                return

        if ax is None:
            _, ax = plt.figure(tight_layout=True)

        ax.scatter(self.scores["baseline"], self.scores["encoder"], s=0.5, alpha=0.5)
        ax.plot([-0.5, 1], [-0.5, 1], color="#666666", linestyle="--", linewidth=0.5)
        ax.plot([-0.5, 1], [-0.5, 1], color="#666666", linestyle="--", linewidth=0.5)
        ax.axhline(y=0, color="k", linewidth=0.5)
        ax.axvline(x=0, color="k", linewidth=0.5)

        ax.set_xlabel(r"$r^2$, drift")
        ax.set_ylabel(r"$r^2$, encoder")

    def plot_sctavg_weights(self, axes, cond="response", subtract_baseline=True):
        if cond not in ["response", "rewarded"]:
            raise NotImplementedError(f"cond={cond} is not currently supported.")

        if axes is None:
            fig, axes = plt.subplots(
                ncols=2, nrows=1, figsize=(4, 2), tight_layout=True
            )

        if not hasattr(self, "encoder"):
            self.fit_encoder()

        if cond == "response":
            keys = ["left", "right"]
        elif cond == "rewarded":
            keys = ["corr", "incorr"]
        idxs = np.array(
            [np.where(self.dm_names == f"{cond}_{key}")[0][0] for key in keys]
        )

        if subtract_baseline:
            if self.separate_drift:
                # get baseline explained variance
                if (
                    not hasattr(self, "robs_predict")
                    or "ps_baseline" not in self.robs_predict.keys()
                ):
                    self.baseline_predict(pseudo=True)
                robs_baseline = self.robs_predict["ps_baseline"]

                # get tv explained variance (besides pivot)
                pivot_idxs = idxs - self.num_tents

                self.tv_pivot_ko = deepcopy(self.tvs)
                self.tv_pivot_ko[:, pivot_idxs] = 0

                robs_tv = self.encoder.predict(self.tv_pivot_ko)
                robs_to_subtract = robs_baseline + robs_tv

            else:
                pivot_idxs = idxs

                self.dm_pivot_ko = deepcopy(self.dm)
                self.dm_pivot_ko[:, pivot_idxs] = 0

                robs_to_subtract = self.encoder.predict(self.dm_pivot_ko)

        else:
            robs_to_subtract = None

        sc_tavg = get_tavg_sc_cond(
            self.robs,
            self.trial_data,
            cond=cond,
            robs_to_subtract=robs_to_subtract,
            subtract_robs=subtract_baseline,
        )

        for i in range(2):
            axes[i].scatter(
                sc_tavg[keys[i]],
                self.encoder_weights[:, idxs[i]],
                s=0.5,
                alpha=0.5,
                vmin=1e-5,
                vmax=1e5,
                c=self.encoder.alpha_,
                cmap="viridis",
                norm="log",
            )

            mn = min(min(sc_tavg[keys[i]]), min(self.encoder_weights[:, idxs[i]]))
            mx = max(max(sc_tavg[keys[i]]), max(self.encoder_weights[:, idxs[i]]))
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
            weights=self.encoder.coef_[self.reg_idxs[reg], :],
            weight_names=self.dm_names,
            robs=self.robs[:, self.reg_idxs[reg]],
            sc_tavg=sc_tavg,
            event_times=get_choice_ts(self.trial_data, mode=mode),
            spike_times=self.spike_times[reg],
            peths=get_psths_cond(self.psths[reg], self.trial_data, mode=mode),
            pres=self.tpre,
            posts=self.tpost,
            binwidth_s=self.binwidth_ms / 1000,
        )

        return NeuronViewer(
            num_units=self.psths[reg].shape[0], render_func=r, fig_dir=FIGURES_DIR
        )


class StrategyEncoder(Encoder):
    def __init__(
        self,
        subj_id,
        sess_id,
        strategy_filter="mb",
        idxs=None,
        **kwargs,
    ):
        self.subj_id = subj_id
        self.sess_id = sess_id

        self.strategy_filter = strategy_filter
        self.idxs = idxs

        if not (self.strategy_filter == "mb" or self.strategy_filter == "mf"):
            raise ValueError("strategy_filter must be mb or mf")

        self.encoder_ref = Encoder(subj_id=subj_id, sess_id=sess_id, **kwargs)

        super().__init__(
            subj_id,
            sess_id,
            num_tents=self.encoder_ref.num_tents,
            separate_drift=True,
            **kwargs,
        )

    def get_data(self):
        super().get_data()

        if self.idxs is None:
            self.idxs_all = get_strategy_filter_idxs(
                self.trial_data, self.strategy_filter, balance_strategy=True
            )
            self.idxs = self.idxs_all[self.strategy_filter]

        self.trial_data = self.trial_data.iloc[self.idxs]
        self.psths = {reg: self.psths[reg][:, self.idxs, :] for reg in self.regions}

    def build_dm(self):
        super().build_dm()

        self.encoder_ref.build_dm()
        self.tents = self.encoder_ref.tents[self.idxs]

    def fit_baseline(self):
        self.encoder_ref.fit_baseline()
        self.baseline_model = self.encoder_ref.baseline_model

    # def baseline_predict(self):
    #     super().baseline_predict()

    # def fit_encoder(self):
    #     super().fit_encoder()

    # def encoder_predict(self):
    #     super().encoder_predict()

    def get_r2(self, n_folds=20, p_train=0.8):
        if not hasattr(self, "robs"):
            self.build_dm()

        self.scores_cv = {
            "baseline": np.zeros((n_folds, self.num_units)),
            "encoder": np.zeros((n_folds, self.num_units)),
        }

        self.encoder_ref.get_r2()
        self.scores_cv["baseline"] = self.encoder_ref.scores_cv["baseline"]

        # encoder
        for i in range(n_folds):
            np.random.seed(seed=i)
            train_idxs = np.sort(
                np.random.choice(
                    self.num_trials,
                    int(self.num_trials * p_train),
                    replace=False,
                )
            )
            test_idxs = np.setdiff1d(np.arange(self.num_trials), train_idxs)

            if (
                not hasattr(self, "robs_predict")
                or "baseline" not in self.robs_predict.keys()
            ):
                self.baseline_predict()

            encoder = RidgeCV(
                alphas=np.logspace(-5, 5, 11, base=10),
                alpha_per_target=True,
            ).fit(
                self.tvs[train_idxs],
                self.robs[train_idxs] - self.robs_predict["baseline"][train_idxs],
            )

            self.scores_cv["encoder"][i] = r2_score(
                self.robs[test_idxs],
                self.robs_predict["baseline"][test_idxs]
                + encoder.predict(self.tvs[test_idxs]),
                multioutput="raw_values",
                force_finite=False,
            )

        self.seed()

        self.scores = {
            "baseline": np.median(self.scores_cv["baseline"], axis=0),
            "encoder": np.median(self.scores_cv["encoder"], axis=0),
        }


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
        ax.set_ylabel(r"$\Delta r^2$")
        ax.tick_params(axis="x", labelrotation=45)
