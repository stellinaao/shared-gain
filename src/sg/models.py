import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from sg.fitlvm_utils import get_dataset_dm
from core.data import load_sess, get_tavg_sc_cond
from squiggs.neuron_viewer import NeuronViewer
from squiggs.renderers import FitRenderer
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

        self.task_vars = kwargs.pop(
            "task_vars",
            [
                "response",
                "rewarded",
                "block_side",
                "response_prev",
                "rewarded_prev",
            ],
        )
        self.num_tents = kwargs.pop("num_tents", 5)
        self.norm = kwargs.pop("norm", True)
        self.separate_drift = kwargs.pop("separate_drift", False)

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
            thresh=self.thresh,
        )

        if (
            self.regions[0] == "DMS"
            and self.regions[1] == "DLS"
            and len(self.regions) == 2
        ):
            self.regions = ["DLS", "DMS"]

    def build_dm(self):
        if not (hasattr(self, "psths")):
            self.get_data()

        data_gd, _ = get_dataset_dm(
            self.psths,
            self.trial_data,
            strategy_filter=None,
            regions=self.regions,
            norm=self.norm,
            num_tents=self.num_tents,
            task_vars=self.task_vars,
            binwidth_ms=25,
            sanity_check=0,
        )

        self.sample = data_gd[:]
        self.num_trials, self.num_tv = self.sample["tv"].shape
        self.num_units = self.sample["robs"].shape[1]

        self.robs = self.sample["robs"].detach().cpu().numpy()

        self.tvs = np.asarray(self.sample["tv"].detach().cpu().numpy())
        self.tents = self.sample["tents"].detach().cpu().numpy()
        self.dm = np.hstack((self.tents, self.tvs))

        ohe = OHE().fit(self.trial_data[self.task_vars])
        self.tv_names = np.concatenate(
            (
                [f"tents_{i}" for i in range(self.tents.shape[1])],
                ohe.get_feature_names_out(),
            )
        )

    def fit_baseline(self):
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
            if not hasattr(self, "encoder"):
                self.fit_encoder()

            dm_tv_ko = deepcopy(self.dm)
            dm_tv_ko[:, 5:] = 0

            self.robs_predict["ps_baseline"] = self.encoder.predict(dm_tv_ko)

    def fit_encoder(self):
        if not (hasattr(self, "dm") and hasattr(self, "robs")):
            self.build_dm()

        if self.separate_drift:
            self.fit_baseline()
            self.baseline_predict()

            self.encoder = RidgeCV(
                alphas=np.logspace(-5, 5, 11, base=10),
                alpha_per_target=True,
            ).fit(self.tvs, self.robs - self.robs_predict["baseline"])
        else:
            self.encoder = RidgeCV(
                alphas=np.logspace(-5, 5, 11, base=10),
                alpha_per_target=True,
            ).fit(self.dm, self.robs)

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

    def get_r2(self, n_folds=10, p_train=0.8):
        if not hasattr(self, "robs"):
            self.build_dm()

        self.scores_cv = {
            "baseline": np.zeros((n_folds, self.num_units)),
            "ps_baseline": np.zeros((n_folds, self.num_units)),
            "encoder": np.zeros((n_folds, self.num_units)),
        }

        for i in range(n_folds):
            train_idxs = np.sort(
                np.random.choice(
                    self.num_trials, int(self.num_trials * p_train), replace=False
                )
            )
            test_idxs = np.setdiff1d(np.arange(self.num_trials), train_idxs)

            baseline_model = RidgeCV(
                alphas=np.logspace(-5, 5, 11, base=10), alpha_per_target=True
            ).fit(self.tents[train_idxs], self.robs[train_idxs])

            self.scores_cv["baseline"][i] = r2_score(
                self.robs[test_idxs],
                baseline_model.predict(self.tents[test_idxs]),
                multioutput="raw_values",
            )

            if self.separate_drift:
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
                )

                self.scores_cv["ps_baseline"][i] = self.scores_cv["baseline"][i]

            else:
                encoder = RidgeCV(
                    alphas=np.logspace(-5, 5, 11, base=10),
                    alpha_per_target=True,
                ).fit(self.dm[train_idxs], self.robs[train_idxs])

                self.scores_cv["encoder"][i] = r2_score(
                    self.robs[test_idxs],
                    encoder.predict(self.dm[test_idxs]),
                    multioutput="raw_values",
                )

                dm_tv_ko = deepcopy(self.dm)
                dm_tv_ko[:, 5:] = 0

                self.scores_cv["ps_baseline"][i] = r2_score(
                    self.robs[test_idxs],
                    encoder.predict(dm_tv_ko[test_idxs]),
                    multioutput="raw_values",
                )

        self.scores = {
            "baseline": np.median(self.scores_cv["baseline"], axis=0),
            "ps_baseline": np.median(self.scores_cv["ps_baseline"], axis=0),
            "encoder": np.median(self.scores_cv["encoder"], axis=0),
        }

    def verify(self, cond="response", subtract_baseline=True):
        _, axes = plt.subplots(ncols=5, nrows=1, figsize=(7.5, 1.5), tight_layout=True)

        # baseline vs encoder r2
        self.plot_r2_comp(axes[0])

        # sctavg vs beta weight
        self.plot_sctavg_weights(
            axes[1:3], cond="response", subtract_baseline=subtract_baseline
        )
        self.plot_sctavg_weights(
            axes[3:5], cond="rewarded", subtract_baseline=subtract_baseline
        )

    def plot_r2_comp(self, ax=None):
        if not hasattr(self, "scores"):
            self.get_r2()

        if ax is None:
            _, ax = plt.figure(tight_layout=True)

        ax.scatter(self.scores["baseline"], self.scores["encoder"], s=0.5, alpha=0.5)
        ax.plot([-0.5, 1], [-0.5, 1], color="#666666", linestyle="--", linewidth=0.5)
        ax.axhline(y=0, color="k", linewidth=0.5)
        ax.axvline(x=0, color="k", linewidth=0.5)

        ax.set_xlabel(r"$r^2$, baseline")
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

        if subtract_baseline:
            if (
                not hasattr(self, "robs_predict")
                or "ps_baseline" not in self.robs_predict.keys()
            ):
                self.baseline_predict(pseudo=True)
            robs_baseline = self.robs_predict["ps_baseline"]
        else:
            robs_baseline = None

        sc_tavg = get_tavg_sc_cond(
            self.robs,
            self.trial_data,
            cond=cond,
            robs_drift=robs_baseline,
            subtract_drift=subtract_baseline,
        )
        if cond == "response":
            keys = ["left", "right"]
            idxs = [6, 5]
        elif cond == "rewarded":
            keys = ["corr", "incorr"]
            idxs = [8, 7]

        for i in range(2):
            axes[i].scatter(
                sc_tavg[keys[i]], self.encoder.coef_[:, idxs[i]], s=0.5, alpha=0.5
            )
            axes[i].axhline(y=0, color="k", linewidth=0.5)
            axes[i].axvline(x=0, color="k", linewidth=0.5)

            axes[i].set_xlabel(f"avg norm sc, {keys[i]}")
            axes[i].set_ylabel(f"beta weight, {keys[i]}")

    def view_fits(self, model="encoder"):
        if not hasattr(self, "robs_predict") or model not in self.robs_predict.keys():
            if model == "encoder":
                self.encoder_predict()
            elif model == "baseline":
                self.baseline_predict()
            else:
                raise ValueError(
                    f"valid arguments for model are 'encoder' and 'baseline,' not {model}"
                )

        r = FitRenderer(y=self.robs, yhat=self.robs_predict[model], mode="lite")

        _ = NeuronViewer(num_units=self.num_units, render_func=r, fig_dir=FIGURES_DIR)


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

        self.task_vars = self.encoder_full.task_vars

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
            for tv in encoder_shuffle.task_vars:
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
