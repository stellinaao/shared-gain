import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from sg.fitlvm_utils import get_data_model
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

    def build_dm(self):
        if not (hasattr(self, "psths")):
            self.get_data()

        (self.data_gd, _, _, _, _, self.num_trials, self.num_tv, self.num_units) = (
            get_data_model(
                self.psths,
                self.trial_data,
                strategy_filter=None,
                regions=self.regions,
                norm=self.norm,
                num_tents=self.num_tents,
                task_vars=self.task_vars,
                sanity_check=0,
            )
        )

        self.sample = self.data_gd[:]

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

    def baseline_predict(self):
        if not hasattr(self, "robs_predict"):
            self.robs_predict = {}
        if not hasattr(self, "baseline_model"):
            self.fit_baseline()

        self.robs_predict["baseline"] = self.baseline_model.predict(self.tents)

    def fit_encoder(self):
        if not (hasattr(self, "dm") and hasattr(self, "robs")):
            self.build_dm()

        self.encoder = RidgeCV(
            alphas=np.logspace(-5, 5, 11, base=10),
            alpha_per_target=True,
        ).fit(self.dm, self.robs)

    def encoder_predict(self):
        if not hasattr(self, "robs_predict"):
            self.robs_predict = {}
        if not hasattr(self, "encoder"):
            self.fit_encoder()
        self.robs_predict["encoder"] = self.encoder.predict(self.dm)

    def get_r2(self, n_folds=10, p_train=0.8):
        if not hasattr(self, "robs"):
            self.build_dm()

        self.scores_cv = {
            "baseline": np.zeros((n_folds, self.num_units)),
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

            encoder = RidgeCV(
                alphas=np.logspace(-5, 5, 11, base=10),
                alpha_per_target=True,
            ).fit(self.dm[train_idxs], self.robs[train_idxs])

            self.scores_cv["encoder"][i] = r2_score(
                self.robs[test_idxs],
                encoder.predict(self.dm[test_idxs]),
                multioutput="raw_values",
            )

        self.scores = {
            "baseline": np.median(self.scores_cv["baseline"], axis=0),
            "encoder": np.median(self.scores_cv["encoder"], axis=0),
        }

    def verify(self):
        _, axes = plt.subplots(ncols=3, nrows=1, figsize=(6, 2), tight_layout=True)

        # baseline vs encoder r2
        self.plot_r2_comp(axes[0])

        # sctavg vs beta weight
        self.plot_sctavg_weights(axes[1:3])

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

    def plot_sctavg_weights(self, axes, cond="response"):
        if cond not in ["response"]:
            raise NotImplementedError(f"cond={cond} is not currently supported.")

        if axes is None:
            fig, axes = plt.subplots(
                ncols=2, nrows=1, figsize=(4, 2), tight_layout=True
            )

        if not hasattr(self, "encoder"):
            self.fit_encoder()

        sc_tavg = get_tavg_sc_cond(self.robs, self.trial_data, cond=cond)

        if cond == "response":
            axes[0].scatter(
                sc_tavg["right"], self.encoder.coef_[:, 5], s=0.5, alpha=0.5
            )
            axes[0].set_xlabel("avg norm sc, right")
            axes[0].set_ylabel("beta weight, right")

            axes[1].scatter(sc_tavg["left"], self.encoder.coef_[:, 6], s=0.5, alpha=0.5)
            axes[1].set_xlabel("avg norm sc, left")
            axes[1].set_ylabel("beta weight, left")

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
