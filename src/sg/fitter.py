import random

import numpy as np
import torch

from sg.data import get_psths
from sg.fitlvm_utils import (
    eval_model,
    fit_autoencoder,
    fit_gain_model,
    fit_model,
    get_data_model,
    check_stable_lowd,
)
from sg.models import SharedGain

"""
TODO
- add verbosity flag
"""


class Encoder:
    def __init__(
        self,
        trial_data=None,
        spike_times=None,
        session_data=None,
        regions=None,
        **kwargs,
    ):
        self.spike_times = spike_times
        self.session_data = session_data
        self.trial_data = trial_data
        self.regions = regions

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # kwargs
        # seed & verbosity & sanity check
        self.seed_val = kwargs.pop("seed", 1234)
        self.seed()

        self.verbosity = kwargs.pop("verbosity", 0)
        self.sanity_check = kwargs.pop("sanity_check", 0)

        # neural data
        self.tpre = kwargs.pop("tpre", 0.5)
        self.tpost = kwargs.pop("tpost", 0.5)
        self.binwidth_ms = kwargs.pop("binwidth_ms", 25)
        self.alignment = kwargs.pop("alignment", "choice")

        # model params
        self.task_vars = kwargs.pop(
            "task_vars",
            [
                "response",
                "rewarded",
                "block_side",
                "strategy",
                "response_prev",
                "rewarded_prev",
            ],
        )
        self.n_splines = kwargs.pop("n_splines", 5)
        self.norm_activity = kwargs.pop("norm_activity", True)
        self.nonlinearity = kwargs.pop("nonlinearity", "Identity")

        self.tv_actv_fn = kwargs.pop("tv_actv_fn", "lin")
        self.tv_reg = kwargs.pop("tv_reg", {"l2": 0.01})

        self.reg = kwargs.pop("reg", {"l2": 0.001})

        if len(kwargs) > 0:
            extra_kwargs = ", ".join('"%s' % k for k in list(kwargs.keys()))
            raise ValueError("Extra arguments %s" % extra_kwargs)

    def fit_all(self):
        self.get_data()
        self.fit_baseline()
        self.fit_taskvar()
        self.get_cids()
        self.update_cids()

    def seed(self):
        random.seed(self.seed_val)
        np.random.seed(self.seed_val)
        torch.manual_seed(self.seed_val)
        torch.cuda.manual_seed(self.seed_val)
        torch.cuda.manual_seed_all(self.seed_val)

    def get_data(self):
        self.psths, self.trial_mask, self.zstd_units = get_psths(
            self.spike_times,
            self.trial_data,
            self.session_data,
            self.regions,
            tpre=self.tpre,
            tpost=self.tpost,
            binwidth_ms=self.binwidth_ms,
            alignment=self.alignment,
            reward_only=False,
            prev_filter=False,
            get_strategy=False,
        )

        # update spike_times with the removed units
        for reg in self.regions:
            if len(self.zstd_units[reg]) > 0:
                self.spike_times[reg] = [
                    st_unit
                    for i, st_unit in enumerate(self.spike_times[reg])
                    if i not in set(self.zstd_units[reg])
                ]
        if self.sanity_check == 1:
            self.psths["DMS"] *= 20
        self.trial_data = self.trial_data[self.trial_mask]

        self.strategy = self.trial_data["strategy"]
        self.rewarded = self.trial_data["rewarded"]
        self.response = self.trial_data["response"]
        self.rewarded_prev = self.trial_data["rewarded_prev"]
        self.response_prev = self.trial_data["response_prev"]
        self.block_side = self.trial_data["block_side"]

        (
            self.data_gd,
            self.train_dl,
            self.val_dl,
            self.test_dl,
            self.indices,
            self.num_trials,
            self.num_tv,
            self.num_units,
        ) = get_data_model(
            self.psths,
            self.trial_data,
            self.regions,
            norm=self.norm_activity,
            num_tents=self.n_splines,
            task_vars=self.task_vars,
            sanity_check=self.sanity_check,
        )
        self.sample = self.data_gd[:]
        self.robs = self.sample["robs"].detach().cpu().numpy()

    def fit_baseline(self):
        self.mod_baseline = SharedGain(
            tv_dims=self.num_tv,
            num_units=self.num_units,
            cids=None,
            num_latent_mult=0,
            num_latent_addt=0,
            num_tents=self.n_splines,
            include_tv=False,
            include_gain=False,
            include_offset=False,
            tents_as_input=False,
            output_nonlinearity=self.nonlinearity,
            tv_act_func=self.tv_actv_fn,
            tv_reg_vals=self.tv_reg,
            reg_vals=self.reg,
        )
        fit_model(self.mod_baseline, self.train_dl, self.val_dl, use_lbfgs=True)

    def fit_taskvar(self):
        self.tv_reg = {"l2": 0.001}
        self.reg = {"l2": 0.001}
        if self.verbosity > 0:
            print(self.tv_actv_fn, self.nonlinearity)
        self.mod_taskvar = SharedGain(
            tv_dims=self.num_tv,
            num_units=self.num_units,
            cids=None,
            num_latent_mult=0,
            num_latent_addt=0,
            num_tents=self.n_splines,
            include_tv=True,
            include_gain=False,
            include_offset=False,
            tents_as_input=False,
            output_nonlinearity=self.nonlinearity,
            tv_act_func=self.tv_actv_fn,
            tv_reg_vals=self.tv_reg,
            reg_vals=self.reg,
        )
        self.mod_taskvar.drift.weight.data = self.mod_baseline.drift.weight.data.clone()

        fit_model(self.mod_taskvar, self.train_dl, self.val_dl, use_lbfgs=True)

    def get_cids(self):
        res_taskvar = eval_model(self.mod_taskvar, self.data_gd, self.test_dl.dataset)
        self.cids_tv = np.where(res_taskvar["r2test"] > 0)[0]
        self.cids_pca = check_stable_lowd(
            self.data_gd,
            self.train_dl.dataset[:]["dfs"] > 0,
            self.val_dl.dataset[:]["dfs"] > 0,
            self.num_units,
            rank=4,
        )
        # it was this stinker that kept letting things through
        self.cids = np.intersect1d(
            self.cids_tv, self.cids_pca
        )  # changed from union to intersection

    def update_cids(self):
        # housekeeping
        self.data_gd[:]["robs"] = self.data_gd[:]["robs"][:, self.cids]
        self.sample["robs"] = self.sample["robs"][:, self.cids]
        self.robs = self.robs[:, self.cids]

        self.sample["reg_keys"] = self.sample["reg_keys"][self.cids]

        self.num_units = len(self.cids)

        # baseline
        self.mod_baseline.cids = self.cids
        self.mod_baseline.bias.data = self.mod_baseline.bias.data[self.cids]
        self.mod_baseline.drift.weight.data = self.mod_baseline.drift.weight.data[
            :, self.cids
        ]
        self.mod_baseline.drift.bias.data = self.mod_baseline.drift.bias.data[self.cids]

        # task variables
        self.mod_taskvar.cids = self.cids
        self.mod_taskvar.bias.data = self.mod_taskvar.bias.data[self.cids]

        self.mod_taskvar.drift.weight.data = self.mod_taskvar.drift.weight.data[
            :, self.cids
        ]
        self.mod_taskvar.tv.weight.data = self.mod_taskvar.tv.weight.data[:, self.cids]

        self.mod_taskvar.drift.bias.data = self.mod_taskvar.drift.bias.data[self.cids]
        self.mod_taskvar.tv.bias.data = self.mod_taskvar.tv.bias.data[self.cids]

    def eval(self):
        # baseline
        self.res_baseline = eval_model(
            self.mod_baseline, self.data_gd, self.test_dl.dataset
        )

        # task variables
        self.res_taskvar = eval_model(
            self.mod_taskvar, self.data_gd, self.test_dl.dataset
        )


class LVMFamily(Encoder):
    def __init__(
        self,
        trial_data=None,
        spike_times=None,
        session_data=None,
        regions=None,
        **kwargs,
    ):
        """
        kwargs:
        """

        self.d2ts = kwargs.pop("d2ts", [0.01])

        self.n_latents_mult = kwargs.pop("n_latents_mult", 1)
        self.n_latents_addt = kwargs.pop("n_latents_addt", 1)

        self.no_mult = self.n_latents_mult == 0
        self.no_addt = self.n_latents_addt == 0
        if self.no_mult and self.no_addt:
            print("WOWZA. someone is feeling nihilistic. try again.")

        self.add_latent_noise = kwargs.pop("add_latent_noise", False)

        self.refit = kwargs.pop("refit", False)
        self.max_iter = kwargs.pop("max_iter", 10) if self.refit else 0

        super().__init__(trial_data, spike_times, session_data, regions, **kwargs)

    def fit_all(self):
        super().fit_all()

        if not self.no_mult:
            self.fit_ae_gain()
        if not self.no_addt:
            self.fit_ae_offset()
        if not self.no_mult and not self.no_addt:
            self.fit_ae_affine()
        elif self.no_mult:
            self.mod_ae_affine = self.mod_ae_offset
        elif self.no_addt:
            self.mod_ae_affine = self.mod_ae_gain
        else:
            print("BOOHOO something is catastrophically wrong")
            return

        self.ae2lvm()

    def fit_ae_gain(self):
        self.tv_reg = {"l2": 1}
        self.reg = {"l2": 0.001}
        self.mod_ae_gain = SharedGain(
            tv_dims=self.num_tv,
            num_units=self.num_units,
            cids=self.cids,
            num_latent_mult=self.n_latents_mult,
            num_latent_addt=0,
            num_tents=self.n_splines,
            latent_noise=self.add_latent_noise,
            include_tv=True,
            include_gain=True,
            include_offset=False,
            tents_as_input=False,
            output_nonlinearity=self.nonlinearity,
            tv_act_func=self.tv_actv_fn,
            tv_reg_vals=self.tv_reg,
            reg_vals=self.reg,
        )

        if self.n_splines > 1:
            self.mod_ae_gain.drift.weight.data = (
                self.mod_taskvar.drift.weight.data.clone()
            )
            self.mod_ae_gain.bias.requires_grad = False
        else:
            self.mod_ae_gain.bias.requires_grad = True

        self.mod_ae_gain.tv.weight.data = self.mod_taskvar.tv.weight.data.clone()
        self.mod_ae_gain.bias.data = self.mod_taskvar.bias.data.clone()
        self.mod_ae_gain.tv.weight.requires_grad = False

        self.mod_ae_gain.readout_gain.weight_scale = 1.0
        self.mod_ae_gain.latent_gain.weight_scale = 1.0
        self.mod_ae_gain.readout_gain.weight.data[:] = 1.0

        self.mod_ae_gain.prepare_regularization()

        fit_autoencoder(
            self.mod_ae_gain,
            self.train_dl,
            self.val_dl,
            min_iter=0,
            max_iter=self.max_iter,
            verbosity=self.verbosity,
        )

    def fit_ae_offset(self):
        self.mod_ae_offset = SharedGain(
            tv_dims=self.num_tv,
            num_units=self.num_units,
            cids=self.cids,
            num_latent_mult=0,
            num_latent_addt=self.n_latents_addt,
            num_tents=self.n_splines,
            latent_noise=self.add_latent_noise,
            include_tv=True,
            include_gain=False,
            include_offset=True,
            tents_as_input=False,
            output_nonlinearity=self.nonlinearity,
            tv_act_func=self.tv_actv_fn,
            tv_reg_vals=self.tv_reg,
            reg_vals=self.reg,
        )

        if self.n_splines > 1:
            self.mod_ae_offset.drift.weight.data = (
                self.mod_taskvar.drift.weight.data.clone()
            )
            self.mod_ae_offset.bias.requires_grad = False
        else:
            self.mod_ae_offset.bias.requires_grad = True

        self.mod_ae_offset.tv.weight.data = self.mod_taskvar.tv.weight.data.clone()
        self.mod_ae_offset.bias.data = self.mod_taskvar.bias.data.clone()
        self.mod_ae_offset.tv.weight.requires_grad = False

        self.mod_ae_offset.readout_offset.weight_scale = 1.0
        self.mod_ae_offset.latent_offset.weight_scale = 1.0
        self.mod_ae_offset.readout_offset.weight.data[:] = 1.0

        self.mod_ae_offset.prepare_regularization()

        fit_autoencoder(
            self.mod_ae_offset,
            self.train_dl,
            self.val_dl,
            min_iter=0,
            max_iter=self.max_iter,
            verbosity=self.verbosity,
        )

    def fit_ae_affine(self):
        self.mod_ae_affine = SharedGain(
            tv_dims=self.num_tv,
            num_units=self.num_units,
            cids=self.cids,
            num_latent_mult=self.n_latents_mult,
            num_latent_addt=self.n_latents_addt,
            num_tents=self.n_splines,
            latent_noise=self.add_latent_noise,
            include_tv=True,
            include_gain=True,
            include_offset=True,
            tents_as_input=False,
            output_nonlinearity=self.nonlinearity,
            tv_act_func=self.tv_actv_fn,
            tv_reg_vals=self.tv_reg,
            reg_vals=self.reg,
        )

        if self.n_splines > 1:
            self.mod_ae_affine.drift.weight.data = (
                self.mod_taskvar.drift.weight.data.clone()
            )
            self.mod_ae_affine.drift.weight.requires_grad = False
            self.mod_ae_affine.bias.requires_grad = False
        else:
            self.mod_ae_affine.bias.requires_grad = True

        # initialize neuron-tv weights with tv model weights
        self.mod_ae_affine.tv.weight.data = self.mod_taskvar.tv.weight.data.clone()
        self.mod_ae_affine.bias.data = self.mod_taskvar.bias.data.clone()
        self.mod_ae_affine.tv.weight.requires_grad = False

        # intialize coupling weights with gain and offset only ae models
        self.mod_ae_affine.readout_gain.weight.data[:] = (
            self.mod_ae_gain.readout_gain.weight.data.detach().clone()
        )  # .shape >> torch.Size([3, 173])
        self.mod_ae_affine.readout_offset.weight.data[:] = (
            self.mod_ae_offset.readout_offset.weight.data.detach().clone()
        )

        self.mod_ae_affine.latent_gain.weight.data[:] = (
            self.mod_ae_gain.latent_gain.weight.data.detach().clone()
        )  # .shape >> torch.Size([173, 3])
        self.mod_ae_affine.latent_offset.weight.data[:] = (
            self.mod_ae_offset.latent_offset.weight.data.detach().clone()
        )

        self.mod_ae_affine.prepare_regularization()

        # fit_autoencoder: initialize latents by only fitting latents, then refit task var and refit latents
        fit_autoencoder(
            self.mod_ae_affine,
            self.train_dl,
            self.val_dl,
            min_iter=0,
            max_iter=self.max_iter,
            verbosity=self.verbosity,
        )

    def ae2lvm(self):
        if not self.no_mult:
            self.mod_gain = fit_gain_model(
                tv_dims=self.num_tv,
                mod1=self.mod_ae_gain,
                num_units=self.num_units,
                num_trials=self.num_trials,
                cids=self.cids,
                num_latent_mult=self.n_latents_mult,
                num_latent_addt=self.n_latents_addt,
                ntents=self.n_splines,
                include_gain=True,
                include_offset=False,
                l2s=[self.reg["l2"]],
                d2ts=self.d2ts,
                train_dl=self.train_dl,
                val_dl=self.val_dl,
                max_iter=self.max_iter,
            )

        if not self.no_addt:
            self.mod_offset = fit_gain_model(
                tv_dims=self.num_tv,
                mod1=self.mod_ae_offset,
                num_units=self.num_units,
                num_trials=self.num_trials,
                cids=self.cids,
                num_latent_mult=self.n_latents_mult,
                num_latent_addt=self.n_latents_addt,
                ntents=self.n_splines,
                include_gain=False,
                include_offset=True,
                l2s=[self.reg["l2"]],
                d2ts=self.d2ts,
                train_dl=self.train_dl,
                val_dl=self.val_dl,
                max_iter=self.max_iter,
            )

        if not self.no_mult and not self.no_addt:
            self.mod_affine = fit_gain_model(
                tv_dims=self.num_tv,
                mod1=self.mod_ae_affine,
                num_units=self.num_units,
                num_trials=self.num_trials,
                cids=self.cids,
                num_latent_mult=self.n_latents_mult,
                num_latent_addt=self.n_latents_addt,
                ntents=self.n_splines,
                include_gain=True,
                include_offset=True,
                l2s=[self.reg["l2"]],
                d2ts=self.d2ts,
                train_dl=self.train_dl,
                val_dl=self.val_dl,
                max_iter=self.max_iter,
            )
        elif self.no_mult:
            self.mod_affine = self.mod_offset
        elif self.no_addt:
            self.mod_affine = self.mod_gain
        else:
            print("KABOOM. The world exploded because you made a non sequitur.")

    def eval(self, do_taskvar=True, do_lvm=True):
        if do_taskvar:
            super().eval()

        # lvms
        if do_lvm:
            if not self.no_mult:
                self.res_gain = eval_model(
                    self.mod_gain, self.data_gd, self.test_dl.dataset
                )
            if not self.no_addt:
                self.res_offset = eval_model(
                    self.mod_offset, self.data_gd, self.test_dl.dataset
                )
            self.res_affine = eval_model(
                self.mod_affine, self.data_gd, self.test_dl.dataset
            )


class ScrambledEncoder:
    def __init__(
        self,
        trial_data=None,
        spike_times=None,
        session_data=None,
        regions=None,
        pivot: str = None,
        **kwargs,
    ):

        self.mod_full = Encoder(
            trial_data=trial_data,
            spike_times=spike_times,
            session_data=session_data,
            regions=regions,
            **kwargs,
        )

        self.pivot = pivot

        if pivot not in self.mod_full.task_vars:
            raise ValueError(f"pivot must be one of {self.mod_full.task_vars}")

        self.trial_data_scramble_d = trial_data.copy(deep=True)
        self.trial_data_scramble_d[self.pivot] = (
            self.trial_data_scramble_d[self.pivot].sample(frac=1).to_numpy()
        )

        self.mod_scramble_d = Encoder(
            trial_data=self.trial_data_scramble_d,
            spike_times=spike_times,
            session_data=session_data,
            regions=regions,
            **kwargs,
        )

        self.trial_data_scramble = trial_data.copy(deep=True)
        for regressor in self.mod_full.task_vars:
            if not regressor == self.pivot:
                self.trial_data_scramble[regressor] = (
                    self.trial_data_scramble[regressor].sample(frac=1).to_numpy()
                )

        self.mod_scramble = Encoder(
            trial_data=self.trial_data_scramble,
            spike_times=spike_times,
            session_data=session_data,
            regions=regions,
            **kwargs,
        )

    def fit_all(self):
        self.fit_full()
        self.fit_scramble_d()
        self.fit_scramble()

    def fit_full(self):
        self.mod_full.fit_all()

    def fit_scramble_d(self):
        self.mod_scramble_d.fit_all()

    def fit_scramble(self):
        self.mod_scramble.fit_all()

    def eval_full(self):
        self.mod_full.eval()

    def eval_scramble_d(self):
        self.mod_scramble_d.eval()

    def eval_scramble(self):
        self.mod_scramble.eval()

    def eval_all(self):
        self.eval_full()
        self.eval_scramble_d()
        self.eval_scramble()

        self.r2_full = self.mod_full.res_taskvar["r2test"].mean()
        self.r2_scramble_d = self.mod_scramble_d.res_taskvar["r2test"].mean()

        self.d_r2 = self.r2_full - self.r2_scramble_d
        self.r2_scramble = self.mod_scramble.res_taskvar["r2test"].mean()
