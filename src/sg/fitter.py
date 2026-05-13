"""
fitter.py

Encoders and LVMs

Author: Stellina X. Ao
Created: 2026-03-26
Last Modified: 2026-04-03
Python Version: 3.11.14
"""

import random

import numpy as np
import torch


from core.data import load_sess
from sg.fitlvm_utils import (
    eval_model,
    fit_autoencoder,
    fit_gain_model,
    fit_model,
    get_data_model,
)
from sg.models import SharedGain

"""
TODO
- add verbosity flag
"""


class Encoder:
    def __init__(
        self,
        subj_id: str = None,
        sess_id: str = None,
        **kwargs,
    ):
        self.subj_id = subj_id
        self.sess_id = sess_id

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # kwargs
        # seed & verbosity & sanity check
        self.seed_val = kwargs.pop("seed", 1234)
        self.seed()

        self.verbosity = kwargs.pop("verbosity", 0)
        self.sanity_check = kwargs.pop("sanity_check", 0)

        # trial data
        self.idxs_subsamp = kwargs.pop("idxs_subsamp", None)
        self.balance_strategy = kwargs.pop("balance_strategy", False)
        self.mb_only = kwargs.pop("mb_only", False)
        self.mf_only = kwargs.pop("mf_only", False)
        if self.mb_only and self.mf_only:
            raise ValueError("gosh pick a side!")
        if self.mb_only or self.mf_only:
            self.subsample_strategy = kwargs.pop("subsample_strategy", True)
        self.enough_trials = False

        # neural data
        self.tpre = kwargs.pop("tpre", 0.5)
        self.tpost = kwargs.pop("tpost", 1)
        self.alignment = kwargs.pop("alignment", "choice")

        self.tpre_ref = kwargs.pop("tpre_ref", 0.5)
        self.tpost_ref = kwargs.pop("tpost_ref", 1)
        self.alignment_ref = kwargs.pop("alignment_ref", "choice")

        self.binwidth_ms = kwargs.pop("binwidth_ms", 25)
        self.trial_start_pre = kwargs.pop("trial_start_pre", 0)
        self.thresh = kwargs.pop("thresh", 1)

        self.regions = kwargs.pop("regions", None)

        # model params
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
        self.n_splines = kwargs.pop("n_splines", 5)
        self.norm_activity = kwargs.pop("norm_activity", True)
        self.nonlinearity = kwargs.pop("nonlinearity", "Identity")

        self.tv_actv_fn = kwargs.pop("tv_actv_fn", "lin")
        self.tv_reg = kwargs.pop("tv_reg", {"l2": 0.01})

        self.reg = kwargs.pop("reg", {"l2": 0.001})

        if len(kwargs) > 0:
            extra_kwargs = ", ".join('"%s' % k for k in list(kwargs.keys()))
            raise ValueError("Extra arguments %s" % extra_kwargs)

        # housekeeping
        self.baseline_fit = False
        self.taskvar_fit = False

    def fit_all(self, cids=None):
        self.get_data()
        if self.enough_trials:
            self.fit_baseline()
            self.fit_taskvar()
            if cids is None:
                self.get_cids()
            self.update_cids(cids)

    def seed(self):
        random.seed(int(self.seed_val))
        np.random.seed(self.seed_val)
        torch.manual_seed(self.seed_val)
        torch.cuda.manual_seed(self.seed_val)
        torch.cuda.manual_seed_all(self.seed_val)

    def get_data(self):
        (
            self.spike_times,
            self.trial_data,
            self.psths,
            self.session_data,
            regions,
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
            trial_start_pre=self.trial_start_pre,
            thresh=self.thresh,
        )

        if self.regions is None:
            # i.e., fit to everything
            self.regions = regions
        else:
            # check that the regions specified in self.regions are actually valid
            if not set(self.regions).issubset(set(regions)):
                raise ValueError(f"{self.regions} must be a subset of {regions}")
            else:
                self.psths = {reg: self.psths[reg] for reg in self.regions}
                self.spike_times = {reg: self.spike_times[reg] for reg in self.regions}

        if self.sanity_check == 1:
            self.psths["DMS"] *= 20

        # update trial_data and psths if needed
        if self.mb_only:
            mb_mask = self.trial_data["strategy"] == 1

            if not self.subsample_strategy:
                if mb_mask.sum() < 20:
                    return
                self.enough_trials = True
                self.trial_data = self.trial_data[mb_mask]
                self.psths = {
                    region: self.psths[region][:, mb_mask, :] for region in self.regions
                }
            else:
                mf_mask = self.trial_data["strategy"] == -1

                num_trial = min(mb_mask.sum(), mf_mask.sum())
                if num_trial < 20:
                    return
                self.enough_trials = True
                self.idxs_subsamp = np.sort(
                    np.random.choice(np.where(mb_mask)[0], num_trial, replace=False)
                )

                self.trial_data = self.trial_data.iloc[self.idxs_subsamp]
                self.psths = {
                    region: self.psths[region][:, self.idxs_subsamp, :]
                    for region in self.regions
                }

        elif self.mf_only:
            mf_mask = self.trial_data["strategy"] == -1
            if not self.subsample_strategy:
                if mf_mask.sum() < 20:
                    return
                self.enough_trials = True
                self.trial_data = self.trial_data[mf_mask]
                self.psths = {
                    region: self.psths[region][:, mf_mask, :] for region in self.regions
                }
            else:
                mb_mask = self.trial_data["strategy"] == 1

                num_trial = min(mb_mask.sum(), mf_mask.sum())

                if num_trial < 20:
                    return
                self.enough_trials = True
                self.idxs_subsamp = np.sort(
                    np.random.choice(np.where(mf_mask)[0], num_trial, replace=False)
                )

                self.trial_data = self.trial_data.iloc[self.idxs_subsamp]
                self.psths = {
                    region: self.psths[region][:, self.idxs_subsamp, :]
                    for region in self.regions
                }

        elif self.balance_strategy:
            mb_mask = self.trial_data["strategy"] == 1
            mf_mask = self.trial_data["strategy"] == -1

            num_trial = min(mb_mask.sum(), mf_mask.sum())

            if num_trial * 2 < 20:
                return
            self.enough_trials = True
            self.idxs_subsamp_mb = np.random.choice(
                np.where(mb_mask)[0], num_trial, replace=False
            )
            self.idxs_subsamp_mf = np.random.choice(
                np.where(mf_mask)[0], num_trial, replace=False
            )

            self.idxs_subsamp = np.sort(
                np.concatenate((self.idxs_subsamp_mb, self.idxs_subsamp_mf))
            )

            self.trial_data = self.trial_data.iloc[self.idxs_subsamp]
            self.psths = {
                region: self.psths[region][:, self.idxs_subsamp, :]
                for region in self.regions
            }
        elif self.idxs_subsamp is not None:
            if len(self.idxs_subsamp) < 20:
                return
            self.enough_trials = True

            self.idxs_subsamp = np.sort(self.idxs_subsamp)

            self.trial_data = self.trial_data.iloc[self.idxs_subsamp]
            self.psths = {
                region: self.psths[region][:, self.idxs_subsamp, :]
                for region in self.regions
            }
        else:
            if self.trial_data.shape[0] > 20:
                self.enough_trials = True

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
        self.baseline_fit = True

    def fit_taskvar(self):
        # self.tv_reg = {"l2": 0.001}
        # self.reg = {"l2": 0.001}
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
        self.taskvar_fit = True

    def get_cids(self):
        res_baseline = eval_model(self.mod_baseline, self.data_gd, self.test_dl.dataset)
        res_taskvar = eval_model(self.mod_taskvar, self.data_gd, self.test_dl.dataset)
        self.cids_tv_zero = np.where(res_taskvar["r2test"] > 0)[0]
        self.cids_tv_baseline = np.where(
            res_taskvar["r2test"] > res_baseline["r2test"]
        )[0]
        self.cids = np.intersect1d(self.cids_tv_zero, self.cids_tv_baseline)
        # self.cids_pca = check_stable_lowd(
        #     self.data_gd,
        #     self.train_dl.dataset[:]["dfs"] > 0,
        #     self.val_dl.dataset[:]["dfs"] > 0,
        #     self.num_units,
        #     rank=4,
        # )
        # # it was this stinker that kept letting things through
        # self.cids = np.intersect1d(
        #     self.cids_tv, self.cids_pca
        # )  # changed from union to intersection

    def update_cids(self, cids=None):
        if cids is not None:
            self.cids = cids
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
        if self.baseline_fit:
            self.res_baseline = eval_model(
                self.mod_baseline, self.data_gd, self.test_dl.dataset
            )

        # task variables
        if self.taskvar_fit:
            self.res_taskvar = eval_model(
                self.mod_taskvar, self.data_gd, self.test_dl.dataset
            )


class LVMFamily(Encoder):
    def __init__(
        self,
        subj_id: str = None,
        sess_id: str = None,
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

        super().__init__(subj_id, sess_id, **kwargs)

        # housekeeping
        self.ae_gain_fit = False
        self.ae_offset_fit = False
        self.ae_affine_fit = False
        self.lvms_fit = False

    def fit_all(self):
        super().fit_all()

        if self.enough_trials and self.num_units > 0:
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
        # self.tv_reg = {"l2": 1}
        # self.reg = {"l2": 0.001}
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

        try:
            fit_autoencoder(
                self.mod_ae_gain,
                self.train_dl,
                self.val_dl,
                min_iter=0,
                max_iter=self.max_iter,
                verbosity=self.verbosity,
            )
        except RuntimeError:
            print(self.num_trials, self.num_units)
            print(
                self.robs.shape,
                self.train_dl.dataset[:]["dfs"].sum(),
                self.val_dl.dataset[:]["dfs"].sum(),
                self.test_dl.dataset[:]["dfs"].sum(),
            )

        self.ae_gain_fit = True

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

        # added and did nothing...\
        self.mod_ae_offset.tv.bias.requires_grad = False
        self.mod_ae_offset.bias.requires_grad = False
        self.mod_ae_offset.drift.weight.requires_grad = False

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

        self.ae_offset_fit = True

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
        self.ae_affine_fit = True

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
        self.lvms_fit = True

    def eval(self, do_taskvar=True, do_lvm=True):
        if do_taskvar:
            super().eval()

        # lvms
        if do_lvm:
            if (not self.no_mult) and (self.ae_gain_fit and self.lvms_fit):
                self.res_gain = eval_model(
                    self.mod_gain, self.data_gd, self.test_dl.dataset
                )
            if (not self.no_addt) and (self.ae_offset_fit and self.lvms_fit):
                self.res_offset = eval_model(
                    self.mod_offset, self.data_gd, self.test_dl.dataset
                )
            if self.ae_affine_fit and self.lvms_fit:
                self.res_affine = eval_model(
                    self.mod_affine, self.data_gd, self.test_dl.dataset
                )
            self.get_qi()

    def get_qi(self):
        if self.lvms_fit and self.taskvar_fit:
            if self.no_mult:
                r2_lvm = self.res_offset["r2test"].mean()
            elif self.no_addt:
                r2_lvm = self.res_gain["r2test"].mean()
            elif self.lvms_fit:
                r2_lvm = self.res_affine["r2test"].mean()
            r2_taskvar = self.res_taskvar["r2test"].mean()
            self.qi = (r2_lvm - r2_taskvar) / (1 - r2_taskvar)
        else:
            self.qi = np.nan


"""
class ScrambledEncoder:
    def __init__(
        self,
        subj_id,
        sess_id,
        pivot: str = None,
        **kwargs,
    ):

        self.mod_full = Encoder(
            subj_id=subj_id,
            sess_id=sess_id,
            **kwargs,
        )

        self.mod_full.get_data()

        self.pivot = pivot

        if pivot not in self.mod_full.task_vars:
            raise ValueError(f"pivot must be one of {self.mod_full.task_vars}")

        self.trial_data_scramble_d = self.mod_full.trial_data.copy(deep=True)
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
"""
