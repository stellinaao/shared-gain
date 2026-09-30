"""
simons_nb.ipynb

figures for simons meeting

Author: Stellina X. Ao
Created: 2026-09-30
Last Modified: 2026-09-30
Python Version: 3.11.14
"""

import numpy as np
import h5py
import os
from core.data import subject_ids, session_ids
from utils.paths import VARS_DIR

from sg.models import (
    Encoder,
    StrategyEncoder,
    ShuffledEncoder,
    BootstrapperShuffle as BSS,
)


def enc_d(enc, tre=False):
    d = dict(
        scores=enc.scores,
        encoder_weights=enc.encoder_weights,
        alpha=enc.encoder.alpha_,
        max_reg=enc.max_reg,
        reg_idxs=enc.reg_idxs,
        regions=list(enc.regions),
        tv_keys=list(enc.tv_keys),
        tv_idxs=enc.tv_idxs,
        num_trials=enc.num_trials,
    )
    if tre:
        d.update(num_bins=enc.num_bins, epoch_keys_str=list(enc.epoch_keys_str))
    return d


def _write(g, key, val):
    key = str(key)
    if isinstance(val, dict):
        sub = g.create_group(key)
        for k, v in val.items():
            _write(sub, k, v)
    elif isinstance(val, (list, tuple)) and all(isinstance(x, str) for x in val):
        ds = g.create_dataset(key, data=np.array(val, dtype=h5py.string_dtype()))
        ds.attrs["kind"] = "str_list"
    elif isinstance(val, str):
        ds = g.create_dataset(key, data=val, dtype=h5py.string_dtype())
        ds.attrs["kind"] = "str"
    else:
        g.create_dataset(key, data=np.asarray(val))


def _read(node):
    if isinstance(node, h5py.Group):
        return {k: _read(v) for k, v in node.items()}
    v = node[()]
    kind = node.attrs.get("kind")
    if kind == "str_list":
        return [s.decode() for s in v]
    if kind == "str":
        return v.decode()
    return v.item() if v.ndim == 0 else v


def save_h5(path, data, meta=None):
    tmp = f"{path}.tmp"
    with h5py.File(tmp, "w") as f:
        for k, v in data.items():
            _write(f, k, v)
        for k, v in (meta or {}).items():
            f.attrs[k] = v
    os.replace(tmp, path)  # atomic: a file at `path` is always complete


for subj_id in ["MR82", "MR83", "MR94", "MR95"]:
    print(subj_id)
    for sess_id in session_ids[np.where(subject_ids == subj_id)[0][0]]:
        print(f">{sess_id}")

        out = VARS_DIR / subj_id / sess_id
        if (out / "bss_t.h5").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)

        try:
            # for population plots
            e_mb = StrategyEncoder(subj_id, sess_id, norm=True, strategy_filter="mb")
            e_mf = StrategyEncoder(subj_id, sess_id, norm=True, strategy_filter="mf")
            e = Encoder(subj_id, sess_id, norm=True)

            e_mb.fit_encoder()
            e_mf.fit_encoder()
            e.fit_encoder()

            e_mb.get_r2()
            e_mf.get_r2()
            e.get_r2()
            save_h5(
                out / "encoder.h5",
                {"full": enc_d(e), "mb": enc_d(e_mb), "mf": enc_d(e_mf)},
            )

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
            se_mb_.get_cvr2_all(n_iters=20)
            print(">> mb, dr2")
            se_mb_.get_dr2_all(n_iters=20)

            print(">> mf, cvr2")
            se_mf_.get_cvr2_all(n_iters=20)
            print(">> mf, dr2")
            se_mf_.get_dr2_all(n_iters=20)

            print(">> full, cvr2")
            se_.get_cvr2_all(n_iters=20)
            print(">> full, dr2")
            (se_.get_dr2_all(n_iters=20),)

            save_h5(
                out / "shuffle.h5",
                {
                    k: dict(cvr2_unit=s.cvr2_unit, dr2_unit=s.dr2_unit)
                    for k, s in [("full", se_), ("mb", se_mb_), ("mf", se_mf_)]
                },
            )

            bss_mb_ = BSS(
                subj_id,
                sess_id,
                StrategyEncoder,
                strategy_filter="mb",
                n=20,
                norm=True,
            )
            bss_mb_.get_ci_idxs()

            bss_mf_ = BSS(
                subj_id,
                sess_id,
                StrategyEncoder,
                strategy_filter="mf",
                n=20,
                norm=True,
            )
            bss_mf_.get_ci_idxs()

            save_h5(
                out / "bss_t.h5",
                {
                    "mb": dict(ci_idxs_reg=bss_mb_.ci_idxs_reg),
                    "mf": dict(ci_idxs_reg=bss_mf_.ci_idxs_reg),
                },
            )

            # # for tre single unit plots
            # print("tre")
            # encoder_t = make_tre(Encoder)(subj_id, sess_id, norm=True, stepsize_s=0.5)
            # encoder_mb_t = make_tre(StrategyEncoder)(
            #     subj_id, sess_id, norm=True, stepsize_s=0.5, strategy_filter="mb"
            # )
            # encoder_mf_t = make_tre(StrategyEncoder)(
            #     subj_id, sess_id, norm=True, stepsize_s=0.5, strategy_filter="mf"
            # )

            # encoder_mb_t.fit_encoder()
            # encoder_mf_t.fit_encoder()
            # encoder_t.fit_encoder()

            # encoder_mb_t.get_scores()
            # encoder_mf_t.get_scores()
            # encoder_t.get_scores()

            # save_h5(out / "shuffle.h5", {
            #     k: dict(cvr2_unit=s.cvr2_unit, dr2_unit=s.dr2_unit)
            #     for k, s in [("full", se_), ("mb", se_mb_), ("mf", se_mf_)]
            # })

            # # get cids
            # bss_mb_t = BSS(
            #     subj_id,
            #     sess_id,
            #     make_tre(StrategyEncoder),
            #     strategy_filter="mb",
            #     n=20,
            #     norm=True,
            #     stepsize_s=0.5,
            # )
            # bss_mb_t.get_ci_idxs()

            # bss_mf_t = BSS(
            #     subj_id,
            #     sess_id,
            #     make_tre(StrategyEncoder),
            #     strategy_filter="mf",
            #     n=20,
            #     norm=True,
            #     stepsize_s=0.5,
            # )
            # bss_mf_t.get_ci_idxs()

        except ValueError:
            print("not enough trials...")
            continue

        # save_h5(out / "bss_t.h5", {
        #     "mb": dict(ci_idxs_reg=bss_mb_t.ci_idxs_reg),
        #     "mf": dict(ci_idxs_reg=bss_mf_t.ci_idxs_reg),
        # })

        # save from encoder, strategyencoder (x2): scores, encoder_weights, encoder.alpha_, encoder.max_reg, reg_idxs, regions, tv_keys, tv_idxs, num_trials
        # save from ses, (full, mb, mf): cvr2_unit, dr2_unit
        # save from bss (mb, mf): ci_reg_idxs
        # save from encoder_t (full, mb, mf): scores, encoder_weights, encoder.alpha_, encoder.max_reg, num_bins, epoch_keys_str, reg_idxs, regions, tv_keys, tv_idxs, num_trials
        # save from bss_t (mb, mf): ci_reg_idxs
