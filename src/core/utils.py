"""
utils.py

General-purpose utilities.

Author: Stellina X. Ao
Created: 2025-12-18
Last Modified: 2026-01-12
Python Version: 3.11.14
"""

from typing import Dict

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from sg import models


def save_var(var_name, tag, data):
    fname = f"vars/{var_name}-{tag}.npy"
    if data is dict:
        save_jagged(var_name, tag, data)  # redirect to the correct function
    else:
        np.save(fname, data)


def load_var(var_name, tag):
    fname = f"vars/{var_name}-{tag}.npy"

    return np.load(fname)


def save_jagged(var_name, tag, data):
    fname = f"vars/{var_name}-{tag}.npz"
    if data is dict:
        np.savez(fname, **data)
    else:
        np.savez(fname, *data)


def load_jagged(var_name, tag):
    fname = f"vars/{var_name}-{tag}.npz"

    load = np.load(fname)
    data = [load[key] for key in load]
    load.close()

    return data


def list_dict_equiv(d1s, d2s, verbose=False):
    if len(d1s) != len(d2s):
        return False

    if not all([len(d1s[i]) == len(d2s[i]) for i in range(len(d1s))]):
        return False

    return all(
        [
            dict_equiv(d1s[i][j], d2s[i][j], verbose=verbose)
            for i in range(len(d1s))
            for j in range(len(d1s[i]))
        ]
    )


def dict_equiv(d1: Dict[str, int], d2: Dict[str, int], verbose: bool = False):
    if not d1.keys() == d2.keys():
        return False

    for k in d1:
        v1, v2 = d1[k], d2[k]
        if verbose:
            print(f"{k}: {type(v1)}")

        if v1 is np.ndarray:
            if not np.array_equal(v1, v2):
                return False
        elif v1 is dict:
            if not dict_equiv(v1, v2, verbose=verbose):
                return False
        elif v1 is pd.core.series.Series:
            if not v1.equals(v2):
                return False
        elif v1 is torch.Tensor:
            if not torch.equal(v1, v2):
                return False
        elif (
            v1 is models.SharedGain
            or v1 is models.SharedLatentGain
            or k.startswith("cvpca")
        ):
            pass
        elif not v1 == v2:
            return False

    return True


def compare_models(m1, m2):
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        if not torch.allclose(p1, p2):
            print(f"Mismatch in {n1}")
            print("max abs diff:", (p1 - p2).abs().max().item())


def list2ndarr(lst):
    arr = np.empty(len(lst), dtype=object)
    for i, sub in enumerate(lst):
        arr[i] = np.array(sub, dtype=object)
    return arr


def spearmanr_vec(x, y):
    # x is one vector, y is a matrix, correlate x with y[:,i]
    xr = rankdata(x)
    yr = np.vstack([rankdata(y[:, i]) for i in range(y.shape[1])]).T

    xr -= xr.mean()
    yr -= yr.mean(axis=0)

    return (xr[:, None] * yr).sum(0) / (
        np.sqrt((xr**2).sum()) * np.sqrt((yr**2).sum(0))
    )


def iterable(lst):
    if isinstance(lst, str):
        return False
    try:
        iter(lst)
    except TypeError:
        return False
    return True


def ndims_list(lst):
    if not iterable(lst):
        return 0
    return 1 + ndims_list(lst[0])
