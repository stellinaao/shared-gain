from typing import Dict
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

from lib import data, models
import srlvm

import numpy as np

def save_var(var_name, tag, data):
    fname = f'vars/{var_name}-{tag}.npy'
    if type(data) == dict:
        save_jagged(var_name, tag, data) # redirect to the correct function
    else:
        np.save(fname, data)

def load_var(var_name, tag):
    fname = f'vars/{var_name}-{tag}.npy'
    
    return np.load(fname)

def save_jagged(var_name, tag, data):
    fname = f'vars/{var_name}-{tag}.npz'
    if type(data) == dict:
        np.savez(fname, **data)
    else:
        np.savez(fname, *data)

def load_jagged(var_name, tag):
    fname = f'vars/{var_name}-{tag}.npz'

    load = np.load(fname)
    data = [load[key] for key in load]
    load.close()

    return data

def list_dict_equiv(d1s, d2s, verbose=False):
    if len(d1s) != len(d2s): return False
    
    if not all([len(d1s[i])==len(d2s[i]) for i in range(len(d1s))]): return False
    
    return all([dict_equiv(d1s[i][j], d2s[i][j], verbose=verbose) for i in range(len(d1s)) for j in range(len(d1s[i]))])
    

def dict_equiv(d1: Dict[str, int], d2: Dict[str, int], verbose: bool=False):
    if not d1.keys() == d2.keys(): return False
    
    for k in d1:
        v1, v2 = d1[k], d2[k]
        if verbose: print(f"{k}: {type(v1)}")
        
        if    type(v1) == np.ndarray:
            if not np.array_equal(v1, v2): return False
        elif  type(v1) == dict:
            if not dict_equiv(v1, v2, verbose=verbose): return False
        elif  type(v1) == pd.core.series.Series:
            if not v1.equals(v2): return False
        elif  type(v1) == torch.Tensor:
            if not torch.equal(v1,v2): return False
        elif  type(v1) == models.SharedGain or type(v1) == models.SharedLatentGain or k.startswith('cvpca'):
            pass
        elif not v1 == v2: return False
    
    return True

def list2ndarr(l):
    arr = np.empty(len(l), dtype=object)
    for i, sub in enumerate(l):
        arr[i] = np.array(sub, dtype=object)
    return arr

def iterable(l):
    if isinstance(l, str): return False
    try:
        iter(l)
    except TypeError:
        return False
    return True

def ndims_list(l):
    if not iterable(l): return 0
    return 1+ndims_list(l[0])