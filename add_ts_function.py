import copy

import numpy as np
import pandas as pd

import pybind11

op = pybind11.operator()

# [ts_std_10, ts_mean_10, ts_max_10, ts_mean_20, ts_max_20, ts_min_10, ts_min_20, ts_sum_10, ts_sum_5, ts_std_20,
#                delta_1, delta_3, delta_5, delay_1, delay_3, delay_5, delay_5, delta_10, signed_power_2, ts_sum_3, ts_mean_5, ts_max_5, ts_min_5]
def ts_max(a, window):
    ret = copy.deepcopy(a)
    return op.ts_max(ret,window)

def ts_nanmean(a, window):
    ret = copy.deepcopy(a)
    return op.ts_nanmean(ret,window)

def _ts_mean_5(a):
    return ts_nanmean(a,5)

def _ts_max_5(a):
    return ts_max(a,5)

from functions import _Function

ts_mean_5 = _Function(function=_ts_mean_5, name='ts_mean_5', arity=1)
ts_max_5 = _Function(function=_ts_max_5, name='ts_max_5', arity=1)

_extra_function_map = {
    "ts_mean_5":ts_mean_5,
    "ts_max_5":ts_max_5,
}
