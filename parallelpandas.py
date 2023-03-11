from __future__ import annotations
from typing import Union, Callable

import pandas as pd
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
from pandas.core.window.rolling import Rolling, RollingGroupby
import numpy as np
import parallelmap
from tqdm import tqdm


class SizedRolling:
    def __init__(self, rolling: Union[Rolling, RollingGroupby]):
        self.size = len(rolling.obj)
        self.rolling = rolling
        self.index = rolling.obj.index
        if rolling.min_periods is None:
            self.min_periods = rolling.window
        else:
            self.min_periods = rolling.min_periods

    def __iter__(self):
        return self.rolling.__iter__()

    def __len__(self):
        return self.size
    ## write codes to let the iterator reset automately:


def parallel_groupby_apply(grouped: Union[DataFrameGroupBy, SeriesGroupBy],
                           func: Callable,
                           **kwargs):
    """
    This function is used to parallelize the calculation of groupby apply
    """
    result = pd.concat(parallelmap.map_loky_raw(func, [group for name, group in grouped], **kwargs))
    return result


def rolling_apply(rolling: Union[Rolling, RollingGroupby],
                  func: Callable,
                  progress_bar=True,
                  ):
    """
    This function is used to rolling-apply the DataFrame, support multi-columns
    """
    rolling = SizedRolling(rolling)
    if progress_bar:
        result = list(tqdm(map(func, rolling), total=len(rolling)))
    else:
        result = list(map(func, rolling))

    ## check the type of result & concat the result
    if isinstance(result[0], pd.Series):
        result = pd.concat(result, axis=1).T
        result.index = rolling.index
    elif isinstance(result[0], pd.DataFrame):
        result = pd.concat(result)
        result.index = rolling.index
    else:
        result = pd.Series(result, index=rolling.index)
    result.iloc[:rolling.min_periods - 1] = np.nan
    return result


def parallel_rolling_apply(rolling: Union[Rolling, RollingGroupby],
                           func: Callable,
                           **kwargs):
    """
    This function is used to parallelize the calculation of rolling apply, support multi-columns
    """
    rolling = SizedRolling(rolling)
    result = parallelmap.map_loky_raw(func, rolling, **kwargs)

    if isinstance(result[0], pd.Series):
        result = pd.concat(result, axis=1).T
        result.index = rolling.index
    elif isinstance(result[0], pd.DataFrame):
        result = pd.concat(result)
        result.index = rolling.index
    else:
        result = pd.Series(result, index=rolling.index)
    result.iloc[:rolling.min_periods - 1] = np.nan
    return result
