from __future__ import annotations

import concurrent.futures
import loky
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import warnings

from loky import get_reusable_executor

from typing import Union, Callable, Iterable, Sized, Any
import tqdm
import os

CPU_COUNT = os.cpu_count()


class TaskTimeoutException(Exception):
    "Task timeout exception"

    def __init__(self, idx=None):
        self.idx = idx

class TaskTimeoutWarning(Warning):
    "Task timeout Warning"
    pass

def map_thread(func: Callable,
               tasks: Union[Iterable, Sized],
               max_workers: int = CPU_COUNT,
               progress_bar: bool = True,
               executor_kwargs: dict = {},
               map_kwargs: dict = {}
               ):
    """
    use the original map function of ThreadPoolExecutor, added progress bar
    :param func: function to be mapped
    :param tasks: tasks to be mapped
    "param executor_kwargs: key arguments passed to the excutor
    """
    with ThreadPoolExecutor(max_workers=max_workers, **executor_kwargs) as executor:
        # check if tasks is sized object, progress bar is only available for sized object
        if progress_bar and hasattr(tasks, '__len__'):
            mapped_values = list(tqdm.tqdm(executor.map(func, tasks, **map_kwargs), total=len(tasks)))
        else:
            mapped_values = executor.map(func, tasks, **map_kwargs)
    return mapped_values


def map_process_raw(func: Callable,
                    tasks: Union[Iterable, Sized],
                    max_workers: int = CPU_COUNT,
                    progress_bar: bool = True,
                    executor_kwargs: dict = {},
                    map_kwargs: dict = {}
                    ):
    """
    use the original map function of ProcessPoolExecutor, unavailable in jupyter notebook or ipython
    """
    with ProcessPoolExecutor(max_workers=max_workers, **executor_kwargs) as executor:
        if progress_bar and hasattr(tasks, '__len__'):
            mapped_values = list(tqdm.tqdm(executor.map(func, tasks, **map_kwargs), total=len(tasks)))
        else:
            mapped_values = executor.map(func, tasks, **map_kwargs)
    return mapped_values


def map_loky_raw(func: Callable,
                 tasks: Union[Iterable, Sized],
                 max_workers: int = CPU_COUNT,
                 progress_bar: bool = True,
                 **kwargs):
    """
    use the map function of joblib/loky, with robust multiprocessing and jupyter support
    """
    with get_reusable_executor(max_workers=max_workers, **kwargs) as executor:
        if progress_bar and hasattr(tasks, '__len__'):
            mapped_values = list(tqdm.tqdm(executor.map(func, tasks), total=len(tasks)))
        else:
            mapped_values = executor.map(func, tasks)
    return mapped_values


def map_loky(func: Callable,
             tasks: Iterable,
             max_workers: int = CPU_COUNT,
             progress_bar: bool = True,
             timeout: Union[float, None] = None,
             timeout_replacer: Any = None,
             **kwargs):
    """
    Based on map_loky_raw, adding task-wise timeout support.
    If the executing time of single task exceeds timeout, then the task will be aborted
        and the result will be replaced by timeout_replacer (defult None)

    :param timeout: timeout value for each task
    :param timeout_replacer: if a task times out, then the result of this task will be replaced by timeout_replacer
    :param kwargs: key arguments for loky multiprocessing executor
    :return: a list of results
    """

    def _abortable_task(item):
        """
        Run the real task in a sub thread with the main thread counting time and
        aborting the timed out task by raising TimeoutError.
        :param item: item[0] is the index of task, item[1] is the arg of the task
        :return:
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, item[1])
            try:
                res = future.result(timeout=timeout)
                return res
            except concurrent.futures.TimeoutError:
                message=f"future {item[0]} aborted due to timeout"
                warnings.warn(message,TaskTimeoutWarning)
                raise TaskTimeoutException

    def _collect_result(future):
        """
        Collect the result of the future, replace the result of timed out tasks by {replacer} arg
        :param future:
        :return:
        """
        try:
            return future.result()
        except TaskTimeoutException:
            return timeout_replacer

    with get_reusable_executor(max_workers=max_workers, **kwargs) as executor:
        futures = list(executor.submit(_abortable_task, (idx, task)) for idx, task in enumerate(tasks))
        if progress_bar:
            results = list(tqdm.tqdm(map(_collect_result, futures), total=len(futures)))
        else:
            results = list(map(_collect_result, executor.map(_abortable_task, tasks)))
    return results

def imap_loky(func: Callable,
              tasks: Iterable,
              max_workers: int = CPU_COUNT,
              progress_bar: bool = True,
              timeout: Union[float, None] = None,
              timeout_replacer: Any = None,
              sorted=True,
              index=False,
              **kwargs):
    """
    Based on imap instead map, also added timeout support.
    Can choose whether to sort the results by index, and whether to return the index of the result.
    :param timeout: timeout value for each task
    :param timeout_replacer: if a task times out, then the result of this task will be replaced by timeout_replacer
    :param kwargs: key arguments for executor
    :return:
    """

    def _abortable_task(item):
        """
        Run the real processing function in a sub thread so the main thread can
        abort timed out task by raising TimeoutError.
        :param item: item[0] is the index, item[1] is the arg of the task
        :return:
        """
        idx = item[0]
        task = item[1]
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, task)
            try:
                res = future.result(timeout=timeout)
                return idx, res
            except concurrent.futures.TimeoutError:
                message = f"future {item[0]} aborted due to timeout"
                warnings.warn(message,TaskTimeoutWarning)
                ## pass the index of task when raise error so that we can find the correct position in results
                raise TaskTimeoutException(idx)

    def _collect_result(future):
        """
        Collect the result of the future, replace the result of timed out tasks by %replacer% arg.
        :param future:
        :return:
        """
        try:
            return future.result()
        except TaskTimeoutException as e:
            return e.idx, timeout_replacer

    with get_reusable_executor(max_workers=max_workers, **kwargs) as executor:
        futures = list(executor.submit(_abortable_task, (idx, task)) for idx, task in enumerate(tasks))
        if progress_bar:
            results = list(tqdm.tqdm(map(_collect_result, loky.as_completed(futures)), total=len(futures)))
        else:
            results = list(map(_collect_result, executor.map(_abortable_task, loky.as_completed(futures))))
    if sorted:
        results.sort(key=lambda x: x[0])    ## sort by task index
    if index:
        return [x[0] for x in results], [x[1] for x in results]
    else:
        return [x[1] for x in results]

def starmap_loky(func: Callable,
                 tasks: Union[Iterable,Sized],
                 max_workers: int = CPU_COUNT,
                 progress_bar: bool = True,
                 timeout: Union[float, None] = None,
                 timeout_replacer: Any = None,
                 **kwargs):
    """
    Based on map_loky, acts like starmap in itertools
    """
    def _func_wrapper(task:Iterable):
        return func(*task)

    return map_loky(func=_func_wrapper,
                    tasks=tasks,
                    max_workers=max_workers,
                    progress_bar=progress_bar,
                    timeout=timeout,
                    timeout_replacer=timeout_replacer,
                    **kwargs)


def istarmap_loky(func: Callable,
                  tasks: Union[Iterable,Sized],
                  max_workers: int = CPU_COUNT,
                  progress_bar: bool = True,
                  timeout: Union[float, None] = None,
                  timeout_replacer: Any = None,
                  sorted=True,
                  index=False,
                  **kwargs):
    """
    Like starmap_loky, but is based on imap instead of map
    """
    def _func_wrapper(task:Iterable):
        return func(*task)

    return imap_loky(func=_func_wrapper,
                    tasks=tasks,
                    max_workers=max_workers,
                    progress_bar=progress_bar,
                    timeout=timeout,
                    timeout_replacer=timeout_replacer,
                    sorted=sorted,
                    index=index,
                    **kwargs)
