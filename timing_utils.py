"""Utilities for timing functions.
"""

from time import time
import datetime
from functools import lru_cache

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from IPython import display

import qkeras


def get_time_info(filepath):
    # get the time that this file was most recently modified
    __mtime__ = int(os.path.getmtime(filepath))
    # get the current time
    __atime__ = int(time())
    # convert __atime__ and __mtime__ to a human-readable format using datetime module
    __atime__ = datetime.datetime.fromtimestamp(__atime__)
    __mtime__ = datetime.datetime.fromtimestamp(__mtime__)
    time_since_mod = __atime__ - __mtime__
    # convert time_since_mod to a human-readable format using datetime module
    time_since_mod = datetime.timedelta(seconds=time_since_mod.total_seconds())
    # get basename of file
    current_file = os.path.basename(filepath)
    print(f"File {current_file} Updated. Last modified {time_since_mod} seconds ago")


def hardware_timeit(func, *, hardware, data_shapes, max_iterations=1, max_time=None):
    assert hardware in ("/CPU:0", "/GPU:0", "/TPU:0")
    max_time = float("inf") if max_time is None else max_time

    res = []

    for shape in data_shapes:
        sub_res = {}
        iterations = 0
        total_time = 0
        total_sq_time = 0
        times = []

        while total_time < max_time and iterations < max_iterations:
            with tf.device(hardware):
                x = tf.random.normal(shape)
                start_time = time()
                _ = func(x)
            run_time = time() - start_time
            total_time += run_time
            total_sq_time += run_time**2
            iterations += 1
            times.append(run_time)

        sub_res["iterations"] = len(times)
        sub_res["avg_time"] = np.mean(times)
        sub_res["time_std"] = np.std(times)
        sub_res["median_time"] = np.median(times)
        sub_res["shape"] = str(shape)
        sub_res["size"] = np.prod(shape)
        res.append(sub_res)

    return pd.DataFrame(res)


@lru_cache()
def _quantized_bits_speed_test_data(
    alt_quantized_bits,
    *,
    hardware,
    data_shapes,
    max_iterations=1,
    max_time=None,
    quantized_bits_kwargs=None,
):
    kwargs = {
        "hardware": hardware,
        "data_shapes": data_shapes,
        "max_iterations": max_iterations,
        "max_time": max_time,
    }

    quantized_bits_kwargs = (
        {} if quantized_bits_kwargs is None else dict(quantized_bits_kwargs)
    )

    baseline_times = hardware_timeit(
        qkeras.quantized_bits(**quantized_bits_kwargs), **kwargs
    )
    alt_times = hardware_timeit(alt_quantized_bits(**quantized_bits_kwargs), **kwargs)

    return baseline_times, alt_times


def quantized_bits_speed_tests(
    *args, plot=False, error_bars=False, field="avg_time", **kwargs
):
    baseline_times, alt_times = _quantized_bits_speed_test_data(*args, **kwargs)

    if plot:
        label = {"median_time": "Median Time", "avg_time": "Average Time"}.get(
            field, field
        )
        plt.errorbar(
            np.log(baseline_times["size"]),
            baseline_times[field],
            2 * baseline_times["time_std"] if error_bars else None,
            label=f"Baseline {label}",
        )
        plt.errorbar(
            np.log(alt_times["size"]),
            alt_times[field],
            2 * alt_times["time_std"] if error_bars else None,
            label=f"New implementation {label}",
        )
        plt.xlabel("Log input size")
        plt.ylabel(f"{label} (seconds)")
        plt.title("Runtime for quantized_bits implementation")
        plt.ylim(bottom=0)
        plt.legend()
        plt.show()
    else:
        print("Current implementation Times")
        display(baseline_times)
        print("Alternative implementation Times")
        display(alt_times)

    improvement = 1 - alt_times[field] / baseline_times[field]
    print(f"\nAverage improvement: {improvement.mean():.2%}\n")


get_time_info(__file__)

if __name__ == "__main__":

    def func(x):
        return tf.linalg.svd(x)

    data_shapes = [(100, 100), (1000, 1000), (10000, 10000)]

    print(hardware_timeit(func, hardware="/CPU:0", data_shapes=data_shapes, max_time=1))
    print(hardware_timeit(func, hardware="/GPU:0", data_shapes=data_shapes, max_time=1))
