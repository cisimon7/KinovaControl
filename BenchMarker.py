from time import perf_counter
from typing import Callable, Any


def measure_time_seconds(function: Callable[[], Any]):
    start_time = perf_counter()
    result = function()
    end_time = perf_counter()

    return round(end_time - start_time, 4), result


def time_it(function: Callable[[], Any], rnd=4):
    start_time = perf_counter()
    result = function()
    end_time = perf_counter()

    return round(end_time - start_time, rnd), result
