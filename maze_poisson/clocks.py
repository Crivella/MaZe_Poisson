"""Implement clock to time the execution of functions."""

import time

clocks: dict[str, 'Clock'] = {}

from .loggers import Logger


class Clock(Logger):
    def __new__(cls, name: str):
        if name in clocks:
            return clocks[name]
        new_clock = super().__new__(cls)
        clocks[name] = new_clock
        return new_clock

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.cumul = 0
        self.num_calls = 0

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            self.num_calls += 1
            self.cumul += time.time() - start
            return result

        return wrapper

    def report(self):
        tot_time = self.cumul
        avg_time = tot_time / self.num_calls
        unit_tot = 's'
        unit_avg = 's'

        self.logger.info(
            f"{self.name:>20s}: {tot_time:>11.5f} {unit_tot:>2s} ({self.num_calls:>6d} calls)  Avg: {avg_time:>11.5f} {unit_avg:>2s}"
            )

    @staticmethod
    def report_all():
        for clock in clocks.values():
            clock.report()
