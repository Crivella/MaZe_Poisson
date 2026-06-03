"""Implement clock to time the execution of functions."""

import time
from collections import defaultdict
from functools import wraps


class ClockItem():
    def __init__(self, name: str):
        self.name: str = name
        self.tot_time: float = 0.0
        self.num_calls: int = 0
        self.sub_clocks: dict[str, ClockItem] = {}

    def report(self, all_total_time: int = None, lvl: int = 0, lvl_sep: str = '| ') -> str:
        all_total_time = all_total_time if all_total_time is not None else self.tot_time

        res = []
        name = self.name
        num_calls = self.num_calls
        tot_time = self.tot_time
        avg_time = (tot_time / num_calls * 1000) if num_calls > 0 else 0

        frc_string = ''
        if all_total_time > 0:
            frc_time = (tot_time / all_total_time * 100)
            frc_string = f'{frc_time:6.1f}% '

        name = f'{lvl_sep * lvl}{name}'

        res.append(
            f'{name:20s}   ({num_calls:>7d} CALLs):' +
            f'{frc_string}{tot_time:>13.4f} s  ({avg_time:>10.1f} ms/CALL)'
        )

        for sub_clock in self.sub_clocks.values():
            res.append(sub_clock.report(all_total_time, lvl + 1, lvl_sep))

        return '\n'.join(res)

    def report_json(self, all_total_time: int = None) -> dict:
        all_total_time = all_total_time if all_total_time is not None else self.tot_time

        res = {
            'name': self.name,
            'tot_time': self.tot_time,
            'tot_time_unit': 's',
            'num_calls': self.num_calls,
            'avg_time': (self.tot_time / self.num_calls * 1000) if self.num_calls > 0 else 0,
            'avg_time_unit': 'ms/CALL',
            'frc_time': (self.tot_time / all_total_time) if all_total_time > 0 else 0,
            'frc_time_unit': '%',
            'sub_clocks': {name: clock.report_json() for name, clock in self.sub_clocks.items()}
        }
        return res

class Clock():
    """Class to time the execution of functions.
    Usage:
    class MyClass(Clock):
        @Clock.register('clock_name')
        def my_function(self):
            pass
    """
    CLOCK_MARKER = '__clock__'

    def __init__(self, *args, **kwargs):
        self._registered: dict[str, callable] = {}
        # self.clocks = {'total': ClockItem('total')}
        self._clocks = ClockItem('TOTAL')
        self._clock_stack = 0
        self._clock_stack_mem = defaultdict(int)
        super().__init__(*args, **kwargs)

    def __getattribute__(self, name):
        res = None
        try:
            res = super().__getattribute__('_registered').get(name, None)
        except AttributeError:
            pass
        if res is not None:
            return res

        func = super().__getattribute__(name)

        # If the object is a function, marked for clocking, and not already wrapped, wrap it with a timer
        if callable(func) and hasattr(func, Clock.CLOCK_MARKER) and not hasattr(func, '__wrapped__'):
            clock_lst, lc_key = getattr(func, Clock.CLOCK_MARKER, (None, None))
            func = self._clock_method(func, clock_lst, lc_key)

        return func

    def report_clocks(self) -> str:
        res = ['Time report:', self._clocks.report()]
        return '\n'.join(res)

    def report_clocks_dct(self) -> dict:
        """Report the clock times as a dictionary for machine readability."""
        return self._clocks.report_json()

    def _clock_method(self, func, clock_lst: list[str], lc_key: str | None):
        """Helper method to wrap a function with a clock. This is used to avoid code duplication in __getattribute__."""
        if not clock_lst:
            return func
        clock_items: list[ClockItem] = [self._clocks]  # start with total clock
        ptr = self._clocks.sub_clocks
        for clock in clock_lst:
            new = ptr.setdefault(clock, ClockItem(clock))
            # ptr = self.clocks.setdefault(clock, defaultdict(int))
            clock_items.append(new)
            ptr = new.sub_clocks
        # total_clock = self.clocks['total']
        @wraps(func)
        def wrapped(*args, **kwargs):

            self._clock_stack += 1

            start = time.time()
            result = func(*args, **kwargs)
            delta = time.time() - start

            # When calling nested clock, remove the time spent in inner clocks from outer clocks
            delta_min = delta - self._clock_stack_mem.pop(self._clock_stack, 0)
            if lc_key is not None:
                setattr(self, lc_key, delta_min)
            for clock_item in clock_items:
                clock_item.tot_time += delta_min
                clock_item.num_calls += 1

            self._clock_stack -= 1

            # Accumulate time spent in this level to subtract from outer clocks
            self._clock_stack_mem[self._clock_stack] += delta
            return result
        wrapped.__wrapped__ = func
        self._registered[func.__name__] = wrapped
        return wrapped

    @staticmethod
    def register(names: str | list[str], lc_key: str | None = None):
        """Decorator to register a function to be timed by the clock.
        
        Args:
            names: The name(s) of the clock(s) to register the function to. If a list is provided, the function will be
                   registered to multiple clocks, where each one is a child of the previous one in the list.
            lc_key: If provided, the time spent in this clock's LAST_CALL will be stored in this attribute of the class
                    instance.
        """
        if isinstance(names, str):
            names = [names]
        def decorator(func):
            setattr(func, Clock.CLOCK_MARKER, (names, lc_key))
            return func
        return decorator
