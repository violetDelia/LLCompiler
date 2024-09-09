from _collections_abc import dict_keys
import time
from typing import Any


def run_time(func):
    def inner(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        result = end_time - start_time
        print(func.__name__, " time is %.3fs" % result)
        return res

    return inner


class Dict_Registry(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, target) -> Any:
        return self.register(target)

    def register(self, key):
        def add_callable(func):
            if not callable(func):
                    raise ValueError("Value must be callable")
            if key in self.keys():
                    raise ValueError(f"Key '{key}' already registered")
            self[key] = func
        return add_callable