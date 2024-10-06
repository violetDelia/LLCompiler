from llcompiler_.entrance import EngineInternel, Tensor
import torch.fx
from typing import Any, Union, List, Dict
import ctypes
import numpy as np


class ExecutionEngine:

    def __init__(self, ExecutionEngine):
        self.engine = ExecutionEngine

    def debug_info(self):
        self.engine.debug_info()

    def trans_to_tensor(self, *args):
        for arg in args:
            if isinstance(arg, torch.Tensor):
                print(arg)
                c = Tensor(
                )
                c.test(arg.data_ptr())

    def run(self, *args) -> Any:
        print("Running")
        self.engine.debug_info()
        inputs = self.trans_to_tensor(*args)

    def __call__(self, *args) -> Any:
        return self.run(*args)
