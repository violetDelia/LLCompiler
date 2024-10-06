from llcompiler_.entrance import ExecutionEngine
import torch.fx
from typing import Any, Union, List, Dict
import ctypes
import numpy as np
from .np_to_memref import *


class ExecutionEngine:

    def __init__(self, ExecutionEngine):
        self.engine = ExecutionEngine

    def debug_info(self):
        self.engine.debug_info()

    def gen_real_call_args(self, *args):
        input_memref = [
            ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(tensor.numpy())))
            for tensor in args
        ]

    def run(self, *args) -> Any:
        print("Running")
        real_args = self.gen_real_call_args(*args)
        

    def __call__(self, *args) -> Any:
        return self.run(*args)
