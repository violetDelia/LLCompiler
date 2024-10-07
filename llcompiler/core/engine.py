from llcompiler_.entrance import EngineInternel, Tensor
import torch.fx
from typing import Any, Union, List, Dict
import ctypes
import numpy as np


TORCH_DTYPE_TO_TYPE = {torch.float32: 4}


class Torch_ExecutionEngine:

    def __init__(self, ExecutionEngine):
        self.engine = ExecutionEngine

    def debug_info(self):
        self.engine.debug_info()

    def trans_to_tensor(self, *args):
        inputs = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = Tensor(
                    arg.data_ptr(),
                    arg.data_ptr(),
                    TORCH_DTYPE_TO_TYPE[arg.dtype],
                    arg.storage_offset(),
                    arg.shape,
                    arg.stride(),
                )
                inputs.append(tensor)
            else:
                raise TypeError(f"Unsupported type: {type(arg)}")
        return inputs

    def trans_to_torch(self, outs: list[Tensor]):
        i = outs[0]
        a = torch.Tensor(i.data)
        a.shape = i.size

    def run(self, *args) -> Any:
        inputs = self.trans_to_tensor(*args)
        res = [torch.as_tensor(out.to_numpy()) for out in self.engine.run(inputs)]
        return res

    def __call__(self, *args) -> Any:
        return self.run(*args)
