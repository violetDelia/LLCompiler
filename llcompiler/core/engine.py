from llcompiler_.entrance import EngineInternel, Tensor
import torch.fx
from typing import Any, Union, List, Dict
import ctypes
import numpy as np


TORCH_DTYPE_TO_TYPE = {torch.float32: 4}


class ExecutionEngine:
    def __init__(self, ExecutionEngine):
        self.engine = ExecutionEngine
        
    def debug_info(self):
        self.engine.debug_info()
    
    def run(self, *args) -> Any:
        pass

    def __call__(self, *args) -> Any:
        return self.run(*args)
    


class Torch_ExecutionEngine(ExecutionEngine):

    def __init__(self, ExecutionEngine):
        super().__init__(ExecutionEngine)
        print("init")
    
    def trans_to_tensor(self, *args):
        print(args)
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
            elif isinstance(arg, int):
                pass
            else:
                raise TypeError(f"Unsupported type: {type(arg)}")
        return inputs

    def run(self, *args) -> Any:
        inputs = self.trans_to_tensor(*args)
        res = [torch.as_tensor(out.to_numpy()) for out in self.engine.run(inputs,[])]
        return res