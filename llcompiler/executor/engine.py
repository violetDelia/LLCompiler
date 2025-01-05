from llcompiler_.entrance import EngineInternel, Tensor
import torch.fx
from typing import Any, Union, List, Dict
import ctypes
import numpy as np
from time import time
from llcompiler.utility import run_time


TORCH_DTYPE_TO_TYPE = {torch.float32: 4, torch.int64: 3, torch.bool: 6}


# 负责执行c++的执行器
class ExecutionEngine:
    def __init__(self, ExecutionEngine):
        self.engine = ExecutionEngine
        self.gen_outs_call = None

    def debug_info(self):
        self.engine.debug_info()

    def run(self, *args) -> Any:
        pass

    def __call__(self, *args) -> Any:
        return self.run(*args)


# TODO 检测输入tensor的target是否合法，不合法转为合法的
class Torch_ExecutionEngine(ExecutionEngine):

    def __init__(self, ExecutionEngine):
        super().__init__(ExecutionEngine)

    def trans_to_tensor(self, *args):
        inputs = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                offset = arg.storage_offset()
                print(arg.data_ptr())
                tensor = Tensor(
                    arg.data_ptr(),
                    arg.data_ptr() + offset,
                    TORCH_DTYPE_TO_TYPE[arg.dtype],
                    offset,
                    arg.shape,
                    arg.stride(),
                )
                inputs.append(tensor)
            elif isinstance(arg, int):
                pass
            elif isinstance(arg, torch.SymInt):
                pass
            else:
                raise TypeError(f"Unsupported type: {type(arg)}")
        return inputs

    def run(self, *args) -> Any:
        # ([inputs...]) case
        print(args)
        if (
            isinstance(args, tuple)
            and len(args) == 1
            and (isinstance(args[0], list)) == 1
        ):
            args = args[0]
        inputs = self.trans_to_tensor(*args)  # 将torch.Tensor 转变为C++定义的Tensor
        outputs = self.gen_outs_call(*args)  # 推导输出的tensor信息，并分配好内存
        outputs_ = self.trans_to_tensor(
            *outputs
        )  # 输出的torch.Tensor 转变为C++定义的Tensor
        self.engine.run(inputs, outputs_)  # 调用执行函数
        return outputs
