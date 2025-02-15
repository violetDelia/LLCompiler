from llcompiler_.executor import Execution
from llcompiler_.tensor import Tensor
import torch.fx
from typing import Any, Union, List, Dict
import ctypes
import numpy as np
from time import time
from llcompiler.utility import run_time


TORCH_DTYPE_TO_TYPE = {torch.float32: 4, torch.int64: 3, torch.bool: 6}


# 负责执行c++的执行器
class ExecutionEngine:
    def __init__(self, so_file):
        self.executor = Execution()
        self.executor.load(so_file)
        self.gen_outs_call = None

    def run(self, *args) -> Any:
        pass

    def __call__(self, *args) -> Any:
        return self.run(*args)


# TODO 检测输入tensor的target是否合法，不合法转为合法的
class Torch_ExecutionEngine(ExecutionEngine):

    def __init__(self, so_file):
        super().__init__(so_file)

    def trans_to_tensor(self, *args):
        inputs = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                offset = arg.storage_offset()
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
            elif arg is None :
                pass           
            else:
                raise TypeError(f"Unsupported type: {type(arg)}")
        return inputs

    def run(self, *args) -> Any:
        # ([inputs...]) case
        if (
            isinstance(args, tuple)
            and len(args) == 1
            and (isinstance(args[0], list)) == 1
        ):
            args = args[0]
        symbols = [arg for arg in args if isinstance(arg, int)]
        inputs = self.trans_to_tensor(*args)  # 将torch.Tensor 转变为C++定义的Tensor
        outputs, multi_results = self.gen_outs_call(
            *args
        )  # 推导输出的tensor信息，并分配好内存
        outputs_ = self.trans_to_tensor(
            *outputs
        )  # 输出的torch.Tensor 转变为C++定义的Tensor
        if len(symbols) == 0:
            self.executor.run(inputs, outputs_)
        else:
            self.executor.run_with_symbols(symbols, inputs, outputs_)  # 调用执行函数
        if not multi_results:
            return outputs[0]
        return outputs
