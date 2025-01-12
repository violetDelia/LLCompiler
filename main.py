import llcompiler.compiler as LLC
import os.path
import subprocess
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo.backends.common import aot_autograd
import torch.fx
from llcompiler.utility import run_time
import onnx
import torchgen
import torch._dynamo
import os
from llcompiler.test_models import *

torch._dynamo.config.suppress_errors = True
torch.nn.Transformer
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import typing


@run_time
def llcompiler_run_time(model, *inputs):
    return model(*inputs)


@run_time
def torch_run_time(model, *inputs):
    return model(*inputs)


@run_time
def loop_torch_run_time(loop_times, model, *inputs):
    for _ in range(0, loop_times):
        model(*inputs)


@run_time
def loop_llcompiler_run_time(loop_times, model, *inputs):
    for _ in range(0, loop_times):
        model(*inputs)


@run_time
def loop_torch_compiler_run_time(loop_times, model, *inputs):
    for _ in range(0, loop_times):
        model(*inputs)


@run_time
def torch_compiler_time(model, *inputs):
    return model(*inputs)


loop_num = 10
loop = False

module_dict = {
    Add: [torch.randn((1,2,2), device="cpu")],
    # Slice: [torch.randn((200, 200, 224, 224), device="cpu")],
    # Conv2D_NCHW_FCHW :[torch.randn((1, 3, 3,3), device="cpu")],
    # MaxPool2D: [torch.randn((3,3,224,224), device="cpu")],
    # MultiHeadedAttention: [
    #     torch.randn((2, 24, 8), device="cpu"),
    #     torch.randn((2, 24, 8), device="cpu"),
    #     torch.randn((2, 24, 8), device="cpu"),
    #     torch.tril(torch.ones((24, 24)), diagonal=0).unsqueeze(0),
    # ]
    # Resnet: [torch.randn((1, 3, 64, 64), device="cpu")]
    # BatchNorm2D_Inference: [torch.ones(1, 3, 2, 2, device="cpu")],
    # Linear: [torch.randn((10, 100000), device="cpu")],
    # ElewiseFusion1: [torch.randn((2,2), device="cpu")],
    # Braodcast: [torch.randn((10, 20), device="cpu")],
    # Matmul: [torch.randn((1,3,3), device="cpu")],
    # Sqrt: [torch.randn((3,3,224,224), device="cpu")],
}


def run_model_dict(dict):
    modes = [
        "inference",
        # "training"
    ]
    for mode in modes:
        for func, inputs in dict.items():
            print("模型: ", func.__name__, ", 模式: ", mode)
            compiler = LLC.LLCompiler(
                mode=mode,
                ir_tree_dir=os.path.join(
                    os.getcwd(), "ir_tree", mode, "fx", func.__name__
                ),
                log_root=os.path.join(
                    os.path.dirname(__file__),
                    "ir_tree",
                    mode,
                    "fx",
                    func.__name__,
                    "log",
                ),
                log_level="debug",
                symbol_infer=True,
                target_layout="NCHW",
                pipeline="transform",
            )
            model: nn.Module = func()
            if mode == "inference":
                model.train(False)
            opt_model: torch._dynamo.eval_frame.OptimizedModule = torch.compile(
                model=model,
                backend=compiler,
                dynamic=False,
                fullgraph=False,
            )
            torch_compiler: torch._dynamo.eval_frame.OptimizedModule = torch.compile(
                model=model,
                dynamic=True,
                fullgraph=False,
            )

            torch_res = torch_run_time(model, *inputs)
            torch_run_time(model, *inputs)
            torch_compiler_time(torch_compiler, *inputs)
            torch_compiler_time(torch_compiler, *inputs)
            engine_res = llcompiler_run_time(opt_model, *inputs)
            llcompiler_run_time(opt_model, *inputs)
            if loop:
                loop_torch_run_time(loop_num, model, *inputs)
                loop_torch_compiler_run_time(loop_num, torch_compiler, *inputs)
                loop_llcompiler_run_time(loop_num, opt_model, *inputs)
            # print(engine_res)
            # print(torch_res)
            is_same = check_same(engine_res, torch_res)
            if not is_same:
                print(func.__name__, " in ", mode, " is incorrect!")
                print((engine_res - torch_res).max())


if __name__ == "__main__":
    run_model_dict(module_dict)
