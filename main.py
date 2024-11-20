import llcompiler.compiler as LLC
import os.path
import subprocess
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo.backends.common import aot_autograd
import torch.fx
from llcompiler.core.utility import run_time
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
def loop_llcompiler_run_time(loop_times, model, *inputs):
    for _ in range(0, loop_times):
        model(*inputs)


@run_time
def torch_compiler_time(model, *inputs):
    return model(*inputs)


module_dict = {
    Add: [torch.randn((200, 3, 224, 256), device="cpu")],
    # Div: [torch.randn((200, 3, 224, 224), device="cpu")],
    # Sub: [torch.randn((200, 3, 224, 224), device="cpu")],
    # Mul: [torch.randn((200, 3, 224, 224), device="cpu")],
    # Abs: [torch.randn((200,3,224,256), device="cpu")],
    # Extract: [torch.randn((200,3,224,256), device="cpu")],
    # Unsqueeze: [torch.randn((200,3,224,256), device="cpu")],
    # MultiHeadedAttention: [
    #     torch.randn((2, 24, 8), device="cpu"),
    #     torch.randn((2, 24, 8), device="cpu"),
    #     torch.randn((2, 24, 8), device="cpu"),
    #     torch.tril(torch.ones((24, 24)), diagonal=0).unsqueeze(0),
    # ]
    # ElementaryArithmetic: [torch.ones((200, 3, 224, 224), device="cpu")],
    # Relu :[torch.randn((200, 3, 224, 224), device="cpu")],
    #Conv2D_NCHW_FCHW :[torch.randn((200, 3, 224,224), device="cpu")],
    # BatchNorm2D_Inference: [torch.randn(200, 3, 224, 224, device="cpu")],
    # Linear: [torch.randn((10,100000), device="cpu")],
    # MaxPool2D: [torch.randn((3,3,224,224), device="cpu")],
    # Resnet: [torch.randn((1, 3, 64, 64), device="cpu")],
    # ElewiseFusion1: [torch.randn((200, 3, 224, 224), device="cpu")],
    # torchvision.models.googlenet: [torch.randn((2, 3, 224, 224), device="cpu")],
    # torchvision.models.alexnet: torch.randn((2, 3, 224, 224), device="cpu"),
    # torchvision.models.efficientnet_b0: torch.randn((2, 3, 224, 224), device="cpu"),
    # torchvision.models.vit_b_16: torch.randn((2, 3, 224, 224), device="cpu")
    # torchvision.models.convnext_tiny: torch.randn((2, 3, 224, 224), device="cpu"),
    # torchvision.models.efficientnet_b0: torch.randn((2, 3, 224, 224), device="cpu"),
}


def run_model_dict(dict):
    modes = ["inference", "training"]
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
                target_layout="NHWC",
                pipeline="transform",
            )
            model = func()
            model.traning = True
            opt_model: torch._dynamo.eval_frame.OptimizedModule = torch.compile(
                model=model,
                backend=compiler,
                dynamic=True,
                fullgraph=True,
            )
            torch_compiler: torch._dynamo.eval_frame.OptimizedModule = torch.compile(
                model=model,
                dynamic=True,
                fullgraph=True,
            )
            torch_res = torch_run_time(model, *inputs)
            torch_compiler_time(torch_compiler, *inputs)
            torch_compiler_time(torch_compiler, *inputs)
            engine_res = llcompiler_run_time(opt_model, *inputs)
            llcompiler_run_time(opt_model, *inputs)
            # loop_llcompiler_run_time(100,opt_model, *inputs)
            is_same = check_same(engine_res, torch_res)
            if not is_same:
                print(func.__name__, " in ", mode, " is incorrect!")
                print((engine_res - torch_res))


if __name__ == "__main__":
    run_model_dict(module_dict)
