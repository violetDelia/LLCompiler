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

module_dict = {
    Add: [torch.randn((200, 3, 224, 224), device="cpu")],
    Div: [torch.randn((200, 3, 224, 224), device="cpu")],
    Sub: [torch.randn((200, 3, 224, 224), device="cpu")],
    Mul: [torch.randn((200, 3, 224, 224), device="cpu")],
    ElementaryArithmetic: [torch.ones((200, 3, 224, 224), device="cpu")],
    # Multi_Add: [
    #     torch.randn((2,2,4,4), device="cpu"),
    #     torch.randn((2,2,4,4), device="cpu"),
    # ],
    # Base: [torch.randn((2, 3, 224, 224), device="cpu")],
    # torchvision.models.resnet18: [torch.randn((2, 3, 224, 224), device="cpu")],
    # torchvision.models.googlenet: torch.randn((2, 3, 224, 224), device="cpu"),
    # torchvision.models.alexnet: torch.randn((2, 3, 224, 224), device="cpu"),
    # torchvision.models.efficientnet_b0: torch.randn((2, 3, 224, 224), device="cpu"),
    # torchvision.models.vit_b_16: torch.randn((2, 3, 224, 224), device="cpu")
    # torchvision.models.convnext_tiny: torch.randn((2, 3, 224, 224), device="cpu"),
    # torchvision.models.efficientnet_b0: torch.randn((2, 3, 224, 224), device="cpu"),
}


def run_model_dict(dict):
    modes = ["training", "inference"]
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
            )
            model: torch._dynamo.eval_frame.OptimizedModule = torch.compile(
                model=func(),
                backend=compiler,
                dynamic=False,
                fullgraph=True,
            )
            engine_res = llcompiler_run_time(model, *inputs)
            torch_res = torch_run_time(func(), *inputs)
            is_same = check_same(engine_res, torch_res)
            if not is_same:
                print(func.__name__, " in ", mode, " is incorrect!")


if __name__ == "__main__":
    run_model_dict(module_dict)
    # model = ElementaryArithmetic()
    # input = torch.ones(2, 2, 2, 5)
    # compiler = LLC.LLCompiler(
    #     mode="inference",
    #     symbol_infer=True,
    # )
    # opt_model: torch._dynamo.eval_frame.OptimizedModule = torch.compile(
    #     model=model,
    #     backend=compiler,
    #     dynamic=False,
    #     fullgraph=True,
    # )
    # print("llcompiler")
    # print(opt_model(input))
    # print("torch")
    # print(model(input))
