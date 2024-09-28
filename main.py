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
def compiler_fx(model, inputs, name):
    compiler = LLC.LLCompiler(
        mode="inference", ir_tree_dir=os.path.join(os.getcwd(), "ir_tree", "fx", name)
    )
    model = torch.compile(
        model=model,
        backend=compiler,
        dynamic=True,
        fullgraph=True,
    )
    return model(inputs)


@run_time
def compiler_onnx(model, inputs):
    compiler = LLC.LLCompiler(
        mode="inference", ir_tree_dir=os.path.join(os.getcwd(), "ir_tree", "onnx")
    )
    onnx_model = torch.onnx.export(model, inputs, dynamo=True).model_proto
    compiler.compiler(onnx_model)
    return model(inputs)


@run_time
def torch_compiler(model, inputs):

    model = torch.compile(
        model=model,
        backend="inductor",
        dynamic=True,
        fullgraph=False,
    )
    return model(inputs)


module_dict = {
    Base: torch.randn((2, 3, 224, 224), device="cpu"),
    torchvision.models.resnet18: torch.randn((2, 3, 224, 224), device="cpu"),
    torchvision.models.googlenet: torch.randn((2, 3, 224, 224), device="cpu"),
    torchvision.models.alexnet: torch.randn((2, 3, 224, 224), device="cpu"),
    # torchvision.models.efficientnet_b0: torch.randn((2, 3, 224, 224), device="cpu"),
    # torchvision.models.vit_b_16: torch.randn((2, 3, 224, 224), device="cpu")
    # torchvision.models.convnext_tiny: torch.randn((2, 3, 224, 224), device="cpu"),
    # torchvision.models.efficientnet_b0: torch.randn((2, 3, 224, 224), device="cpu"),
}


def run_model_dict(dict):
    for func, inputs in dict.items():
        compiler = LLC.LLCompiler(
            mode="inference",
            ir_tree_dir=os.path.join(os.getcwd(), "ir_tree", "fx", func.__name__),
        )
        model = torch.compile(
            model=func(),
            backend=compiler,
            dynamic=True,
            fullgraph=True,
        )
        model(inputs)


if __name__ == "__main__":
    run_model_dict(module_dict)

    # model = Net()
    # model = torchvision.models.googlenet()
    # input = (torch.rand((10, 32, 512)), torch.rand((20, 32, 512)))
    # model = Net()
    # input = torch.randn((2, 3, 224, 224), device="cpu")

    # onnx_model = torch.onnx.dynamo_export(
    #     model, input, export_options=torch.onnx.ExportOptions(dynamic_shapes=True)
    # )

    # compiler_fx(model, input)

    # torch_compiler(model, input)
