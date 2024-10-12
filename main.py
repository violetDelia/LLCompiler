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

# @run_time
# def compiler_onnx(model, inputs):
#     compiler = LLC.LLCompiler(
#         mode="inference", ir_tree_dir=os.path.join(os.getcwd(), "ir_tree", "onnx")
#     )
#     onnx_model = torch.onnx.export(model, inputs, dynamo=True).model_proto
#     compiler.compiler(onnx_model)
#     return model(inputs)


module_dict = {
    Add: [torch.randn((2, 2, 1, 4), device="cpu")],
    # Multi_Add: [
    #     torch.randn((2,2,4,4), device="cpu"),
    #     torch.randn((2,2,4,4), device="cpu"),
    # ],
    #Base: [torch.randn((2, 3, 224, 224), device="cpu")],
    Broadcast: [torch.randn((2, 2, 5, 5), device="cpu")],
    #torchvision.models.resnet18: [torch.randn((2, 3, 224, 224), device="cpu")],
    # torchvision.models.googlenet: torch.randn((2, 3, 224, 224), device="cpu"),
    # torchvision.models.alexnet: torch.randn((2, 3, 224, 224), device="cpu"),
    # torchvision.models.efficientnet_b0: torch.randn((2, 3, 224, 224), device="cpu"),
    # torchvision.models.vit_b_16: torch.randn((2, 3, 224, 224), device="cpu")
    # torchvision.models.convnext_tiny: torch.randn((2, 3, 224, 224), device="cpu"),
    # torchvision.models.efficientnet_b0: torch.randn((2, 3, 224, 224), device="cpu"),
}


def run_model_dict(dict):
    for func, inputs in dict.items():
        compiler = LLC.LLCompiler(
            mode="training",
            ir_tree_dir=os.path.join(os.getcwd(), "ir_tree", "fx", func.__name__),
            log_root=os.path.join(
                os.path.dirname(__file__), "ir_tree", "fx", func.__name__, "log"
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
        res = model(*inputs)

        print(res)
        print(func()(*inputs))


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
