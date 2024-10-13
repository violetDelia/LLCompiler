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


def check_same(out1: torch.Tensor, out2: torch.Tensor):
    res = True
    if isinstance(out1, torch.Tensor) and isinstance(out2, torch.Tensor):
        return torch.allclose(out1, out2,atol=1e-5)
    if isinstance(out1, torch.Tensor) or isinstance(out2, torch.Tensor):
        raise ValueError
    for out_lhs, out_rhs in zip(out1, out2):
        if res == False:
            return False
        res = check_same(out_lhs, out_rhs)
    return res


def run_dynamic_training_compiler(model, input):
    compiler = compiler = LLC.LLCompiler(mode="training", symbol_infer=True)
    opt_model: torch._dynamo.eval_frame.Optimizedmodel = torch.compile(
        model=model,
        backend=compiler,
        dynamic=True,
        fullgraph=True,
    )
    return opt_model(input)


def run_dynamic_inference_compiler(model, input):
    compiler = compiler = LLC.LLCompiler(mode="inference", symbol_infer=True)
    opt_model: torch._dynamo.eval_frame.Optimizedmodel = torch.compile(
        model=model,
        backend=compiler,
        dynamic=True,
        fullgraph=True,
    )
    return opt_model(input)


def check_model(model, input):
    torch_out = model(input)
    train_out = run_dynamic_training_compiler(model, input)
    infer_out = run_dynamic_training_compiler(model, input)
    train_is_correct = check_same(torch_out, train_out)
    infer_is_correct = check_same(torch_out, infer_out)
    if train_is_correct and infer_is_correct:
        print("The calculations are correct.")
    else:
        if not infer_is_correct:
            print("The inference output does not match the expected output.")
            print(torch_out - infer_out)
        if not train_is_correct:
            print("The training output does not match the expected output.")
            print(torch_out - train_out)
