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


def check_same(out1: torch.Tensor, out2: torch.Tensor,eps = 1e-5):
    res = True
    if isinstance(out1, torch.Tensor) and isinstance(out2, torch.Tensor):
        return torch.allclose(out1, out2, rtol =eps,atol=eps,equal_nan = True)
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


def run_training_compiler(model, input):
    compiler = compiler = LLC.LLCompiler(mode="training", symbol_infer=True)
    opt_model: torch._dynamo.eval_frame.Optimizedmodel = torch.compile(
        model=model,
        backend=compiler,
        dynamic=False,
        fullgraph=True,
    )
    return opt_model(input)


def run_inference_compiler(model, input):
    compiler = compiler = LLC.LLCompiler(mode="inference", symbol_infer=True)
    opt_model: torch._dynamo.eval_frame.Optimizedmodel = torch.compile(
        model=model,
        backend=compiler,
        dynamic=False,
        fullgraph=True,
    )
    return opt_model(input)


def check_static_model_inference(model, input,eps = 1e-5):
    torch_out = model(input)
    compiler_out = run_inference_compiler(model, input)
    is_correct = check_same(torch_out, compiler_out,eps)
    if is_correct:
        print("static model inference are correct.")
    else:
        if not is_correct:
            print("static model inference output does not match the expected output.")
            print((torch_out - compiler_out).max())


def check_static_model_training(model, input,eps = 1e-5):
    torch_out = model(input)
    compiler_out = run_training_compiler(model, input)
    is_correct = check_same(torch_out, compiler_out,eps)
    if is_correct:
        print("static model training are correct.")
    else:
        if not is_correct:
            print("static model training output does not match the expected output.")
            print((torch_out - compiler_out).max())


def check_dynamic_model_inference(model, input,eps = 1e-5):
    torch_out = model(input)
    compiler_out = run_dynamic_inference_compiler(model, input)
    is_correct = check_same(torch_out, compiler_out,eps)
    if is_correct:
        print("dynamic model inference are correct.")
    else:
        if not is_correct:
            print("dynamic model inference output does not match the expected output.")
            print((torch_out - compiler_out).max())


def check_dynamic_model_training(model, input,eps = 1e-5):
    torch_out = model(input)
    compiler_out = run_dynamic_training_compiler(model, input)
    is_correct = check_same(torch_out, compiler_out,eps)
    if is_correct:
        print("dynamic model training are correct.")
    else:
        if not is_correct:
            print("dynamic model training output does not match the expected output.")
            print((torch_out - compiler_out).max())


def check_static_model(model, input,eps = 1e-5):
    check_static_model_inference(model, input,eps)
    check_static_model_training(model, input,eps)

def check_dynamic_model(model, input,eps = 1e-5):
    check_dynamic_model_inference(model, input,eps)
    check_dynamic_model_training(model, input,eps)

def check_model(model, input,eps = 1e-5):
    check_static_model(model, input,eps)
    check_dynamic_model(model, input,eps)
