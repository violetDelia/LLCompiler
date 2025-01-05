import torch.fx
import llcompiler.core
from typing import Any, Union, List, Dict
import sys
import os
import onnx
from torch._subclasses.fake_tensor import FakeTensor
from ..importer.fx_translate import get_result_type
import sympy.core.core
import sympy.core.facts
import sympy.core.mul
from sympy.core.numbers import Integer
import sympy.core.numbers
import sympy.core.power
from inspect import isfunction
from sympy.core.symbol import Symbol
import sympy.core
from torch.fx.passes.shape_prop import TensorMetadata


def gen_outshape_form_faketensor(exp, symbol_dict: Dict[str, int]):
    if isinstance(exp, sympy.core.Number):
        return int(exp)
    elif isinstance(exp, sympy.core.Add):
        return gen_outshape_form_faketensor(
            exp.args[1], symbol_dict
        ) + gen_outshape_form_faketensor(exp.args[0], symbol_dict)
    elif isinstance(exp, sympy.core.mul.Mul):
        return gen_outshape_form_faketensor(
            exp.args[1], symbol_dict
        ) * gen_outshape_form_faketensor(exp.args[0], symbol_dict)
    elif type(exp).__name__ == "FloorDiv":
        return gen_outshape_form_faketensor(
            exp.args[0], symbol_dict
        ) // gen_outshape_form_faketensor(exp.args[1], symbol_dict)
    elif isinstance(exp, sympy.core.power.Pow):
        assert isinstance(exp.args[1], sympy.core.numbers.Integer)
        pow = int(exp.args[1])
        base = gen_outshape_form_faketensor(exp.args[0], symbol_dict)
        while pow >= 2:
            base = base * base
            pow -= 1
        return base
    elif isinstance(exp, Symbol):
        name: str = exp.name
        return symbol_dict[name]
    else:
        raise NotImplementedError(exp, type(exp), type(exp).__name__)


# 负责根据模型以及输入分配输出
class GenOutput:
    def __init__(self):
        super(GenOutput, self).__init__()

    def get_out_call(self, model):
        if isinstance(model, torch.nn.Module):
            return self._torch_get_out_call(model)
        if isinstance(model, torch.fx.GraphModule):
            return self._fx_get_out_call(model)
        if isinstance(model, onnx.ModelProto):
            raise NotImplementedError
        if isinstance(model, onnx.GraphModule):
            raise NotImplementedError
        raise NotImplementedError

    # TODO: support move inplace to out
    def _fx_get_out_call(self, model: torch.fx.GraphModule):
        inputs_tensor_or_symbol = []
        outputs_tensor_or_symbol = []
        for node in model.graph.nodes:
            if node.op == "placeholder":
                if node.type is torch.Tensor:
                    fake_tensor = node.meta["example_value"]
                    inputs_tensor_or_symbol.append(fake_tensor)
                elif node.type is None:
                    val = node.meta["val"]
                    if isinstance(val, FakeTensor):
                        fake_tensor = node.meta["val"]
                        inputs_tensor_or_symbol.append(fake_tensor)
                    # 符号输入
                    elif isinstance(val, torch.SymInt):
                        pass
                    else:
                        print("unimplemented placeholder type: ", type(val))
                # 符号输入
                elif node.type is torch.SymInt:
                    pass
                else:
                    print("unimplemented placeholder type: ", node.type)
            elif node.op == "output":

                def trav_args(args):
                    for arg in args:
                        if isinstance(arg, tuple):
                            trav_args(arg)
                        elif isinstance(arg, list):
                            trav_args(arg)
                        elif isinstance(arg, torch.fx.node.Node):
                            outputs_tensor_or_symbol.append(get_result_type(arg))
                        elif arg is None:
                            pass
                        else:
                            raise NotImplementedError(type(arg))

                trav_args(node.args)

        def _get_out_form_inputs(*tensors):
            symbol_dict = dict()
            input_tensor_index = 0
            for tensor in tensors:
                if isinstance(tensor, torch.Tensor):
                    tensor_fake: FakeTensor = inputs_tensor_or_symbol[
                        input_tensor_index
                    ]
                    input_tensor_index += 1
                    for symbol, real_dim in zip(tensor_fake.shape, tensor.shape):
                        if isinstance(symbol, torch.SymInt):
                            if str(symbol) not in symbol_dict:
                                symbol_dict[str(symbol)] = real_dim
                elif isinstance(tensor, int):
                    pass
                elif isinstance(tensor, torch.SymInt):
                    pass
                else:
                    raise TypeError(f"Unsupported type: {type(tensor)}")
            outs = []
            for out_tensor_or_symbol in outputs_tensor_or_symbol:
                if isinstance(out_tensor_or_symbol, torch.Tensor) or isinstance(
                    out_tensor_or_symbol, TensorMetadata
                ):
                    shape = []
                    for dim in out_tensor_or_symbol.shape:
                        if isinstance(dim, int):
                            shape.append(dim)
                        elif isinstance(dim, torch.SymInt):
                            if str(dim) in symbol_dict:
                                shape.append(symbol_dict[str(dim)])
                            else:
                                shape.append(
                                    gen_outshape_form_faketensor(
                                        dim.node.expr, symbol_dict
                                    )
                                )
                        else:
                            raise TypeError(f"Unsupported type: {type(dim)}")
                    outs.append(torch.empty(shape))
                if isinstance(out_tensor_or_symbol, torch.SymInt):
                    outs.append(out_tensor_or_symbol)
            return (outs[0]) if len(outs) == 1 else outs

        return _get_out_form_inputs

    def _torch_get_out_call(self, model: torch.nn.Module):
        return self._fx_get_out_call(model)
