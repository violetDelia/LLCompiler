import torch.fx
from xdsl.context import MLContext
from xdsl.dialects.func import Func, FuncOp, Return
from xdsl.dialects.builtin import SymbolRefAttr, SymbolNameAttr, Builtin, ModuleOp
from xdsl.ir import SSAValue, BlockArgument, Region, Block
import torch.utils._pytree as pytree
from xdsl.printer import Printer
from ..dialect.llh import ConvBiasOp, LLH, WeightOp
from .onnx_translate import (
    onnx_weight_translate,
    onnx_value_translate,
    onnx_node_translate,
)
import numpy as np
from xdsl.dialects.builtin import (
    TensorType,
    i64,
    i32,
    i16,
    i1,
    f16,
    f32,
    f64,
    DYNAMIC_INDEX,
    DenseArrayBase,
    FloatAttr,
    Float64Type,
    IntegerAttr,
    BoolAttr,
    StringAttr,
    AffineMapAttr,
    SymbolNameAttr,
    SymbolRefAttr,
    UnitAttr,
)
from ..dialect.llh import TorchSymbolicIntOp
import tempfile
import numpy as np
import os
from .fx_translate import (
    torch_symbol_translate,
    torch_fake_or_mate_tensor_translate,
    torch_function_translate,
    torch_translate_to_mlir_module,
    get_result_type,
    torch_build_func,
)
from datetime import datetime
import torch
import onnx
from torch._subclasses.fake_tensor import FakeTensor
from torch._inductor.fx_passes.pre_grad import pre_grad_passes
from torch._inductor.fx_passes.post_grad import post_grad_passes
from torch._inductor.freezing import freeze
from torch._inductor.compile_fx import (
    _recursive_pre_grad_passes,
    _recursive_post_grad_passes,
)
from xdsl.irdl import IRDLOperation


class MLIR_Builder:
    def __init__(self, **kwargs) -> None:
        self.context = MLContext()
        self.context.load_dialect(Builtin)
        self.context.load_dialect(Func)
        self.context.load_dialect(LLH)
        self.kwargs = kwargs

    def mlir_gen(self, input):
        if isinstance(input, torch.fx.GraphModule):
            return self._fx_mlir_gen(input, **self.kwargs)
        if isinstance(input, onnx.GraphProto):
            return self._onnx_mlir_gen(input, **self.kwargs)
        raise NotImplementedError

    def _fx_mlir_gen(self, model: torch.fx.GraphModule, **kwargs):
        model.graph.print_tabular()
        module = torch_translate_to_mlir_module(model)
        return module

    def _onnx_mlir_gen(self, model: onnx.GraphProto, **kwargs):
        printer = Printer()
        symbol_map = dict()
        op_map = dict()
        input_types = [onnx_value_translate(value, symbol_map) for value in model.input]
        output_types = [
            onnx_value_translate(value, symbol_map) for value in model.output
        ]
        func = FuncOp("mian", (input_types, output_types))
        for weight in model.initializer:
            op = onnx_weight_translate(weight)
            op_map[weight.name] = op
            func.body.block.add_op(op)
        for node in model.node:
            op = onnx_node_translate(node, op_map, symbol_map)
            op_map[node.name] = op
            func.body.block.add_op(op)
        raise NotImplementedError
