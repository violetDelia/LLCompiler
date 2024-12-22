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
    torch_module_translate,
    torch_function_translate,
    torch_symbol_bind,
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

TORCH_DTYPE_TO_NUMPY_DTYPE = {
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.bool: bool,
}


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
        params: dict[str, torch.Tensor] = {
            **dict(model.named_parameters(remove_duplicate=False)),
            **dict(model.named_buffers(remove_duplicate=False)),
        }
        value_map: dict[str, list[SSAValue]] = dict()
        symbol_map: dict[str, TorchSymbolicIntOp] = dict()
        block = Block()
        weight_dir = os.path.join(
            os.path.dirname(__file__),
            "LLcompiler_weight_temp",
            datetime.now().astimezone().isoformat(),
        )
        os.makedirs(weight_dir)
        for name, tensor in params.items():
            weight_file = os.path.join(
                weight_dir,
                name + ".npy",
            )
            np.save(
                weight_file,
                np.array(tensor.tolist(), TORCH_DTYPE_TO_NUMPY_DTYPE[tensor.dtype]),
            )
            op = WeightOp.build(
                result_types=[torch_fake_or_mate_tensor_translate(tensor)],
                attributes={"weight_file": StringAttr(weight_file)},
            )
            value_map[name] = op.results
            block.add_op(op)
        func = torch_build_func(model.graph, "main", block, value_map, symbol_map)
        func.attributes.update({"entrance": UnitAttr()})
        module = ModuleOp([func])
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
