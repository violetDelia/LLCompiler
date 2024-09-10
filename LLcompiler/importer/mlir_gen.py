import torch.fx
from xdsl.context import MLContext
from xdsl.dialects.func import Func, FuncOp, Return
from xdsl.dialects.builtin import (
    Builtin,
    ModuleOp,
    Block,
    DenseIntOrFPElementsAttr,
    StringAttr,
)
from xdsl.ir import SSAValue, BlockArgument
from xdsl.ir import Region
import torch.utils._pytree as pytree
from xdsl.printer import Printer
from ..dialect.llh import ConvBiasOp, LLH, WeightOp
from .onnx.onnx_translate import (
    onnx_weight_translate,
    onnx_value_translate,
    onnx_node_translate,
)
from ..dialect.llh import SymbolicIntOp
import tempfile
import numpy as np
import os
from .fx_graph import (
    torch_symbol_translate,
    torch_fake_tensor_translate,
    torch_module_translate,
    torch_function_translate,
    torch_bind_shape,
)
from datetime import datetime
import torch
import onnx
from torch._subclasses.fake_tensor import FakeTensor


class MLIR_Builder:
    def __init__(self) -> None:
        self.context = MLContext()
        self.context.load_dialect(Builtin)
        self.context.load_dialect(Func)
        self.context.load_dialect(LLH)

    def mlir_gen(self, input, **kwargs):
        if isinstance(input, torch.fx.GraphModule):
            return self._fx_mlir_gen(input, **kwargs)
        if isinstance(input, onnx.GraphProto):
            return self._onnx_mlir_gen(input, **kwargs)
        raise NotImplementedError

    def _fx_mlir_gen(self, model: torch.fx.GraphModule, **kwargs):
        params = {
            **dict(model.named_parameters(remove_duplicate=False)),
            **dict(model.named_buffers(remove_duplicate=False)),
        }
        model.graph.print_tabular()
        value_map: dict[str, list[SSAValue]] = dict()
        symbol_map: dict[str, SymbolicIntOp] = dict()
        input_types = []
        output_types = []
        return_values = []
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
            np.save(weight_file, tensor.detach().numpy())
            op = WeightOp.build(
                result_types=[torch_fake_tensor_translate(tensor)],
                attributes={"weight_file": StringAttr(weight_file)},
            )
            value_map[name] = op.results
            block.add_op(op)
        for node in model.graph.nodes:
            if node.op == "placeholder":
                if node.type is torch.Tensor:
                    fake_tensor = node.meta["example_value"]
                    tensor_type = torch_fake_tensor_translate(fake_tensor)
                    arg_value = block.insert_arg(tensor_type, len(input_types))
                    value_map[node.name] = [arg_value]
                    input_types.append(tensor_type)
                    shape_bind = torch_bind_shape(arg_value, fake_tensor, symbol_map)
                    print(shape_bind)
                elif node.type is None:
                    val = node.meta["val"]
                    if isinstance(val, FakeTensor):
                        fake_tensor = node.meta["val"]
                        tensor_type = torch_fake_tensor_translate(fake_tensor)
                        arg_value = block.insert_arg(tensor_type, len(input_types))
                        value_map[node.name] = [arg_value]
                        input_types.append(tensor_type)
                        shape_bind = torch_bind_shape(arg_value, fake_tensor, symbol_map)
                        print(shape_bind)
                    elif isinstance(val, torch.SymInt):
                        op: SymbolicIntOp = torch_symbol_translate(
                            node.meta["val"], symbol_map
                        )
                        value_map[node.name] = [
                            block.insert_arg(op.result.type, len(input_types))
                        ]
                        input_types.append(op.result.type)
                        block.add_op(op)
                    else:
                        print("unimplemented placeholder type: ", type(val))
                elif node.type is torch.SymInt:
                    symbol: torch.SymInt = node.meta["example_value"]
                    op: SymbolicIntOp = torch_symbol_translate(symbol, symbol_map)
                    value_map[node.name] = [
                        block.insert_arg(op.result.type, len(input_types))
                    ]
                    input_types.append(op.result.type)
                    block.add_op(op)
                else:
                    print("unimplemented placeholder type: ", node.type)
            elif node.op == "call_module":
                module = model.get_submodule(node.target)
                op = torch_module_translate(node, value_map, module)
                value_map[node.name] = op.results
                block.add_op(op)
            elif node.op == "output":

                def trav_args(args):
                    for arg in args:
                        if isinstance(arg, tuple):
                            trav_args(arg)
                        elif isinstance(arg, list):
                            trav_args(arg)
                        elif isinstance(arg, torch.fx.node.Node):
                            output_types.append(value_map[arg.name][0].type)
                            return_values.append(value_map[arg.name][0])
                        else:
                            raise NotImplementedError(type(arg))

                trav_args(node.args)
            elif node.op == "call_function":
                op = torch_function_translate(node, value_map)
                value_map[node.name] = op.results
                block.add_op(op)
            else:
                raise NotImplementedError(node.op, type(node.op))
        block.add_op(Return(*return_values))
        func = FuncOp("mian", (input_types, output_types))
        func.regions[0].add_block(block)
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
        printer.print(func)
        raise NotImplementedError
