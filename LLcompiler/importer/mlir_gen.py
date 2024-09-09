import torch.fx
from xdsl.context import MLContext
from xdsl.dialects.func import Func, FuncOp, Return
from xdsl.dialects.builtin import Builtin, ModuleOp, Block, DenseIntOrFPElementsAttr
from xdsl.ir import Region
from ..dialect.llh import *
import torch.utils._pytree as pytree
from xdsl.printer import Printer
from .fx_translate import *
from .onnx_translate import *
import tempfile
import numpy as np
import os
from datetime import datetime
import torch
import onnx


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
        printer = Printer()
        params = {
            **dict(model.named_parameters(remove_duplicate=False)),
            **dict(model.named_buffers(remove_duplicate=False)),
        }
        model.graph.print_tabular()
        value_map: dict[str, list[SSAValue]] = dict()
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
        input_types = []
        output_types = []
        return_values = []
        for node in model.graph.nodes:
            if node.op == "placeholder":
                if node.type is torch.Tensor:
                    args = torch_fake_tensor_translate(node.meta["example_value"])
                    value_map[node.name] = [block.insert_arg(args, len(input_types))]
                    input_types.append(args)
                if node.type is None:
                    print(node.meta)
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
                        elif isinstance(arg, torch.fx.node.Node):
                            output_types.append(value_map[arg.name][0].type)
                            return_values.append(value_map[arg.name][0])
                        else:
                            raise NotImplementedError(type(arg))

                trav_args(node.args)
            else:
                raise NotImplementedError(node.op)
        block.add_op(Return(*return_values))
        func = FuncOp("mian", (input_types, output_types))
        func.regions[0].add_block(block)
        module = ModuleOp([func])
        printer.print(module)
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
