import torch.fx
from xdsl.context import MLContext
from xdsl.dialects.func import Func, FuncOp
from xdsl.dialects.builtin import Builtin, ModuleOp, Block, DenseIntOrFPElementsAttr
from xdsl.ir import Region
from ..dialect.llh import *
import torch.utils._pytree as pytree
from xdsl.printer import Printer
from .translate import *
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
        if isinstance(input,onnx.GraphProto):
            return self._onnx_mlir_gen(input, **kwargs)
        raise NotImplementedError

    def _fx_mlir_gen(self, model: torch.fx.GraphModule, **kwargs):
        printer = Printer()
        params = {
            **dict(model.named_parameters(remove_duplicate=False)),
            **dict(model.named_buffers(remove_duplicate=False)),
        }
        model.graph.print_tabular()
        value_map = dict()
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
                result_types=[torch_tensor_translate(tensor)],
                attributes={"weight_file": StringAttr(weight_file)},
            )
            value_map[name] = op.result
            block.add_op(op)
        input_types = []
        output_types = []
        for node in model.graph.nodes:
            if node.op == "placeholder" :
                if node.type is torch.Tensor:
                    args = torch_tensor_translate(node.meta["example_value"])
                    value_map[node.name] = block.insert_arg(args, len(input_types))
                    input_types.append(args)
            if node.op == "call_module":
                op = torch_module_translate(node,value_map)
                
            # if node.op == "output":
            #     print(node)
            #     print(node.meta)
            #     print(node.op)
            #     print(node.name)
            #     print(node.type)
            #     print(node.args)
            #     print(node.kwargs)
            #     print(node.target)
            #     args = torch_tensor_translate(node.meta["example_value"])
            #     printer.print(args)
            #     output_type.append(args)
        func = FuncOp("mian", (input_types, output_types))
        func.regions[0].add_block(block)
        # print(model.graph.print_tabular())

        # for node in model.graph.nodes:
        #     if node.op == "placeholder":
        #         print(node)
        #         print(node.meta)
        #         print(node.op)
        #         print(node.name)
        #         print(node.type)
        #         print(node.args)
        #         print(node.kwargs)
        #         print(node.target)
        module = ModuleOp([func])

        printer.print(module)
        raise NotImplementedError

    
    def _onnx_mlir_gen(self, model: onnx.GraphProto, **kwargs):
        printer = Printer()
        symbol_map = dict()
        op_map = dict()
        input_types = [onnx_value_translate(value,symbol_map) for value in model.input]
        output_types = [onnx_value_translate(value,symbol_map) for value in model.output]
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