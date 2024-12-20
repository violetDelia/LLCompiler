import sympy.core.core
import sympy.core.facts
import sympy.core.mul
from sympy.core.numbers import Integer
import sympy.core.numbers
import sympy.core.power
from inspect import isfunction
from sympy.core.symbol import Symbol
import sympy.core
import torch._ops as op
import torch.fx.experimental
import torch.fx.experimental.sym_node
from xdsl.ir import SSAValue, Block, Operation, Mapping, Attribute, Region
from xdsl.ir.affine.affine_expr import (
    AffineSymExpr,
    AffineConstantExpr,
    AffineBinaryOpExpr,
    AffineBinaryOpKind,
)
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
    DenseResourceAttr,
    FloatAttr,
    Float64Type,
    IntegerAttr,
    BoolAttr,
    StringAttr,
    AffineMapAttr,
    SymbolNameAttr,
    SymbolRefAttr,
    DictionaryAttr,
    ArrayAttr,
)
from xdsl.ir.affine.affine_map import AffineMap
from ..dialect.llh_utility import build_llh_constant, build_llh_scalar_tensor
from ..core.utility import Dict_Registry
from datetime import datetime
import torch.nn
from torch.fx.passes.shape_prop import TensorMetadata
from torch._subclasses.fake_tensor import FakeTensor
from ..core.utility import run_time
import os
import numpy as np
import torch.fx
from torch.fx.experimental.sym_node import SymNode
from ..dialect.llh import TorchSymbolicIntOp, SymbolicBindOp, ConstantOp
from xdsl.dialects.func import Return, FuncOp
import sympy


def torch_symbol_translate(
    symbol: torch.SymInt, symbol_map: dict[str, TorchSymbolicIntOp]
):
    name: str = symbol.node.expr.name
    atts = {"sym_name": StringAttr(name)}
    op = TorchSymbolicIntOp(attributes=atts, result_types=[i64])
    symbol_map[name] = op
    return op


def _generate_affine_symbolic(
    arg,
    symbol_map: dict[str, TorchSymbolicIntOp],
    affine_expr_map: dict[str, AffineSymExpr],
    bind_symbols: list[SSAValue],
):
    if isinstance(arg, sympy.core.Number):
        return AffineConstantExpr(int(arg))
    elif isinstance(arg, sympy.core.Add):
        return AffineBinaryOpExpr(
            AffineBinaryOpKind.Add,
            _generate_affine_symbolic(
                arg.args[1], symbol_map, affine_expr_map, bind_symbols
            ),
            _generate_affine_symbolic(
                arg.args[0], symbol_map, affine_expr_map, bind_symbols
            ),
        )
    elif isinstance(arg, sympy.core.mul.Mul):

        return AffineBinaryOpExpr(
            AffineBinaryOpKind.Mul,
            _generate_affine_symbolic(
                arg.args[1], symbol_map, affine_expr_map, bind_symbols
            ),
            _generate_affine_symbolic(
                arg.args[0], symbol_map, affine_expr_map, bind_symbols
            ),
        )
    elif type(arg).__name__ == "FloorDiv":
        return AffineBinaryOpExpr(
            AffineBinaryOpKind.FloorDiv,
            _generate_affine_symbolic(
                arg.args[0], symbol_map, affine_expr_map, bind_symbols
            ),
            _generate_affine_symbolic(
                arg.args[1], symbol_map, affine_expr_map, bind_symbols
            ),
        )
    elif isinstance(arg, sympy.core.power.Pow):
        assert isinstance(arg.args[1], sympy.core.numbers.Integer)
        pow = int(arg.args[1])
        base = _generate_affine_symbolic(
            arg.args[0], symbol_map, affine_expr_map, bind_symbols
        )
        while pow >= 2:
            base = AffineBinaryOpExpr(AffineBinaryOpKind.Mul, base, base)
            pow -= 1
        return base

    elif isinstance(arg, Symbol):
        name: str = arg.name
        if name in symbol_map:
            if name not in affine_expr_map:
                affine_expr = AffineSymExpr(len(bind_symbols))
                affine_expr_map[name] = affine_expr
                bind_symbols.append(symbol_map[name].result)
            return affine_expr_map[name]
        else:
            raise NotImplementedError("name must in symbol map")

    else:
        raise NotImplementedError(arg, type(arg), type(arg).__name__)


def torch_symbol_bind(
    operand: SSAValue, tensor: FakeTensor, symbol_map: dict[str, TorchSymbolicIntOp]
):
    bind_symbols: list[SSAValue] = []
    affine_expr_map: dict[str, AffineSymExpr] = dict()
    results: list[AffineSymExpr] = []
    for dim in tensor.shape:
        if isinstance(dim, int):
            results.append(AffineConstantExpr(dim))
            continue
        elif str(dim).isdigit():
            results.append(AffineConstantExpr(int(dim)))
            continue
        else:
            affine_exp = _generate_affine_symbolic(
                dim.node.expr, symbol_map, affine_expr_map, bind_symbols
            )
            results.append(affine_exp)
    map = AffineMap(0, len(symbol_map), results=results)
    expressions = AffineMapAttr(map)
    return SymbolicBindOp(
        operands=[operand, bind_symbols], attributes={"expressions": expressions}
    )


def torch_fake_or_mate_tensor_translate(tensor: FakeTensor):
    ele_type = torch_dtype_translate(tensor.dtype)
    shape = []
    for dim in tensor.shape:
        if isinstance(dim, int):
            shape.append(dim)
        if isinstance(dim, torch.SymInt):
            if str(dim).isdigit():
                shape.append(dim.node.int_())
            else:
                shape.append(DYNAMIC_INDEX)
    return TensorType(element_type=ele_type, shape=shape)


def make_input_tensor_symbol_attrs(tensor: FakeTensor):
    shape = []
    for dim in tensor.shape:
        if isinstance(dim, int):
            shape.append(str("c") + str(dim))
        if isinstance(dim, torch.SymInt):
            if str(dim).isdigit():
                shape.append(str("c") + str(dim.node.int_()))
            else:
                shape.append(str(dim))
    string_attr_dict = dict()
    for index, dim in zip(range(len(shape)), shape):
        string_attr_dict["func.input_symbol_" + str(index)] = StringAttr(dim)
    encode = DictionaryAttr(string_attr_dict)
    return encode


TORCH_DTYPE_TO_MLIR_TYPE = {
    torch.int64: i64,
    torch.int32: i32,
    torch.float16: f16,
    torch.float32: f32,
    torch.float64: f64,
    torch.bool: i1,
}


def torch_dtype_translate(dtype: torch.dtype):
    return TORCH_DTYPE_TO_MLIR_TYPE[dtype]


def get_result_type_ext(node: torch.fx.node.Node, index: int):
    if "tensor_meta" in node.meta:
        return node.meta["tensor_meta"][index]
    if "val" in node.meta:
        return node.meta["val"][index]
    if "example_value" in node.meta:
        return node.meta["example_value"][index]
    raise ValueError("No example_value found in node meta")


# 一些特殊的op,val里面有多个fake tensor，但是只需要使用1个返回值，保存返回的索引。
SPECIAL_RESULT_FAKE_INDEX_MAP = {
    "aten.max_pool2d_with_indices.default": 0,
}

# 一些特殊的op，实际getitem 拿到是输入
SPECIAL_GETITEM_IS_OPERAND_MAP = {}


def get_result_type(
    node: torch.fx.node.Node,
):
    target_name = node.target.__str__()
    if target_name in SPECIAL_RESULT_FAKE_INDEX_MAP:
        return get_result_type_ext(node, SPECIAL_RESULT_FAKE_INDEX_MAP[target_name])
    if "tensor_meta" in node.meta:
        return node.meta["tensor_meta"]
    if "val" in node.meta:
        return node.meta["val"]
    if "example_value" in node.meta:
        return node.meta["example_value"]
    raise ValueError("No example_value found in node meta")


def get_arg_value(
    arg: str | int | float,
    value_map: dict[str:[SSAValue]],
    block: Block,
    index: int = 0,
    tensor_const=False,
    const_type=None,
):
    if isinstance(arg, torch.fx.node.Node):
        return value_map[arg.name][index]
    elif isinstance(arg, int) or isinstance(arg, float):
        if tensor_const:
            const = build_llh_scalar_tensor(arg, const_type)
            block.add_op(const)
            return const.result
        else:
            const = build_llh_constant(arg)
            block.add_op(const)
            return const.result
    elif isinstance(arg, torch.fx.immutable_collections.immutable_list):
        return [get_arg_value(arg[i], value_map, block) for i in range(len(arg))]
    else:
        raise NotImplementedError(arg, type(arg))


TORCH_FUNCTION_TRANSLATE = Dict_Registry()


def torch_function_translate(
    node: torch.fx.node.Node,
    value_map: dict[str, list[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
) -> Operation:
    target: op.OpOverload = node.target
    if type(target).__name__ == "builtin_function_or_method":
        build_fn = TORCH_FUNCTION_TRANSLATE[target.__name__]
    elif isfunction(target):
        build_fn = TORCH_FUNCTION_TRANSLATE[target]
    else:
        build_fn = TORCH_FUNCTION_TRANSLATE[target.name()]
    return build_fn(node, value_map, symbol_map, block)


TORCH_MODULE_TRANSLATE = Dict_Registry()


def torch_module_translate(
    node: torch.fx.node.Node,
    value_map: dict[str, list[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    module: torch.nn.Module,
    block: Block,
) -> Operation:
    module_stack = node.meta["nn_module_stack"]
    target = node.target
    build_fn = TORCH_MODULE_TRANSLATE[module_stack[target][1]]
    return build_fn(node, value_map, symbol_map, module, block)


TORCH_METHOD_TRANSLATE = Dict_Registry()


def torch_method_translate(
    node: torch.fx.node.Node,
    value_map: dict[str, list[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
    block: Block,
) -> Operation:
    target = node.target
    build_fn = TORCH_METHOD_TRANSLATE[target]
    return build_fn(node, value_map, symbol_map, block)


def commond_build_op(
    op_build: callable,
    operand_nums: int,
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    block: Block,
    attrs: Mapping[str, Attribute | None] = None,
):
    out = get_result_type(node)
    if isinstance(out, TensorMetadata):
        result_type = torch_fake_or_mate_tensor_translate(out)
        return op_build(
            operands=[
                get_arg_value(
                    node.args[n],
                    value_map,
                    block,
                    tensor_const=True,
                    const_type=TORCH_DTYPE_TO_MLIR_TYPE[out.dtype],
                )
                for n in range(operand_nums)
            ],
            result_types=[result_type],
            attributes=attrs,
        )
    if isinstance(out, FakeTensor):
        result_type = torch_fake_or_mate_tensor_translate(out)
        return op_build(
            operands=[
                get_arg_value(
                    node.args[n],
                    value_map,
                    block,
                    tensor_const=True,
                    const_type=TORCH_DTYPE_TO_MLIR_TYPE(out.dtype),
                )
                for n in range(operand_nums)
            ],
            result_types=[result_type],
            attributes=attrs,
        )
    if isinstance(out, torch.SymInt) or isinstance(out, torch.SymFloat):
        return op_build(
            operands=[
                get_arg_value(node.args[n], value_map, block)
                for n in range(operand_nums)
            ],
            result_types=[i64],
            attributes=attrs,
        )
    raise NotImplementedError(out, type(out))


def _expand_to_2_if_int(value):
    if isinstance(value, int):
        return [value, value]
    return value


def torch_build_func(
    graph: torch.fx.Graph,
    name: str,
    block: Block,
    value_map: dict[str, list[SSAValue]],
    symbol_map: dict[str, TorchSymbolicIntOp],
):
    # 输入输出
    input_types = []
    output_types = []
    return_values = []
    arg_attrs = []
    for node in graph.nodes:
        if node.op == "placeholder":
            # 张量输入
            if node.type is torch.Tensor:
                fake_tensor = node.meta["example_value"]
                tensor_type = torch_fake_or_mate_tensor_translate(fake_tensor)
                arg_attrs.append(make_input_tensor_symbol_attrs(fake_tensor))
                arg_value = block.insert_arg(tensor_type, len(input_types))
                value_map[node.name] = [arg_value]
                input_types.append(tensor_type)
                shape_bind = torch_symbol_bind(arg_value, fake_tensor, symbol_map)
                block.add_op(shape_bind)
            elif node.type is None:
                val = node.meta["val"]
                # 张量输入
                if isinstance(val, FakeTensor):
                    fake_tensor = node.meta["val"]
                    tensor_type = torch_fake_or_mate_tensor_translate(fake_tensor)
                    arg_attrs.append(make_input_tensor_symbol_attrs(fake_tensor))
                    arg_value = block.insert_arg(tensor_type, len(input_types))
                    value_map[node.name] = [arg_value]
                    input_types.append(tensor_type)
                    shape_bind = torch_symbol_bind(arg_value, fake_tensor, symbol_map)
                    block.add_op(shape_bind)
                # 符号输入
                elif isinstance(val, torch.SymInt):
                    op: TorchSymbolicIntOp = torch_symbol_translate(
                        node.meta["val"], symbol_map
                    )
                    value_map[node.name] = op.results
                    # value_map[node.name] = [
                    #     block.insert_arg(op.result.type, len(input_types))
                    # ]
                    # input_types.append(op.result.type)
                    block.add_op(op)
                else:
                    print("unimplemented placeholder type: ", type(val))
            # 符号输入
            elif node.type is torch.SymInt:
                symbol: torch.SymInt = node.meta["example_value"]
                if isinstance(symbol.node.expr, sympy.core.numbers.Integer):
                    op = build_llh_constant(int(symbol))
                elif isinstance(symbol.node.expr, sympy.core.symbol.Symbol):
                    op: TorchSymbolicIntOp = torch_symbol_translate(symbol, symbol_map)
                else:
                    raise NotImplementedError
                value_map[node.name] = op.results
                # value_map[node.name] = [
                #     block.insert_arg(op.result.type, len(input_types))
                # ]
                # input_types.append(op.result.type)
                block.add_op(op)
            else:
                print("unimplemented placeholder type: ", node.type)
        elif node.op == "call_module":
            module = graph.owning_module.get_submodule(node.target)
            op = torch_module_translate(node, value_map, symbol_map, module, block)
            value_map[node.name] = op.results
            block.add_op(op)
            _updata_torch_symbol_bind(op, block, symbol_map, node)
        # 输出
        elif node.op == "output":

            def trav_args(args):
                for arg in args:
                    if isinstance(arg, tuple):
                        trav_args(arg)
                    elif isinstance(arg, list):
                        trav_args(arg)
                    elif isinstance(arg, torch.fx.node.Node):
                        type = get_result_type(arg)
                        if isinstance(type, FakeTensor):
                            # None 是一些不需要多余输出的aten生成的
                            if value_map[arg.name] != None:
                                output_types.append(value_map[arg.name][0].type)
                                return_values.append(value_map[arg.name][0])
                    elif arg is None:
                        pass
                    else:
                        print(arg)
                        print(type(arg))
                        raise NotImplementedError(type(arg))

            trav_args(node.args)
        elif node.op == "call_function":
            op = torch_function_translate(node, value_map, symbol_map, block)
            # some op is identity
            if op is not None:
                value_map[node.name] = op.results
                block.add_op(op)
                _updata_torch_symbol_bind(op, block, symbol_map, node)
        elif node.op == "call_method":
            if node.target == "size":
                value_map[node.name] = value_map[node.args[0].name]
            else:
                op = torch_method_translate(node, value_map, symbol_map, block)
                value_map[node.name] = op.results
                block.add_op(op)
                _updata_torch_symbol_bind(op, block, symbol_map, node)
        elif node.op == "get_attr":
            value_map[node.name] = value_map[node.target]
        else:
            raise NotImplementedError(node.op, type(node.op))
    block.add_op(Return(*return_values))
    region: Region = Region(block)
    func = FuncOp(
        name, (input_types, output_types), region=region, arg_attrs=ArrayAttr(arg_attrs)
    )
    return func


def _updata_torch_symbol_bind(
    op: Operation,
    block: Block,
    symbol_map: dict[str, TorchSymbolicIntOp],
    node: torch.fx.node.Node,
):
    for i in range(len(op.results)):
        if not isinstance(op.results[i].type, TensorType):
            continue
        shape_bind = torch_symbol_bind(op.results[0], get_result_type(node), symbol_map)
        block.add_op(shape_bind)
