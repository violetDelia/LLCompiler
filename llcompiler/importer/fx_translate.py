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
from xdsl.irdl import IRDLOperation
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
    UnitAttr,
    ModuleOp,
)
from xdsl.ir.affine.affine_map import AffineMap
from ..dialect.llh_utility import build_llh_constant, build_llh_scalar_tensor
from ..utility import Dict_Registry
from datetime import datetime
import torch.nn
from torch.fx.passes.shape_prop import TensorMetadata
from torch._subclasses.fake_tensor import FakeTensor
import os
import numpy as np
import torch.fx
from torch.fx.experimental.sym_node import SymNode
from ..dialect.llh import (
    TorchSymbolicIntOp,
    SymbolicBindOp,
    ConstantOp,
    WeightOp,
    PowOp,
    MulOp,
    ConvertToOp,
    ScalarCastOp,
)
from xdsl.dialects.func import Return, FuncOp
import sympy

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


TORCH_DTYPE_TO_NUMPY_DTYPE = {
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.bool: bool,
}


def torch_dtype_to_numpy_dtype(dtype: torch.dtype):
    return TORCH_DTYPE_TO_NUMPY_DTYPE[dtype]


TORCH_FUNCTION_TRANSLATE = Dict_Registry()


def torch_function_translate(
    node: torch.fx.node.Node,
    value_map: dict[str, list[SSAValue]],
    block: Block,
) -> Operation:
    target: torch._ops.OpOverload = node.target
    if type(target).__name__ == "builtin_function_or_method":
        build_fn = TORCH_FUNCTION_TRANSLATE[target.__name__]
    elif isfunction(target):
        build_fn = TORCH_FUNCTION_TRANSLATE[target]
    else:
        build_fn = TORCH_FUNCTION_TRANSLATE[target.name()]
    return build_fn(node, value_map, block)


def torch_symbol_translate(
    symbol: torch.SymInt | Symbol,
    value_map: dict[str:[SSAValue]],
    block: Block,
):
    if isinstance(symbol, torch.SymInt):
        return torch_symbol_translate(symbol.node.expr, value_map, block)
    elif isinstance(symbol, Symbol):
        name: str = symbol.name
        if name not in value_map:
            atts = {"sym_name": StringAttr(name)}
            op = TorchSymbolicIntOp(attributes=atts, result_types=[i64])
            value_map[name] = op.results
            block.add_op(op)
        return value_map[name][0]
    elif isinstance(symbol, sympy.core.numbers.Integer):
        name: str = str(symbol)
        op = build_llh_constant(int(symbol))
        block.add_op(op)
        return op.results[0]
    elif isinstance(symbol, sympy.core.power.Pow):
        name: str = str(symbol)
        if name not in value_map:
            left_value = torch_symbol_translate(symbol.args[0], value_map, block)
            right_value = torch_symbol_translate(symbol.args[1], value_map, block)
            op = PowOp.build(operands=[left_value, right_value], result_types=[i64])
            value_map[name] = op.results
            block.add_op(op)
        return value_map[name][0]
    elif isinstance(symbol, sympy.core.mul.Mul):
        name: str = str(symbol)
        if name not in value_map:
            left_value = torch_symbol_translate(symbol.args[0], value_map, block)
            right_value = torch_symbol_translate(symbol.args[1], value_map, block)
            op = MulOp.build(operands=[left_value, right_value], result_types=[i64])
            value_map[name] = op.results
            block.add_op(op)
        return value_map[name][0]
    else:
        raise NotImplementedError(f"Unsupported type {type(symbol)}")


def get_fake_or_mate_tensor_dims(
    tensor: FakeTensor,
    block: Block,
    value_map: dict[str:[SSAValue]],
):
    dims = []
    for dim in tensor.shape:
        if isinstance(dim, int):
            const = build_llh_constant(dim)
            block.add_op(const)
            dims.append(const)
        elif isinstance(dim, torch.SymInt):
            symbol = torch_symbol_translate(dim, value_map, block)
            dims.append(symbol)
    # shape=torch.Size([])
    if dims.__len__() == 0:
        one = build_llh_constant(1)
        dims.append(one)
        block.add_op(one)
    return dims


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
    # shape=torch.Size([])
    if shape.__len__() == 0:
        shape = [1]
    return TensorType(element_type=ele_type, shape=shape)


def make_input_tensor_symbol_attrs(tensor: FakeTensor):
    shape = []
    tensor_shape = tensor.shape
    # if input is () , it will be (1)
    if tensor_shape.__len__() == 0:
        tensor_shape = [1]
    for dim in tensor_shape:
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


def make_input_symbol_attrs(symbol: torch.SymInt):
    attr_dict = dict()
    if str(symbol).isdigit():
        attr_dict["func.symbol_int"] = StringAttr(str("c") + str(symbol))
    else:
        attr_dict["func.symbol_int"] = StringAttr(str(symbol))
    return DictionaryAttr(attr_dict)


aten = torch.ops.aten
# 一些特殊的op,val里面有多个fake tensor，但是只需要使用1个返回值，保存返回的索引。
SPECIAL_RESULT_FAKE_INDEX_MAP = {
    # aten.max_pool2d_with_indices.default: 0,
    # aten._native_batch_norm_legit_no_training.default: 0,
    # aten.native_dropout.default: 0,
}

# 一些特殊的op，实际getitem 拿到是输入
SPECIAL_GETITEM_IS_OPERAND_MAP = {}


def get_result_type(node: torch.fx.node.Node, index=None):
    # if index is None and node.target in SPECIAL_RESULT_FAKE_INDEX_MAP:
    #     return get_result_type(node, SPECIAL_RESULT_FAKE_INDEX_MAP[node.target])
    if "tensor_meta" in node.meta:
        return (
            node.meta["tensor_meta"]
            if isinstance(node.meta["tensor_meta"], TensorMetadata)
            else node.meta["tensor_meta"][index]
        )
    if "val" in node.meta:
        return (
            node.meta["val"]
            if isinstance(node.meta["val"], (FakeTensor, torch.SymInt, torch.SymFloat))
            else node.meta["val"][index]
        )
    if "example_value" in node.meta:
        raise ValueError("No example_value found in node meta")
        return (
            node.meta["example_value"][index]
            if isinstance(node.meta["example_value"], tuple)
            else node.meta["example_value"]
        )
    raise ValueError("No example_value found in node meta")


def get_arg_value(
    arg: torch.fx.node.Node | int | float,
    value_map: dict[str:[SSAValue]],
    block: Block,
    index: int = 0,
    const_tensor=False,
    const_type=None,
):
    if isinstance(arg, torch.fx.node.Node):
        return value_map[arg.name][index]
    elif isinstance(arg, int) or isinstance(arg, float):
        if const_tensor:
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


def commond_build_op(
    op_build: callable,
    operand_nums: int,
    node: torch.fx.node.Node,
    value_map: dict[str:[SSAValue]],
    block: Block,
    attrs: Mapping[str, Attribute | None] = None,
):
    out = get_result_type(node)
    if isinstance(out, (TensorMetadata, FakeTensor)):
        result_type: TensorType = torch_fake_or_mate_tensor_translate(out)
        operands = [
            get_arg_value(
                node.args[n],
                value_map,
                block,
                const_tensor=True,
                const_type=torch_dtype_translate(out.dtype),
            )
            for n in range(operand_nums)
        ]
        for index, operand in enumerate(operands):
            if not isinstance(operand.type, TensorType):
                scalar_cast = ScalarCastOp.build(
                    operands=[operand],
                    result_types=[TensorType(element_type=operand.type, shape=[1])],
                )
                block.add_op(scalar_cast)
                convert = ConvertToOp.build(
                    operands=[scalar_cast.result],
                    result_types=[
                        TensorType(element_type=result_type.element_type, shape=[1])
                    ],
                )
                block.add_op(convert)
                operands[index] = convert.result
        return op_build(operands=operands, result_types=[result_type], attributes=attrs)
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


def build_tensor_inputs(
    graph: torch.fx.Graph,
    block: Block,
    value_map: dict[str, list[SSAValue]],
    input_types: list,
    arg_attrs: list,
):
    for node in graph.nodes:
        if node.op != "placeholder":
            continue
        # is symbol input
        if node.type is torch.SymInt:
            continue
        if node.type is None:
            val = node.meta["val"]
            if isinstance(val, torch.SymInt):
                continue
            elif isinstance(val, torch.SymFloat):
                continue
            elif isinstance(val, FakeTensor):
                tensor_type = torch_fake_or_mate_tensor_translate(val)
                arg_attrs.append(make_input_tensor_symbol_attrs(val))
                arg_value = block.insert_arg(tensor_type, len(input_types))
                value_map[node.name] = [arg_value]
                input_types.append(tensor_type)
            else:
                raise ValueError(val, type(val))


def build_symbol_inputs(
    graph: torch.fx.Graph,
    block: Block,
    value_map: dict[str, list[SSAValue]],
    input_types: list,
    arg_attrs: list,
):
    for node in graph.nodes:
        if node.op != "placeholder":
            continue
        if node.type is torch.SymInt:
            symbol: torch.SymInt = node.meta["example_value"]
            raise ValueError
        elif node.type is torch.SymFloat:
            symbol: torch.SymFloat = node.meta["example_value"]
            raise ValueError
        elif node.type is None and isinstance(node.meta["val"], torch.SymInt):
            symbol: torch.SymInt = node.meta["val"]
            arg_attrs.append(make_input_symbol_attrs(symbol))
            arg_value = block.insert_arg(i64, len(input_types))
            value_map[node.name] = [arg_value]
            input_types.append(i64)
        elif node.type is None and isinstance(node.meta["val"], torch.SymFloat):
            symbol: torch.SymFloat = node.meta["val"]
            arg_attrs.append(make_input_symbol_attrs(symbol))
            arg_value = block.insert_arg(i64, len(input_types))
            input_types.append(i64)
            cast = ScalarCastOp.build(operands=[arg_value], result_types=[f32])
            block.add_op(cast)
            value_map[node.name] = cast.results
        elif node.type is None and isinstance(node.meta["val"], FakeTensor):
            continue
        else:
            raise ValueError(node.meta, type(node.meta["val"]))


def build_model_parameters(
    model: torch.fx.GraphModule, block: Block, value_map: dict[str, list[SSAValue]]
):
    params: dict[str, torch.Tensor] = {
        **dict(model.named_parameters(remove_duplicate=False)),
        **dict(model.named_buffers(remove_duplicate=False)),
    }
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
            np.array(tensor.tolist(), torch_dtype_to_numpy_dtype(tensor.dtype)),
        )
        op = WeightOp.build(
            result_types=[torch_fake_or_mate_tensor_translate(tensor)],
            attributes={"weight_file": StringAttr(weight_file)},
        )
        value_map[name] = op.results
        block.add_op(op)


def build_call_function(
    graph: torch.fx.Graph, block: Block, value_map: dict[str, list[SSAValue]]
):
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        op = torch_function_translate(node, value_map, block)
        # some op is identity
        if op is not None:
            value_map[node.name] = op.results
            block.add_op(op)


def build_get_attr(
    graph: torch.fx.Graph, block: Block, value_map: dict[str, list[SSAValue]]
):
    for node in graph.nodes:
        if node.op != "get_attr":
            continue
        value_map[node.name] = value_map[node.target]


def build_output(
    graph: torch.fx.Graph,
    value_map: dict[str, list[SSAValue]],
    output_types: list,
    return_values: list,
):
    def precess_arg(arg):
        if isinstance(arg, torch.fx.node.Node):
            type = get_result_type(arg)
            if isinstance(type, FakeTensor) or isinstance(type, TensorMetadata):
                output_types.append(value_map[arg.name][0].type)
                return_values.append(value_map[arg.name][0])
            # None 是一些不需要多余输出的aten生成的
        elif arg is None:
            pass
        else:
            raise NotImplementedError(type(arg))

    for node in graph.nodes:
        if node.op != "output":
            continue
        for arg in node.args:
            if isinstance(arg, (list, tuple)):
                for arg_i in arg:
                    precess_arg(arg_i)
            else:
                precess_arg(arg)


def torch_build_func(
    model: torch.fx.GraphModule,
    name: str,
):
    graph: torch.fx.Graph = model.graph
    value_map: dict[str, list[SSAValue]] = dict()
    block: Block = Block()
    input_types: list = []
    output_types: list = []
    return_values: list = []
    arg_attrs: list = []
    build_model_parameters(model, block, value_map)
    build_symbol_inputs(graph, block, value_map, input_types, arg_attrs)
    build_tensor_inputs(graph, block, value_map, input_types, arg_attrs)
    build_get_attr(graph, block, value_map)
    build_call_function(graph, block, value_map)
    build_output(graph, value_map, output_types, return_values)
    block.add_op(Return(*return_values))
    func = FuncOp(
        name,
        (input_types, output_types),
        region=Region(block),
        arg_attrs=ArrayAttr(arg_attrs),
    )
    return func


def torch_translate_to_mlir_module(model: torch.fx.GraphModule):
    func: FuncOp = torch_build_func(model, "main")
    func.attributes.update({"entrance": UnitAttr()})
    module = ModuleOp([func])
    return module
