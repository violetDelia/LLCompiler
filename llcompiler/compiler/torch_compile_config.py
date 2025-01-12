import torch
from torch.fx.passes.operator_support import (
    OperatorSupport,
    SupportDict,
    CALLABLE_NODE_OPS,
    get_node_target,
)
import typing as t

aten = torch.ops.aten
prims = torch.ops.prims
LLC_DECOMPOSITIONS = {
    aten.addmm,
    aten.expand,
    aten._unsafe_view,
    aten.transpose,
    aten.add,
    aten.mul,
    aten.sub,
    aten.div,
    aten.threshold_backward,
    aten.native_batch_norm_backward,
    aten.masked_fill,
    aten._softmax,
    aten.where,
}

LLC_SUPPORT_DICT = {
    # tensor transform
    "torch.ops.prims.broadcast_in_dim.default": None,
    "torch.ops.aten.view.default": None,
    "torch.ops.aten.permute.default": None,
    "torch.ops.aten.reshape.default": None,
    # "torch.ops.aten.clone.default": None,
    # activation
    ""
    # element
    "torch.ops.prims.div.default": None,
    "torch.ops.prims.sub.default": None,
    "torch.ops.prims.add.default": None,
    "torch.ops.prims.mul.default": None,
    "torch.ops.aten.relu.default": None,
    "torch.ops.aten.abs.default": None,
    "torch.ops.aten.select.int": None,
    # "torch.ops.prims.where.default": None,
    # "torch.ops.aten.exp.default": None,
    # reduce
    # matmul
    "torch.ops.aten.bmm.default": None,
    # "torch.ops.aten.mm.default": None,
    # symbol
    "torch._sym_sqrt": None,
    "_operator.mul": None,
}


class LLCOperatorSupport(OperatorSupport):
    def __init__(self, support_dict: t.Optional[SupportDict] = LLC_SUPPORT_DICT):
        self._support_dict = support_dict or {}

    def is_node_supported(
        self, submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        """
        Args:
            `submodules`: mapping from module name to the module. This can be
                          retrieved by calling model.named_modules().

            `node`: a Fx node that we want to determine whether it's supported.

        Returns:
            `is_supported`: whether the arg `node` is supported.
        """
        if node.op in ["placeholder", "output"]:
            return False
        if node.op not in CALLABLE_NODE_OPS:
            return True

        target = get_node_target(submodules, node)
        # Target not found in _support_dict meaning that we don't support this op at all
        if target not in self._support_dict:
            return False
        # The rule for target is None meaning that we accept any dtype
        if self._support_dict[target] is None:
            return True

        args_dtypes, kwargs_dtypes = self._support_dict[target]  # type: ignore[misc]

        # Check args dtypes
        for i, dtypes in enumerate(args_dtypes):
            if len(node.args) <= i:
                break

            # None indicates we don't care about the dtype of args[i]
            if dtypes is None:
                continue

            # If arg is not a node then we don't check it
            if not isinstance(node.args[i], torch.fx.Node):
                continue

            arg_dtype = _get_arg_dtype(node.args[i])  # type: ignore[arg-type]
            if arg_dtype not in dtypes:
                return False

        # Check kwargs dtypes
        for k, dtypes in kwargs_dtypes.items():
            if k not in node.kwargs:
                continue

            # If arg is not a node then we don't check it
            if not isinstance(node.kwargs[k], torch.fx.Node):
                continue

            kwarg_dtype = _get_arg_dtype(node.kwargs[k])  # type: ignore[arg-type]
            if kwarg_dtype not in dtypes:
                return False

        return True
