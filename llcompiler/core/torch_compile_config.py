import torch
from torch.fx.passes.operator_support import OperatorSupport

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
    "torch.ops.aten.view.default": None,
    "torch.ops.aten.bmm.default": None,
    "torch.ops.prims.broadcast_in_dim.default": None,
    "torch.ops.aten.clone.default": None,
    "torch.ops.prims.div.default": None,
    "torch.ops.aten.exp.default": None,
}
