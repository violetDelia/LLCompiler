from .adaptive_avgpool import (
    adaptive_avg_pool2d_convert,
    torch_adaptive_avg_pool2d_convert,
)
from .binary_ops import mul_convert, div_convert, sub_convert, add_convert
from .unary_ops import (
    abs_convert,
    relu_convert,
    torch_relu_convert,
    sqrt_convert,
    rsqrt_convert,
    reciprocal_convert,
)
from .reshape_ops import (
    expand_convert,
    flatten_convert,
    torch_flatten_convert,
    torch_reshape_convert,
    aten_view_convert,
    view_convert,
    collapse_view_convert,
    inductor_force_stride_order_convert
)
from .alloc_ops import empty_convert
from .binary_ops import add_convert
from .batch_norm import batch_norm_convert
from .cat import cat_convert
from .conv import convolution_convert, torch_conv_convert
from .dim import aten_sym_size_int_convert
from .drop import torch_drop_convert
from .getitem import builtin_getitem_convert
from .layer_norm import torch_layernorm_convert
from .maxpool import max_pool2d_convert, torch_maxpool_convert
from .transpose import transpose_convert, permute_convert
from .slice import aten_select_convert
from .linear import torch_linear_convert
from .matmul import matmul_convert
from .broadcast import broadcast_in_dim_convert
from .identity import aten_clone_convert
from .compare import eq_convert, le_convert
from .alloc_ops import scalar_convert
from .convert_to import aten_to_copy_convert
from .where import where_convert
from .reduce_ops import amax_convert,prims_sum_convert
