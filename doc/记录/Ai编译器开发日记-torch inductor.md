# torch inductor

之前提到torch dynamo ,经过dynamo后模型会变为一个torch.fx图。inductor 是 torch 编译器默认的后端。他可将fx图或者fx的node生成trition/cuda源代码，torch/_inductor中是 inductor的具体实现

## Torch Pass

torch 中实现很多fx上的IR pass，这些pass大部分都在fx_passes这个文件夹上，一些是做规范化处理的，一些是做算子融合的。

比较重要的是：

pre_grad_passes： 做一些预处理工作，保证IR能够在之后的优化/lowing pass 中合法的Pass集合。

post_grad_passes:  规范化的pass集合

xxx_fusion: 一些融合算子/大算子的融合匹配pattern，一部分融合是以pattern注册的形式实现，一部分是单独写为了一个pass。

还又很多很细碎的pass，这里就不详细说明了。

在 _inductor/config.py 文件中的config 是来管理编译中的选项的，包括编译时的融合策略，c++/cuda/triton源码的生成策略，执行的设置等等。如果想自定义 inductor的 pass 可以在config上面进行设置。

如果想自己写一个IR转换的逻辑的话可以参考：

```python
@register_fusion("batch_linear")
class PreGradBatchLinearFusion(BatchFusion):
    """
    Batch linear fusion in pre grad pass.
    Fuse linear with same size with torch.baddmm
    """

    def _getitem_args(self, getitem_node: torch.fx.Node):
        if getitem_node.target != operator.__getitem__ or (
            getitem_node.op != "call_function"
        ):
            return None
        return getitem_node.args[0]

    def match(self, node: torch.fx.Node):
        if CallFunctionVarArgs(torch.nn.functional.linear).match(
            node
        ) and is_linear_node_can_be_fused(node):
            input = get_arg_value(node, 0, "input")
            weight = get_arg_value(node, 1, "weight")
            bias = get_arg_value(node, 2, "bias")
            if self.graph_search_options.get("fuse_nodes_with_same_users", False):
                users = [user.target for user in node.users.keys()]
            else:
                users = ""  # type: ignore[assignment]
            group_key = (
                "batch_linear",
                self._getitem_args(input),
                str(input.meta["example_value"].shape),
                str(weight.meta["example_value"].shape),
                bias is None,
                str(users),
            )
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: List[torch.fx.Node]):
        batch_nodes = []
        batch_inputs = []
        batch_weights = []
        batch_biases = []
        batch_inputs_metadata = []
        batch_weights_metadata = []
        batch_biases_metadata = []
        for node in subset:
            batch_nodes.append(node)
            input = get_arg_value(node, 0, "input")
            batch_inputs.append(input)
            batch_inputs_metadata.append(input.meta["example_value"])
            weight = get_arg_value(node, 1, "weight")
            batch_weights.append(weight)
            batch_weights_metadata.append(weight.meta["example_value"])
            bias = get_arg_value(node, 2, "bias")
            batch_biases.append(bias)
            if bias is not None and hasattr(bias, "meta"):
                batch_biases_metadata.append(bias.meta["example_value"])

        with graph.inserting_before(subset[0]):
            stack_inputs = graph.call_function(
                torch.stack, args=(batch_inputs,), kwargs={"dim": 0}
            )
            update_stack_example_value(stack_inputs, batch_inputs_metadata)
            stack_weights = graph.call_function(
                torch.stack, args=(batch_weights,), kwargs={"dim": 0}
            )
            update_stack_example_value(stack_weights, batch_weights_metadata)
            transpose_weight = graph.call_function(
                torch.transpose, args=(stack_weights, 1, 2)
            )
            if all(bias is None for bias in batch_biases):
                bmm = graph.call_function(
                    torch.bmm,
                    args=(stack_inputs, transpose_weight),
                )
            else:
                stack_biases = graph.call_function(
                    torch.stack, args=(batch_biases,), kwargs={"dim": 0}
                )
                update_stack_example_value(stack_biases, batch_biases_metadata)
                unsqueeze_biases = graph.call_function(
                    torch.unsqueeze, args=(stack_biases, 1)
                )
                bmm = graph.call_function(
                    torch.baddbmm,
                    args=(unsqueeze_biases, stack_inputs, transpose_weight),
                )

            bmm = graph.call_function(torch.unbind, args=(bmm,), kwargs={"dim": 0})
            for i, linear in enumerate(batch_nodes):
                with graph.inserting_after(bmm):
                    getitem = graph.call_function(operator.getitem, args=(bmm, i))
                linear.replace_all_uses_with(getitem)
                getitem.meta.update(linear.meta)
                graph.erase_node(linear)
        counters["inductor"]["batch_linear"] += 1
```

## codegen

走完fx的pass后，进行codegen之前，inductor会选择一部分node进行回调（走AOT算子），剩下的node会进行IR的生成，被编译为源码。实现逻辑在_inductor.codegen中，一些比较重要的kernel，是以tuner之后匹配生成的，如下：

```python
mm_template = TritonTemplate(
    name="mm",
    grid=mm_grid,
    source=r"""
{{def_kernel("A", "B")}}
    M = {{size("A", 0)}}
    N = {{size("B", 1)}}
    K = {{size("A", 1)}}
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = {{stride("A", 0)}}
    stride_ak = {{stride("A", 1)}}
    stride_bk = {{stride("B", 0)}}
    stride_bn = {{stride("B", 1)}}

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    {{store_output(("idx_m", "idx_n"), "acc", "mask")}}
""",
)

@register_lowering(aten.mm, type_promotion_kind=None)
def tuned_mm(mat1, mat2, *, layout=None):
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=layout)

    aten_layout = layout
    if not use_max_autotune():
        aten_layout = FlexibleLayout(
            device=layout.device, dtype=layout.dtype, size=layout.size
        )

    # options to tune from
    choices = (
        [aten_mm.bind((mat1, mat2), aten_layout)] if use_aten_gemm_kernels() else []
    )
    static_shape, is_nonzero = _is_static_problem([mat1, mat2], layout)
    if is_nonzero and use_triton_template(layout):
        for config in mm_configs(m, n, k):
            mm_template.maybe_append_choice(
                choices,
                input_nodes=(mat1, mat2),
                layout=layout,
                **mm_options(config, m, n, k, layout),
            )

    if static_shape and is_nonzero and use_cutlass_template(layout, m, n, k):
        CUTLASSGemmTemplate.add_cutlass_gemm_choices(choices, layout, [mat1, mat2])

    if use_cpp_packed_gemm_template(layout, mat1, mat2):
        CppPackedGemmTemplate.add_choices(
            choices,
            layout,
            [mat1, mat2],
        )

    if (
        len(choices) == 0
        and not use_aten_gemm_kernels()
        and inductor_config.autotune_fallback_to_aten
    ):
        log.warning("No choices for GEMM, using ATen backend as fallback")
        return aten_mm.bind((mat1, mat2), aten_layout).output_node()

    try:
        return autotune_select_algorithm("mm", choices, [mat1, mat2], layout)
    except NoValidChoicesError:
        if not inductor_config.autotune_fallback_to_aten:
            raise
        log.warning("All choices for GEMM were invalid, using ATen backend as fallback")
        return aten_mm.bind((mat1, mat2), aten_layout).output_node()
```

生成源码后，这些源码会保存在_inductor/codecache.py 的代码池中，并且在XXXCodeCache中实现相应的编译逻辑， 在_inductor/async_compile.py 中进行编译。
