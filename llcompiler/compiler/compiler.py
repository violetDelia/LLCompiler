import torch.fx
from llcompiler.executor import (
    Importer,
    GenOutput,
    Torch_ExecutionEngine,
)
from llcompiler.compiler import LLC_DECOMPOSITIONS, LLCOperatorSupport
from typing import Any, Union, List, Dict
import sys
from torch._functorch.aot_autograd import aot_module_simplified
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd
from xdsl.printer import Printer
from llcompiler_.entrance import do_compile, CompilerOptions
import os
import onnx
from torch._decomp import get_decompositions
from torch._subclasses.fake_tensor import FakeTensor
from torch._inductor.compile_fx import (
    _recursive_pre_grad_passes,
    _recursive_post_grad_passes,
    fw_compiler_freezing,
    _recursive_joint_graph_passes,
    _graph_counter,
)
from torch._inductor.freezing import freeze
from torch._inductor.fx_passes.freezing_patterns import freezing_passes
from torch._inductor.fx_passes.dedupe_symint_uses import dedupe_symints
from torch._inductor.fx_passes.reinplace import reinplace_inplaceable_ops
from torch._functorch.partitioners import default_partition
from torch._inductor.utils import BoxedBool
from torch._inductor import config
from torch._inductor.cudagraph_utils import BoxedDeviceIndex
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport


import functools


class LLCompiler(Importer, GenOutput):
    """
    LLCompiler

    example:
        compiler = LLCompiler()
        execution = compiler.compiler(model)
        out = execution.run(inputs)
    """

    def __init__(
        self,
        mode: str = "inference",  # 推理/训练
        target: str = "cpu",  # 执行平台
        pipeline: str = "transform",
        index_bit_width: int = 64,
        symbol_infer=True,
        opt_level=3,
        L3_cache_size=0,
        L2_cache_size=0,
        L1_cache_size=0,
        target_layout="NCHW",
        vebose_first_ir=False,  # 输出构建的xdsl IR
        log_root: str = "",  # 日志保存路径
        log_level: str = "debug",  # 日志级别
        log_llvm: bool = True,  #
        **kwargs,
    ) -> None:
        """
        args:
            mode: 推理/训练
        """
        super().__init__(**kwargs)
        self.vebose_first_ir = vebose_first_ir
        assert pipeline in ["basic", "transform"]
        self.pipeline = pipeline
        assert mode in ["training", "inference"]
        self.mode = mode
        self.log_root = log_root
        assert log_level in ["debug", "info", "warn", "error", "fatal"]
        self.log_level = log_level
        assert target in ["cpu"]
        self.target = target
        self.symbol_infer = symbol_infer
        self.index_bit_width = index_bit_width
        self.L3_cache_size = L3_cache_size
        self.L2_cache_size = L2_cache_size
        self.L1_cache_size = L1_cache_size
        assert target_layout in ["NCHW", "NHWC"]
        self.target_layout = target_layout
        assert opt_level > 0 and opt_level <= 3
        self.opt_level = opt_level
        self.log_llvm = log_llvm
        self.compile_count = 0

    def _process_fx(
        self, model: torch.fx.GraphModule, inputs: List[torch.Tensor], **kwargs
    ):
        op_support = LLCOperatorSupport()
        partition = CapabilityBasedPartitioner(model, op_support)
        model = partition.partition_and_fuse()
        if self.mode == "inference":
            _recursive_joint_graph_passes(model)
        model = _recursive_pre_grad_passes(model, inputs)
        _recursive_post_grad_passes(model, inputs)
        return model

    def not_compile(self, model: Any, inputs: List[torch.Tensor], **kwargs):
        return model

    def _gen_compiler_options(self):
        compiler_options = CompilerOptions()
        compiler_options.pipeline = self.pipeline
        compiler_options.mode = self.mode
        compiler_options.target = self.target
        compiler_options.symbol_infer = self.symbol_infer
        compiler_options.opt_level = self.opt_level
        compiler_options.L3_cache_size = self.L3_cache_size
        compiler_options.L2_cache_size = self.L2_cache_size
        compiler_options.L1_cache_size = self.L1_cache_size
        compiler_options.target_layout = self.target_layout
        compiler_options.index_bit_width = self.index_bit_width
        compiler_options.log_root = (
            self.log_root
            if self.log_root == ""
            else self.log_root + "_" + str(self.compile_count)
        )
        self.compile_count = self.compile_count + 1
        compiler_options.log_level = self.log_level
        compiler_options.log_llvm = self.log_llvm
        return compiler_options

    def compiler_fx(
        self, model: torch.fx.GraphModule, inputs: List[torch.Tensor], **kwargs
    ):
        model = self._process_fx(model, inputs, **kwargs)
        model.graph.print_tabular()
        for name, submodule in dict(model._modules).items():
            executor = self.compiler_fx_submodule(submodule)
            model.delete_submodule(name)
            setattr(model, name, executor)
        return model.forward

    def compiler_fx_submodule(self, model: torch.fx.GraphModule, **kwargs):
        self._mlir_module = self.importer(model)
        if self.vebose_first_ir:
            print(self._mlir_module)
        compiler_options = self._gen_compiler_options()
        engine = do_compile(self._mlir_module.__str__(), compiler_options)
        executor = Torch_ExecutionEngine(engine)
        executor.gen_outs_call = self.get_out_call(model)
        return executor

    def __call__(self, model, inputs: List[torch.Tensor]) -> Any:
        if isinstance(model, torch.fx.GraphModule):
            fw_compiler = self.compiler_fx
            bw_compiler = self.compiler_fx
            inference_compiler = self.compiler_fx
            if self.mode == "inference":
                config.freezing = True
                config.cpp.weight_prepack = False
                # TODO: 常量冻结要改torch源码，要么自定义一个pass来实现
                # fw_compiler = functools.partial(
                #     fw_compiler_freezing,
                #     dynamo_model=model,
                #     num_example_inputs=len(inputs),
                #     inner_compile=self.compiler_fx,
                #     cudagraphs=BoxedBool(config.triton.cudagraphs),
                #     graph_id=next(_graph_counter),
                #     forward_device=BoxedDeviceIndex(None),
                # )
                bw_compiler = self.not_compile
            else:
                config.freezing = False
            return aot_autograd(
                fw_compiler=fw_compiler,
                bw_compiler=bw_compiler,
                inference_compiler=inference_compiler,
                decompositions=get_decompositions(LLC_DECOMPOSITIONS),
            )(model, inputs)
        else:
            raise ValueError("Unsupported")
