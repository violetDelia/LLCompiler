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
from llcompiler_.compiler import Compiler, CompileOptions
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
        target: str = "x86_64",  # 执行平台
        pipeline: str = "transform",
        index_bit_width: int = 64,
        symbol_infer=True,
        target_layout="NCHW",
        vebose_first_ir=False,  # 输出构建的xdsl IR
        log_root: str = "",  # 日志保存路径
        log_level: str = "debug",  # 日志级别
        log_llvm: bool = True,  #
        cpu="",
        mtriple="",
        **kwargs,
    ) -> None:
        """
        args:
            mode: 推理/训练
        """
        super().__init__(**kwargs)
        self.vebose_first_ir = vebose_first_ir
        self.pipeline = pipeline
        self.mode = mode
        self.log_root = log_root
        self.log_level = log_level
        self.target = target
        self.target_layout = target_layout
        self.log_llvm = log_llvm
        self.cpu = cpu
        self.mtriple = mtriple
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
        compiler_options = CompileOptions()
        compiler_options.set_log_root(
            self.log_root
            if self.log_root == ""
            else self.log_root + "_" + str(self.compile_count)
        )
        compiler_options.set_mode(self.mode)
        compiler_options.set_target(self.target)
        compiler_options.set_log_level(self.log_level)
        compiler_options.set_pipeline(self.pipeline)
        compiler_options.set_global_layout(self.target_layout)
        compiler_options.set_cpu(self.cpu)
        compiler_options.set_mtriple(self.mtriple)
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
        compiler = Compiler()
        so_file = compiler.compile_mlir_to_shared_lib(
            self._mlir_module.__str__(), compiler_options
        )
        executor = Torch_ExecutionEngine(so_file)
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
