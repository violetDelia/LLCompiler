import torch.fx
import llcompiler.core
from typing import Any, Union, List, Dict
import sys
from torch._functorch.aot_autograd import aot_module_simplified
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd
from xdsl.printer import Printer
from llcompiler_.entrance import do_compile, CompilerOptions
import os


def empty_call(*args, **kwargs):
    return 1


class LLCompiler(llcompiler.core.Importer):
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
        index_bit_width: int = 64,
        symbol_infer=True,
        opt_level=2,
        L3_cache_size=0,
        L2_cache_size=0,
        L1_cache_size=0,
        vebose_first_ir=False,  # 输出构建的xdsl IR
        ir_tree_dir: str = "",  # mlir ir tree dir
        log_root: str = "",  # 日志保存路径
        log_level: str = "debug",  # 日志级别
        log_llvm: bool = True,  #
        **kwargs
    ) -> None:
        """
        args:
            mode: 推理/训练
        """
        super().__init__(**kwargs)
        self.vebose_first_ir = vebose_first_ir
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
        assert opt_level > 0 and opt_level <= 3
        self.opt_level = opt_level
        if ir_tree_dir != "":
            os.makedirs(ir_tree_dir, exist_ok=True)
        self.ir_tree_dir = ir_tree_dir
        self.log_llvm = log_llvm

    def compiler(self, model: Any, inputs: List[torch.Tensor]):
        self._mlir_module = self.importer(model)
        if self.vebose_first_ir:
            print(self._mlir_module)
        compiler_options = CompilerOptions()
        compiler_options.mode = self.mode
        compiler_options.target = self.target
        compiler_options.symbol_infer = self.symbol_infer
        compiler_options.opt_level = self.opt_level
        compiler_options.L3_cache_size = self.L3_cache_size
        compiler_options.L2_cache_size = self.L2_cache_size
        compiler_options.L1_cache_size = self.L1_cache_size
        compiler_options.index_bit_width = self.index_bit_width
        compiler_options.ir_tree_dir = self.ir_tree_dir
        compiler_options.log_root = self.log_root
        compiler_options.log_level = self.log_level
        compiler_options.log_llvm = self.log_llvm

        engine = do_compile(self._mlir_module.__str__(), compiler_options)
        print("meke engine")
        return llcompiler.core.engine.Torch_ExecutionEngine(engine).run

    def _compiler_torch_module():
        raise NotImplementedError

    def _compiler_fx_module():
        raise NotImplementedError

    def _compiler_onnx():
        raise NotImplementedError

    def __call__(self, model, inputs: List[torch.Tensor]) -> Any:
        if self.mode in ["training"]:
            return aot_module_simplified(
                model,
                inputs,
                fw_compiler=self.compiler,
            )
        if self.mode in ["inference"]:
            return self.compiler(model, inputs)
