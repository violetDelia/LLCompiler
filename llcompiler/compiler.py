import torch.fx
import llcompiler.core
from typing import Any, Union, List, Dict

from torch._functorch.aot_autograd import aot_module_simplified
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd
from xdsl.printer import Printer
from llcompiler_.entrance import do_compile


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
        mode: str = "inference",
        vebose_first_ir=False,
        log_path: str = "",
        log_level: str = "debug",
        target: str = "cpu",
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
        self.log_path = log_path
        assert log_level in ["debug", "info", "warn", "error", "fatal"]
        self.log_level = log_level
        assert target in ["cpu"]
        self.target = target

    def compiler(self, model: Any, inputs: List[torch.Tensor]):
        self._mlir_module = self.importer(model)
        if self.vebose_first_ir:
            print(self._mlir_module)
        print(self.log_level)
        do_compile(
            self._mlir_module.__str__(),
            self.mode,
            self.target,
            self.log_path,
            self.log_level,
        )
        return model

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
