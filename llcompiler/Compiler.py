import torch.fx
from . import core
from typing import Any, Union, List, Dict

from torch._functorch.aot_autograd import aot_module_simplified
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd
from xdsl.printer import Printer

def empty_call(*args, **kwargs):
    return 1


class LLCompiler(core.importer.Importer):
    """
    LLCompiler

    example:
        compiler = LLCompiler()
        execution = compiler.compiler(model)
        out = execution.run(inputs)
    """

    def __init__(self, mode: str = "training", **kwargs) -> None:
        """
        args:
            mode: 推理/训练
        """
        assert mode in ["inference", "training"]
        kwargs["mode"] = mode
        super().__init__(**kwargs)
        self.kwargs = kwargs

    def compiler(self, model: Any, inputs: List[torch.Tensor]):
        mlir_module = self.importer(model)
        print(mlir_module.__str__())
        return model

    def _compiler_torch_module():
        raise NotImplementedError

    def _compiler_fx_module():
        raise NotImplementedError

    def _compiler_onnx():
        raise NotImplementedError

    def __call__(self, model, inputs: List[torch.Tensor]) -> Any:
        if self.kwargs["mode"] in ["training"]:
            return aot_module_simplified(
                model,
                inputs,
                fw_compiler=self.compiler,
            )
        if self.kwargs["mode"] in ["inference"]:
            return self.compiler(model,inputs)
