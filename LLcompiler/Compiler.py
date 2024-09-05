import torch.fx
from . import core

class LLCompiler(core.importer.Importer):
    """
    LLCompiler

    example:
        compiler = LLCompiler()
        execution = compiler.compiler(model)
        out = execution.run(inputs)
    """

    def __init__(self) -> None:
        return

    def compiler(self, model: torch.fx.GraphModule):
        raise NotImplementedError

    def _compiler_torch_module():
        raise NotImplementedError

    def _compiler_fx_module():
        raise NotImplementedError

    def _compiler_onnx():
        raise NotImplementedError
