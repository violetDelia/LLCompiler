import torch.fx


class LLCompiler:
    """
    LLCompiler

    example:
        compiler = LLCompiler()
        execution = compiler.compiler(model)
        out = execution.run(inputs)
    """

    def __init__(self) -> None:
        return

    def importer(self, model, *args, **kwargs):
        """
        import model to MLIR module.
        """
        raise NotImplementedError
    
    def _importer_torch_module(self, model, *args, **kwargs):
        raise NotImplementedError

    def compiler(self, model: torch.fx.GraphModule):
        raise NotImplementedError

    def _compiler_torch_module():
        raise NotImplementedError

    def _compiler_fx_module():
        raise NotImplementedError

    def _compiler_onnx():
        raise NotImplementedError
