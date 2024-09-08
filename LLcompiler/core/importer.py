from typing import Any, Union
import torch
import torch.fx
from torch.fx import symbolic_trace
from ..builder import MLIR_Builder


class Importer:
    """
    import model to MLIR module.
    """

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def importer(self, model: Any):
        """
        import model to MLIR module.
        """
        if isinstance(model, torch.nn.Module):
            return self._importer_torch_module(model, **self.kwargs)
        if isinstance(model, torch.fx.GraphModule):
            return self._importer_fx_module(model, **self.kwargs)
        raise NotImplementedError

    def _importer_torch_module(self, model: torch.nn.Module, **kwargs):
        return self._importer_fx_module(model, **kwargs)

    def _importer_fx_module(self, model: torch.fx.GraphModule, **kwargs):
        builder = MLIR_Builder()
        return builder.mlir_gen(model, **kwargs)

    def _importer_onnx(self, model, **kwargs):
        raise NotImplementedError
