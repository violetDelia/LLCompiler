from . import core_env
from typing import Any, Union
import torch


class Importer:
    """
    import model to MLIR module.
    """

    def __init__(self, *args, **kwargs) -> None:
        return

    def importer(self, model: Any, *args, **kwargs):
        """
        import model to MLIR module.
        """
        if isinstance(model, torch.nn.Module):
            return self._importer_torch_module(model, *args, **kwargs)

        raise NotImplementedError

    def _importer_torch_module(self, model: torch.nn.Module, *args, **kwargs):
        raise NotImplementedError

    def _importer_fx_module(self, model, *args, **kwargs):
        raise NotImplementedError

    def _importer_onnx(self, model, *args, **kwargs):
        raise NotImplementedError
