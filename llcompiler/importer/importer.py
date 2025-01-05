#   Copyright 2024 时光丶人爱

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
from typing import Any, Union
import torch
import torch.fx
from torch.fx import symbolic_trace
from . import MLIR_Builder
import onnx
from torch._export.passes.replace_sym_size_ops_pass import _replace_sym_size_ops_pass


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
        if isinstance(model, onnx.ModelProto):
            return self._importer_onnx(model.graph, **self.kwargs)
        if isinstance(model, onnx.GraphModule):
            return self._importer_onnx(model, **self.kwargs)
        raise NotImplementedError

    def _importer_torch_module(self, model: torch.nn.Module, **kwargs):
        return self._importer_fx_module(model, **kwargs)

    def _importer_fx_module(self, model: torch.fx.GraphModule, **kwargs):
        builder = MLIR_Builder(**kwargs)
        return builder.mlir_gen(model)

    def _importer_onnx(self, model, **kwargs):
        builder = MLIR_Builder(**kwargs)
        raise builder.mlir_gen(model)
