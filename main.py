import llcompiler.compiler as LLC
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo.backends.common import aot_autograd
import torch.fx
from llcompiler.core.utility import run_time
import onnx
import torchgen
import torch._dynamo

torch._dynamo.config.suppress_errors = True
torch.nn.Transformer
from transformers import BertTokenizer, BertModel, BertForMaskedLM


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_layer1 = nn.Conv2d(
            3, 10, stride=2, kernel_size=5, padding=2, dilation=5
        )
        self.conv_layer2 = nn.Conv2d(10, 3, kernel_size=5, padding=5, bias=True)
        self.batch = nn.BatchNorm2d(100)
        self.cf = nn.Linear(int((224 - 17) / 2 + 7), 2)

    def forward(self, x: torch.Tensor):
        x = self.conv_layer1(x)
        x = x + x
        c = 2 + 2 * 5 / 3
        x = x / c
        x = x + x + x * x
        x = self.conv_layer2(x)
        # x = self.cf(x)
        return x


@run_time
def compiler_model(model, inputs):
    compiler = LLC.LLCompiler(mode="inference")
    model = torch.compile(
        model=model,
        backend=compiler,
        dynamic=True,
        fullgraph=True,
    )
    return model(inputs)


@run_time
def torch_compiler(model, inputs):

    model = torch.compile(
        model=model,
        backend="inductor",
        dynamic=True,
        fullgraph=True,
    )
    return model(inputs)


if __name__ == "__main__":

    model = Net()
    # input = (torch.rand((10, 32, 512)), torch.rand((20, 32, 512)))
    # model = Net()
    input = torch.randn((2, 3, 224, 224))

    # onnx_model = torch.onnx.dynamo_export(
    #     model, input, export_options=torch.onnx.ExportOptions(dynamic_shapes=True)
    # )

    compiler_model(model, input)
    torch_compiler(model, input)
