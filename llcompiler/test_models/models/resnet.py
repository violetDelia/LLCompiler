import llcompiler.compiler as LLC
import os.path
import subprocess
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
import os
from llcompiler.test_models import *


class Resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet18(num_classes = 10)
        self.resnet.avgpool = nn.Sequential(nn.Flatten(1), nn.Linear(4096, 512))
        for sub in self.modules():
            sub.training = False

    def forward(self, x: torch.Tensor):
        #x = x.reshape(x.shape[0],x.shape[1],224,224)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        # x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x
