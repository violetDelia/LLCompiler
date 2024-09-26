import torch
import torch.fx.experimental
import torch.nn as nn
import torch.nn.functional as F
import torchvision
model = torchvision.models.resnet18()
from torch.fx.experimental import get_isolated_graphmodule

