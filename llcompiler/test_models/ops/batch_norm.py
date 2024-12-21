
import torch.nn as nn
import torch

class BatchNorm2D_Inference(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch = nn.BatchNorm2d(3)
    def forward(self, x: torch.Tensor):
        x = self.batch(x)
        
        
        # x = self.batch(x)
        # x = self.batch(x)
        # x = self.batch(x)
        return x
    
class BatchNorm1D_Inference(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch = nn.BatchNorm1d(3)
        self.batch.training = False
    def forward(self, x: torch.Tensor):
        x = self.batch(x)
        x = self.batch(x)
        x = self.batch(x)
        x = self.batch(x)
        return x