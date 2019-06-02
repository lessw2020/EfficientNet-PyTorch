#EfficientNet implementation (lessw2020) - Util functions

import torch
import torch.nn as nn

class Swish(nn.Module):
    def forward(self, x):
        x = x * torch.sigmoid(x)  #nn.functional.sigmoid is deprecated, use torch.sigmoid instead
        return x
    
class Flatten(nn.Module):
    def forward(self, x): 
        return x.view(x.size(0), -1)
    
class Drop_Connect(nn.Module):
    """create a tensor mask and apply to inputs, for dropping drop_ratio % of connections"""
    def __init__(self, drop_ratio=0):
        super().__init__()
        self.keep_percent = 1.0 - drop_ratio

    def forward(self, x):
        if not self.training():
            return x
        
        batch_size = x.size(0)
        random_tensor = self.keep_percent
        random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=x.dtype)
        binary_tensor = torch.floor(random_tensor)
        output = x / self.keep_percent * binary_tensor
        
        return output