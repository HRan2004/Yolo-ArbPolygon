import numpy as np
import torch

a = torch.rand((4,5))
print(a)
a[:,1::2] = 1-a[:,1::2]
print(a)


