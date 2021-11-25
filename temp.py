import numpy as np
from utils.general import *
from utils.plots import *
from PIL import Image
import torch

a = torch.Tensor([[10,100,100,100,100,10,10,10],[10,100,100,100,100,10,10,10]])
b = torch.Tensor([[20,100,100,100,100,10,20,10],[20,100,100,100,100,10,20,10]])
print(ppoly_loss_faster(a,b))


