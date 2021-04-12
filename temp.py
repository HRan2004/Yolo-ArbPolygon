import numpy as np
from utils.general import *
from utils.plots import *
from PIL import Image
import torch

a = torch.rand((2,3))
print(a)
b = a[1].repeat(2)
print(b)


