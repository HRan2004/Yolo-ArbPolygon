import numpy as np
from utils.general import *
from utils.plots import *
from PIL import Image
import torch

a = torch.rand((1,3))
a = a.repeat(1,2)
print(a)

