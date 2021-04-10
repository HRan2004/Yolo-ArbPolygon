import numpy as np
from utils.general import *
from utils.plots import *
from PIL import Image
import torch

img = Image.open("E:\\coco128\\images\\train2017\\000000000338.jpg")
img = np.array(img)
label = np.array("0.730945 0.765627 0.088922 0.468746".split())
label = list(map(float, label))
whs = np.array([img.shape[1],img.shape[0],img.shape[1],img.shape[0]])
print(whs)
label *= whs
print(label)
plot_one_box(label, img)
Image.fromarray(img).save("output.jpg")
