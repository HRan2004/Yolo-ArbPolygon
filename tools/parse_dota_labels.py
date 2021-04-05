import numpy as np
import matplotlib.pyplot as plt

x = []
img = plt.imread("K:\\BaiduNetdiskDownload\\train\\images\\part1\\images\\P0008.png")
with open("K:\\BaiduNetdiskDownload\\train\labelTxt-v1.5\\DOTA-v1.5_train_hbb\\P0008.txt","r") as txt:
    lines = txt.readlines()
    for line in lines:
        if len(line.split())>=10:
            x.append([float(x) for x in line.split()[0:8]])
    txt.close()
x = np.array(x)