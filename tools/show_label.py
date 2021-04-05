import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import time

def plot_one_box(x, img, color=None, label=None, line_thickness=3, edges=4):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 =  (int(x[0::2].min()),int(x[1::2].min())), (int(x[0::2].max()),int(x[1::2].max()))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    print("c1c2: ",c1,c2)
    for i in range(edges):
        pt1 = (int(x[i*2]),int(x[i*2+1]))
        if i!=edges-1:
            pt2 = (int(x[i*2+2]),int(x[i*2+3]))
        else:
            pt2 = (int(x[0]),int(x[1]))
        cv2.line(img,pt1,pt2,color,thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

x = []
img = plt.imread("K:\\BaiduNetdiskDownload\\train\\images\\part1\\images\\P0008.png")
with open("K:\\BaiduNetdiskDownload\\train\labelTxt-v1.5\\DOTA-v1.5_train_hbb\\P0008.txt","r") as txt:
    lines = txt.readlines()
    for line in lines:
        if len(line.split())>=10:
            x.append([float(x) for x in line.split()[0:8]])

x = np.array(x)
for label in x:
    plot_one_box(label,img)
plt.imshow(img)
plt.show()
cv2.imwrite("./output.png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
