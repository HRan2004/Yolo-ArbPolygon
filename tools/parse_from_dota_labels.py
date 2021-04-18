import numpy as np
import matplotlib.pyplot as plt
import os

imgs_path = "E:\\dota\\train\\images\\"
labels_source_path = "E:\\dota\\train\\source_labels\\"
labels_path = "E:\\dota\\train\\labels\\"

names = ["plane","ship","storage-tank","baseball-diamond","tennis-court","swimming-pool",
        "ground-track-field","harbor","bridge","large-vehicle","small-vehicle","helicopter",
        "roundabout","soccer-ball-field","basketball-court","container-crane"]

start_name = None # start point - file name
allow = True
if start_name:
    allow = False

for file in os.listdir(imgs_path):
    if not allow:
        if file==start_name:
            allow = True
        else:
            continue
    x = []
    img = plt.imread(imgs_path + file)
    w = img.shape[1]
    h = img.shape[0]
    label_name = file.split(".")[0] + ".txt"
    with open(labels_source_path + label_name, "r") as txt:
        lines = txt.readlines()
        for line in lines:
            paras = line.split()
            if len(paras)>=10:
                segment = [float(x) for x in paras[0:8]]
                try:
                    cls = names.index(paras[8])
                except:
                    print("class not found : "+paras[8])
                    cls = len(names)
                label = np.concatenate(([cls],segment))
                x.append(label)
        txt.close()
    x = np.array(x)
    new_lines = []
    try:
        x[:,1::2] /= w
        x[:,2::2] /= h

        for label in x:
            line = ""
            for i in range(len(label)):
                para = label[i]
                if i==0:
                    para = int(para)
                line += str(para)
                if i!=len(label)-1:
                    line += " "
                else:
                    line += "\n"
            new_lines.append(line)
    except:
        print("empty label : "+file)
    with open(labels_path+label_name,"w+") as txt:
        txt.writelines(new_lines)
        txt.close()
    print(file)


