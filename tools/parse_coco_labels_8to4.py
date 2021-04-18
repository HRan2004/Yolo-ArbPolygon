import os
import numpy

read_path = "E:\\coco128\\labels\\train2017\\"
save_path = "E:\\coco128\\labels\\train2017\\"

for file in os.listdir(read_path):
    if ".txt" in file:
        with open(read_path + file, "r", encoding='UTF-8') as f:
            lines = f.readlines()
            if len(lines)>0 and len(lines[0].split())==5:
                f.close()
                f = open(save_path + file, "w", encoding='UTF-8')
                newLines = []
                for line in lines:
                    at = numpy.array(list(map(int,line.split())))
                    x_min = at[0::2].min()
                    x_max = at[0::2].max()
                    y_min = at[1::2].min()
                    y_max = at[1::2].max()
                    x = (x_min+x_max)/2
                    y = (y_min+y_max)/2
                    w = x_max-x_min
                    h = y_max-y_min
                    if len(at)==5:
                        newLines.append(str(x)+" "+str(y)+" "+str(w)+" "+str(h)+"\n")
                print(file)
                print(newLines)
                f.writelines(newLines)
        f.close()