import os

read_path = "E:\\coco128-4\\labels\\train2017\\"
save_path = "E:\\coco128\\labels\\train2017\\"

d = 1000000

for file in os.listdir(read_path):
    if ".txt" in file:
        with open(read_path + file, "r", encoding='UTF-8') as f:
            lines = f.readlines()
            if len(lines)>0 and len(lines[0].split())==5:
                f.close()
                f = open(save_path + file, "w", encoding='UTF-8')
                newLines = []
                for line in lines:
                    at = line.split()
                    at[1] = str(max(int(float(at[1])-float(at[3])*d/2)/d,0)) # x_min
                    at[2] = str(max(int(float(at[2])-float(at[4])*d/2)/d,0)) # y_min
                    at[3] = str(max(int(float(at[1])+float(at[3])*d)/d,0)) # x_max
                    at[4] = str(max(int(float(at[2])+float(at[4])*d)/d,0)) # y_max
                    if len(at)==5:
                        newLines.append(at[0]+" "+at[1]+" "+at[4]+" "+at[3]+" "+at[4]+" "+at[3]+" "+at[2]+" "+at[1]+" "+at[2]+"\n")
                print(file)
                print(newLines)
                f.writelines(newLines)
        f.close()