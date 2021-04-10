import os

path = "E:\\coco128\\labels\\train2017\\"

for file in os.listdir(path):
    if ".txt" in file:
        with open(path+file,"r", encoding='UTF-8') as f:
            lines = f.readlines()
            if len(lines)>0 and len(lines[0].split())==5:
                f.close()
                f = open(path+file,"w", encoding='UTF-8')
                newLines = []
                for line in lines:
                    at = line.split()
                    at[1] = str(float(at[1])-float(at[3])/2)
                    at[2] = str(float(at[2])-float(at[4])/2)
                    at[3] = str(float(at[1])+float(at[3]))
                    at[4] = str(float(at[2])+float(at[4]))
                    if len(at)==5:
                        newLines.append(at[0]+" "+at[1]+" "+at[2]+" "+at[3]+" "+at[2]+" "+at[3]+" "+at[4]+" "+at[1]+" "+at[4]+"\n")
                print(file)
                print(newLines)
                f.writelines(newLines)
        f.close()