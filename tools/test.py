import numpy as np

def sumArray(a,b,c):
    d = [list(map(int,str.split(","))) for str in [a,b,c]]
    e = np.zeros([len(d), len(max(d, key=lambda x: len(x)))])
    for i,j in enumerate(d) : e[i][0:len(j)] = j
    return e.max(0).sum()
print(sumArray("1,2,3","3,2,2,8","-4,10"))


