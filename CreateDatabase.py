import operator
import os
#画图
from functools import reduce
import matplotlib.pyplot as plt
#载入图片
import matplotlib.image as mpimg
#导出数组文件
import numpy as np
from numpy import delete

def file_name(file_dir):
    global files
    for root, dirs, files in os.walk(file_dir):
        print('files:', files)

def CreateDatabase():
    file_name('C:\\Users\\78111\\Desktop\\TrainDatabase')
    # print(len(files))
    files.remove('Thumbs.db')
    # print(len(files))
    # T = []
    T = np.empty((98304, 1))
    for i in range(1,len(files)+1):
        location = str(i)
        location = '/'+location+'.jpg'
        location = 'C:/Users/78111/Desktop/TrainDatabase'+location
        # print(location)
        img = mpimg.imread(location)
        row = img.shape[0]
        col = img.shape[1]
        temp = np.transpose(img).reshape(row*col, 1)
        # print(temp)
        # print(temp.shape[0])
        T = np.c_[T,temp]
        # plt.imshow(img)
        # plt.axis('off')
        # print(img)
        # print(reduce(operator.add,img))
        # T = np.append(T,reduce(operator.add,img))
    T = delete(T, 0, axis=1)
    # print(T)
    return T

# 以下为模块测试所需参数、代码
# CreateDatabase()