import cv2
import numpy as np
from matplotlib.image import imread
from numpy.dual import norm
import CreateDatabase
from EigenfaceCore import EigenfaceCore

def Recognition1(img,m,A,Eigenfaces):
    # ProjectedImages = np.empty((98304, 1))
    ProjectedImages = np.empty((19, 1))
    Train_Number = Eigenfaces.shape[1]
    # print(Train_Number)
    # print(Eigenfaces.shape)
    print(Eigenfaces)
    print(A)
    for i in range(0,Train_Number):
        temp = np.dot(np.transpose(Eigenfaces),A[:,i])
        # print(temp)
        # print(temp.shape)
        ProjectedImages = np.c_[ProjectedImages, temp]
    ProjectedImages = np.delete(ProjectedImages,0,axis=1)
    # print(ProjectedImages)
    InputImage = img
    # print(InputImage)
    temp = InputImage[:,:,1]
    # print(temp)
    row,col = temp.shape
    # print('row=',row)
    # print('col=',col)
    InImage = np.transpose(temp).reshape(row*col,1)
    # print(InImage)
    # print(InImage.shape)
    # m = m.astype(np.int16)
    InImage = m.astype(np.int16)
    Difference = np.double(InImage) - m
    # print(Difference)
    # temp = np.double(T[:, i]) - np.double(m)
    ProjectedTestImages = np.dot(np.transpose(Eigenfaces),Difference)
    # print(ProjectedTestImages)
    # global Euc_dist
    Euc_dist = []
    for i in range(0,Train_Number):
        q = ProjectedImages[:,i]
        # print(q)
        temp = np.dot((norm(ProjectedTestImages - q)),(norm(ProjectedTestImages - q)))
        # print(norm(ProjectedTestImages - q))
        # print('temp=',temp)
        Euc_dist.append(temp)
    # print('list=',Euc_dist)
    Euc_dist_min = min(Euc_dist)
    # print("min=",Euc_dist_min)
    Recognized_index= Euc_dist.index(min(Euc_dist))
    # print('index=',Recognized_index)
    OutputName = str(Recognized_index) + '.jpg'
    # print(OutputName)
    return OutputName

#以下为模块测试所需参数、代码
# img = cv2.imread("C:/Users/78111/Desktop/TestDatabase/9.jpg")
# T = CreateDatabase.CreateDatabase()
# m,A,Eigenfaces = EigenfaceCore(T)
# Recognition1(img,m,A,Eigenfaces)