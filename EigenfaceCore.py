import numpy as np
from numpy import size, delete
import CreateDatabase
from numpy import linalg as LA
import numpy

def EigenfaceCore(T):
    # m = T.mean()
    m = np.mean(T, axis=1)
    # print(m)
    Train_Number = T.shape[1]
    # print("Train_Number is:",Train_Number)
    # A = np.matrix([])
    A = np.empty((98304, 1))
    for i in range(0, Train_Number):
        temp = np.double(T[:,i]) - m
        # print(len(temp))
        A = np.c_[A, temp]
    # print(A.shape)
    A = delete(A, 0, axis=1)
    # print(A.shape)
    # print(A)
    L = np.dot(np.transpose(A),A)
    # print(L)
    # print(L.shape)
    # V,D = LA.eig(np.diag(L))
    d,V = np.linalg.eig(L)
    D = np.diag(d)
    # V,D = LA.eig(L)
    # print(V)
    # print("D:",D)
    L_eig_vec = np.empty((20, 1))
    # print(L_eig_vec)
    for i in range(0,20):
        if(D[i,i]>1):
            L_eig_vec = np.c_[L_eig_vec,V[:,i]]
            # print("++++++++1")
    # print(L_eig_vec)
    # print(L_eig_vec)
    L_eig_vec = delete(L_eig_vec,0,axis=1)
    # print(L_eig_vec)
    Eigenfaces = np.dot(A,L_eig_vec)
    # print("EEEE:",Eigenfaces)
    return m,A,Eigenfaces

# 以下为模块测试所需参数、代码
# T = CreateDatabase.CreateDatabase()
# EigenfaceCore(T)