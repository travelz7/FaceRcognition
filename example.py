import tkinter
import cv2
import cv2 as cv
import CreateDatabase
from EigenfaceCore import EigenfaceCore
from Recognition import Recognition1

def compare():
    global img
    value = E1.get()
    print("您选择了第" + value + "张图片！")
    img = cv2.imread("C:/Users/78111/Desktop/TestDatabase/"+value+".jpg")
    cv2.imshow("Test Image", img)
    cv2.waitKey(0)
    T = CreateDatabase.CreateDatabase()
    m, A, Eigenfaces = EigenfaceCore(T)
    OutputName = Recognition1(img, m, A, Eigenfaces)
    SelectedImage = "C:/Users/78111/Desktop/TrainDatabase" + "/" + OutputName
    SelectedImage = cv2.imread(SelectedImage)
    cv2.imshow("SelectedImage", SelectedImage)
    cv2.waitKey(0)
    print('匹配的图片是：' + OutputName)

top = tkinter.Tk()
L1 = tkinter.Label(top, text="your number:")
L1.pack()
E1 = tkinter.Entry(top)
E1.pack()
B = tkinter.Button(top, text="确定", command=compare)
B.pack()

top.mainloop()

