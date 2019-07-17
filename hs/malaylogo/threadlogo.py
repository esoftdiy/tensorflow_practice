# -*- coding: utf-8 -*-
import os
import cv2
import pandas as pd
import numpy as np
location="location.txt"
imgPath="E:\\images\\Malay\\Img\\original"
imgTargetPath='E:\\images\\Malay\\Img\\targetlogo'
#dataframe = pd.read_csv(location, header=None, sep=',')#无头部读取csv
#print(dataframe)
dataframe = pd.read_csv(location, names=['filename','x','y','w','h'], sep=',')
for index in dataframe.index:
    print(dataframe.loc[index, 'filename'])
    x=dataframe.loc[index, 'x']
    y = dataframe.loc[index, 'y']
    w = dataframe.loc[index, 'w']
    h = dataframe.loc[index, 'h']
    if x>0:
        readFilePath = imgPath + "\\" + dataframe.loc[index, 'filename']
        saveFilePath = imgTargetPath + "\\" + str(index)+'.jpg'

        #image = cv2.imread(readFilePath)  # 从指定路径读取图像(中文路径出错)
        image = cv2.imdecode(np.fromfile(readFilePath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

        x0=x-int(w/2)
        x1=x+int(w/2)
        y0=y-int(h/2)
        y1=y+int(h/2)
        cropImg = image[y0:y1, x0:x1]  # 获取感兴趣区域# 裁剪坐标为[y0:y1, x0:x1]
        #cv2.imshow("source img", cropImg)
        #k = cv2.waitKey(0)
        #cv2.imwrite('image/'+str(index)+'.jpg', cropImg)  # 保存到指定目录
        cv2.imwrite(saveFilePath, cropImg)  # 保存到指定目录
