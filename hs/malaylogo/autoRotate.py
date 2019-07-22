# -*- coding: utf-8 -*-
from hs.commonFunc.xfile import *
from hs.commonFunc.xImage import *
from math import *
import cv2
import numpy as np
import pandas as pd
import os, shutil

originallogoPath="E:\\images\\Malay\\allimage\\cemian"
cemianPath="E:\\images\\Malay\\allimage\\rotate0"#保存纯侧面图的文件夹
rotatePath="E:\\images\\Malay\\allimage\\rotate"#保存旋转的侧面图
dataframe = pd.read_csv('location.txt', names=['filename','x','y','w','h'], sep=',')
imageLogo = cv2.imdecode(np.fromfile('spark.jpg', dtype=np.uint8), cv2.IMREAD_UNCHANGED)
imageLogoResize = cv2.resize(imageLogo, (160, 55))
for index in dataframe.index:
    print('------------------------------------------------------------------------------')
    print(dataframe.loc[index, 'filename'])
    x=dataframe.loc[index, 'x']
    y = dataframe.loc[index, 'y']
    w = dataframe.loc[index, 'w']
    h = dataframe.loc[index, 'h']
    readFilePath = originallogoPath + "\\" + dataframe.loc[index, 'filename']
    if x==0:#复制文件
        saveCemianFilePath = cemianPath + "\\" + dataframe.loc[index, 'filename']
        shutil.copyfile(readFilePath, saveCemianFilePath)
    else:
        #image = cv2.imread(readFilePath)  # 从指定路径读取图像(中文路径出错)
        saveRotateFilePath = rotatePath + "\\" + dataframe.loc[index, 'filename']
        image = cv2.imdecode(np.fromfile(readFilePath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        x0=x-int(w/2)
        x1=x+int(w/2)
        y0=y-int(h/2)
        y1=y+int(h/2)
       # if(w>160):
         #   imageLogoResize = cv2.resize(imageLogo, (w, 55))
        cropImg = image[y0:y1, x0:x1]  # 获取感兴趣区域# 裁剪坐标为[y0:y1, x0:x1]
        img_old = cropImg.copy()
        cropImg = cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY)
        gblur = cv2.GaussianBlur(cropImg, (5, 5), 0)
        canny = cv2.Canny(gblur, 150, 380)
        lines = cv2.HoughLinesP(canny, 1, np.pi / 90, 50, minLineLength=60, maxLineGap=65)
        #cv2.imwrite(saveFilePath, cropImg)  # 保存到指定目录
        xstart=0
        ystart=0
        xend=0
        yend=0
        if (lines is not None):
            lines1 = lines[:, 0, :]  # 提取为为二维
            for x1, y1, x2, y2 in lines1[:]:
                xstart = x1
                ystart = y1
                xend = x2
                yend = y2
                break
            cv2.line(img_old, (xstart, ystart), (xend, yend), (0, 0, 255), 1)
            # 计算是横线还是竖线,夹角，方向
            direction = 0;  # 0表示往上的直线，1表示往下的直线
            degree = 0  # 旋转角度
            vecAB = [xstart, ystart, xend, yend]
            vecAC = [xstart, ystart, xend, ystart]
            jiajiao = angle(vecAB, vecAC)
            if (ystart > yend):
                direction = 0
                degree = jiajiao * -1
            else:
                direction = 1
                degree = jiajiao
            print("方向：%d 夹角：%d" % (direction, jiajiao))
            # cv2.imshow("after drawContour", canny);
            # 图像旋转,只处理夹角小于45度的
            if (jiajiao < 45):
                imgRotation=rotate_bound(imageLogoResize,degree)
                imgRotation=mergeImg(image,imgRotation,x,y)
                cv2.imwrite(saveRotateFilePath, imgRotation)  # 保存到指定目录
                #cv2.imshow("source img", imgRotation)
                #cv2.moveWindow("source img", 1000, 100)
                #cv2.waitKey(0)



    #图像替换
    def mergeImg(ImgSrc,imgLogo,centerx,centery):
        h, w, c = imgLogo.shape
        leftx=centerx-int(w/2)
        lefty=centery-int(h/2)
        #此两句会使得旋转后背景为黑色
        #roiImg = imgLogo[0:w,0:h]
        #ImgSrc[0:w, 0:h] = roiImg
        try:
            indexs = np.where(np.amin(imgLogo, -1) > 0)
            image[lefty:lefty + h, leftx:leftx + w, :][indexs] = imgLogo[indexs]
        except :
           print('error')
        return ImgSrc


    #旋转logo图像
    def rotate_bound(image, angle):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))




    def angle(v1, v2):
        dx1 = v1[2] - v1[0]
        dy1 = v1[3] - v1[1]
        dx2 = v2[2] - v2[0]
        dy2 = v2[3] - v2[1]
        angle1 = atan2(dy1, dx1)
        angle1 = int(angle1 * 180 / pi)
        # print(angle1)
        angle2 = atan2(dy2, dx2)
        angle2 = int(angle2 * 180 / pi)
        # print(angle2)
        if angle1 * angle2 >= 0:
            included_angle = abs(angle1 - angle2)
        else:
            included_angle = abs(angle1) + abs(angle2)
            if included_angle > 180:
                included_angle = 360 - included_angle
        return included_angle
