# -*- coding: utf-8 -*-
from hs.commonFunc.xfile import *
from hs.commonFunc.xImage import *
from math import *
import cv2
import numpy as np
originallogoPath="E:\\images\\Malay\\Img\\targetlogo";
files=getFiles(originallogoPath)
for nowfile in files:
    print(nowfile)
    img=readimage_utf8(originallogoPath+'\\'+nowfile)
    img_old = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gblur = cv2.GaussianBlur(img, (5, 5), 0)
    canny = cv2.Canny(gblur, 150, 380)
    lines = cv2.HoughLinesP(canny, 1, np.pi / 90, 50, minLineLength=60, maxLineGap=65)
    #从所有直线中，选择出现模式最多的一条横线

    xstart=0
    ystart=0
    xend=0
    yend=0
    if (lines is not None):
        lines1 = lines[:, 0, :]  # 提取为为二维
        for x1, y1, x2, y2 in lines1[:]:
            xstart=x1
            ystart=y1
            xend=x2
            yend=y2
            break
        cv2.line(img_old, (xstart, ystart), (xend, yend), (0, 0, 255), 1)
        #计算是横线还是竖线,夹角，方向
        direction=0;#0表示往上的直线，1表示往下的直线
        degree = 0#旋转角度
        vecAB=[xstart,ystart,xend,yend]
        vecAC = [xstart, ystart, xend, ystart]
        jiajiao=angle(vecAB,vecAC)
        if (ystart > yend):
            direction = 0
            degree=jiajiao*-1
        else:
            direction = 1
            degree = jiajiao
        print("方向：%d 夹角：%d"%(direction,jiajiao))
        # cv2.imshow("after drawContour", canny);
        #图像旋转,只处理夹角小于45度的
        if(jiajiao<45):
            height, width, channel = img_old.shape
            # 旋转后的尺寸
            heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
            widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
            matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

            matRotation[0, 2] += (widthNew - width) / 2  # 重点在这步，目前不懂为什么加这步
            matRotation[1, 2] += (heightNew - height) / 2  # 重点在这步

            imgRotation = cv2.warpAffine(img_old, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))

            cv2.imshow("source img", imgRotation)
            cv2.moveWindow("source img", 1000, 100)
            cv2.waitKey(0)

    '''
    #检测并画出所有直线
    if(lines is not None):
        lines1 = lines[:, 0, :]  # 提取为为二维
        for x1, y1, x2, y2 in lines1[:]:
            cv2.line(img_old, (x1, y1), (x2, y2), (0, 0, 255), 1)

        # cv2.imshow("after drawContour", canny);
        cv2.imshow("source img", img_old)
        cv2.moveWindow("source img", 1000, 100)
        cv2.waitKey(0)
    '''


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