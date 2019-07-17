# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 00:04:44 2018

@author: Administrator
用于测试从图像中获取指定区域的轮廓
"""
import numpy as np
import cv2
import sys
import os

"""
sobel算法测试
"""
def sobel_process(sourceImg):
    image = cv2.imread(sourceImg)
    cv2.imshow("Original", image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray", gray)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    sobelx = np.uint8(np.absolute(sobelx))
    sobely = np.uint8(np.absolute(sobely))
    sobelcombine = cv2.bitwise_or(sobelx,sobely)
    #display two images in a figure
    cv2.imshow("Edge detection by Sobel", np.hstack([gray,sobelx,sobely, sobelcombine]))

    cv2.imwrite("1_edge_by_sobel.jpg", np.hstack([gray,sobelx,sobely, sobelcombine]))


    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return;
    
    
    
    
def main_process(sourceImg):
    img = cv2.imread(sourceImg)
    title = 'cv2py'
    cv2.namedWindow(title)
    print("1111")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("source img", img)
    #img = gray
# =============================================================================
#    高斯滤波
# =============================================================================
    imblur = cv2.GaussianBlur(gray, (3,3), 1.5)
    #cv2.imshow(title,imblur)
    cv2.waitKey(1000)
# =============================================================================
#     阈值化处理
# =============================================================================
    bina = cv2.adaptiveThreshold(imblur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,-2)
    #cv2.imshow(title, bina)
   
    #thresh = cv2.adaptiveThreshold(imblur, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    #cv2.imshow(title, thresh)
    cv2.waitKey(150)
   
# =============================================================================
#     边缘检测
# =============================================================================
    canny = cv2.Canny(bina, 50, 130, 3)
    cv2.imshow("after canny", canny)
    cv2.waitKey(1500)
# =============================================================================
#    霍夫变换
# =============================================================================
    '''
    #lines = cv2.HoughLines(canny,1,np.pi/180,200)
    lines1 = lines[:,0,:]#提取为为二维
    for rho,theta in lines1[:]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a)) 
        cv2.line(img,(x1,y1),(x2,y2),(255,0,0),1)
    '''
        
    lines = cv2.HoughLinesP(canny,1,np.pi/180,30,minLineLength=60,maxLineGap=15)
    lines1 = lines[:,0,:]#提取为为二维
    for x1,y1,x2,y2 in lines1[:]: 
        cv2.line(img,(x1,y1),(x2,y2),(255,0,0),1)

    #cv2.imshow("after drawContour", canny);
    cv2.imshow("source img", img)
    cv2.waitKey(25500)
    cv2.destroyAllWindows()
    
    return;
    
    
    
if __name__ == "__main__":
    print("hello world")
    main_process("4.png")
    #sobel_process("3.png")

