import cv2
import numpy as np
img = cv2.imread('image/4.png')
#cv2.imshow('img',img)
img_old = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray', img)
gblur = cv2.GaussianBlur(img, (5, 5), 0)
#cv2.imshow('guass', gblur)
canny = cv2.Canny(gblur, 150, 380)
cv2.imshow('car-canny', canny)




lines = cv2.HoughLinesP(canny,1,np.pi/90,30,minLineLength=60,maxLineGap=65)

lines1 = lines[:,0,:]#提取为为二维
for x1,y1,x2,y2 in lines1[:]:
    cv2.line(img_old,(x1,y1),(x2,y2),(0,0,255),1)

#cv2.imshow("after drawContour", canny);
cv2.imshow("source img", img_old)
cv2.moveWindow("source img",1000,100)
cv2.waitKey(0)