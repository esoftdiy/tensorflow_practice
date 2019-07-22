# -*- coding: utf-8 -*-
import cv2
import numpy as np
'''
Function: opencv 打开中文路径的图片
return:image
'''
def readimage_utf8(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return image