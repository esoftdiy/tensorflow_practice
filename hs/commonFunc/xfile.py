# -*- coding: utf-8 -*-
import os


# print('root_dir:', root)  # 当前目录路径
# print('sub_dirs:', dirs)  # 当前路径下所有子目录
# print('files:', files)  # 当前路径下所有非目录子文件
'''
获取指定目录下所有文件列表
'''
def getFiles(file_dir):
    for root, dirs, files in os.walk(file_dir):
       return files