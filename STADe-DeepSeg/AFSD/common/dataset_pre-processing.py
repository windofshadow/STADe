# -*- coding:utf-8 -*-
"""
作者：LENOVO
日期：2023年 04月08日
题目：
解释：将mat格式的数据集转为npy格式的
"""
'''1 boxing; 2 hand swing; 3 picking up; 4 hand raising; 5 running; 6 pushing; 7 squatting; 8 drawing O; 9 walking; 10 drawing X'''
# 读取.mat文件所用到的包：scipy.io 或 h5py
import h5py
import hdf5storage as hdf5
import numpy as np
import os
filePath = r'F:\Pycode_2\共公数据集\deepseg\Data_npy'
file_list = os.listdir(filePath)
print(len(file_list),file_list)
# for i in file_list:
#     imgpath = r"F:\Pycode_2\共公数据集\deepseg\Data_CsiAmplitudeCut\user5\\"+i
#     mat = hdf5.loadmat(imgpath)
#     mat_t = np.transpose(mat['data_'])
#     # print(mat.keys())
#     # print(mat.values())
#     print(mat_t.shape)  # (3, 30, 6542)
#     np.save('F:\Pycode_2\共公数据集\deepseg\Data_npy\\'+i[:-4]+'.npy', mat_t)
#     data1 = np.load('F:\Pycode_2\共公数据集\deepseg\Data_npy\\'+i[:-4]+'.npy')
#     print('加载数据后的标签',data1.shape)
#     # print(data1)

