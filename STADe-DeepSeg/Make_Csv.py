# -*- coding:utf-8 -*-
"""
作者：LENOVO
日期：2022年 10月12日
题目：
解释：
"""
import csv

import numpy as np
import pandas as pd

# # '''----------------------------------------------------得info--------------------------------------------------------------'''
# df_anno = pd.DataFrame(pd.read_csv(r"C:\Users\LENOVO\Desktop\硕士毕业论文\结果\击键识别\All-Amend_Backbone_I3D-num0-9_public\\thumos_annotations\user1ManualSegment.csv")).values[:]
# # for j in range(1,6):
# #         for k in ['iw','ph','rp','sd','wd']:
# #                 video = ['55user'+str(j)+'_'+k+'_'+str(i) for i in range(1,6)]
# #
# video = ['55user'+str(j)+'_'+k+'_6'for k in ['iw','ph','rp','sd','wd']for j in range(1,6)]
# print('len(video)',len(video))
# fps = [1 for i in range(25)]
# sample_fps = [1 for i in range(25)]
# count = [8000 for i in range(25)]
# sample_count = [8000 for i in range(25)]
# #
# # # 字典中的key值即为csv中的列名
# test = pd.DataFrame({'video': video, 'fps': fps, 'sample_fps': sample_fps, 'count': count, 'sample_count': sample_count})
# #
# # train = frame.sample(frac=0.8, replace=False,random_state=30)  # 随机选取一些数做训练集
# # test = frame.drop(train.index,axis=0)
# # print('train',train)
# # print('test',test)
# # # # 将DataFrame存储为csv，index表识时候显示行名，default=True
# # train.to_csv("train_BehavePublic_info.csv", index=False, sep=',')
# test.to_csv("test_BehavePublic_info.csv", index=False, sep=',')
# #
# #
# #
# # # '''-------------------------------------------------------over---------------------------------------------------------'''
# place_lable = np.load('E:\Pacode\MAP2\count-mAP-txt-master\place_lable30.npy',allow_pickle=True)
# # '''第3条序列在最后一次击键动作在序列分段时丢失'''
'''--------------------------------------------------------得annotation------------------------------------------------------------'''
df_anno1 = pd.DataFrame(pd.DataFrame(pd.read_csv(r"C:\Users\LENOVO\Desktop\硕士毕业论文\结果\击键识别\All-Amend_Backbone_I3D-num0-9_public\\thumos_annotations\user1ManualSegment.csv")))
train = pd.DataFrame(pd.read_csv(r"C:\Users\LENOVO\Desktop\硕士毕业论文\结果\击键识别\All-Amend_Backbone_I3D-num0-9_public\thumos_annotations\train_BehavePublic_info.csv"))
test = pd.DataFrame(pd.read_csv(r"C:\Users\LENOVO\Desktop\硕士毕业论文\结果\击键识别\All-Amend_Backbone_I3D-num0-9_public\thumos_annotations\test_BehavePublic_info.csv"))
for i in range(2,6):
    # print(i)
    df_anno4 = pd.DataFrame(pd.DataFrame(pd.read_csv(
        r"C:\Users\LENOVO\Desktop\硕士毕业论文\结果\击键识别\All-Amend_Backbone_I3D-num0-9_public\\thumos_annotations\user"+str(i)+"ManualSegment.csv")))
    # print(df_anno4)
    df_anno1 = df_anno1.merge(df_anno4,on=["fileNumber","activityNumber","startPoint","endPoint","fileName","ativityCategory"], how='outer')

# print(df_anno1.iloc[:2, 0:7])
# type = [2,7,9,4,0,3,8,1,5,6,2,7,9,4,0,3,8,1,5,6,2,7,9,4,0,3,8,1,5,2,7,9,4,0,3,8,1,5,6,2,7,9,4,0,3,8,1,5,6,2,7,9,4,0,3,8,1,5,6,
#         2,7,9,4,0,3,8,1,5,6,2,7,9,4,0,3,8,1,5,6,2,7,9,4,0,3,8,1,5,6,2,7,9,4,0,3,8,1,5,6,3,6,0,7,2,5,4,1,9,8,3,6,0,7,2,5,4,1,9,8,
#         3,6,0,7,2,5,4,1,9,8,3,6,0,7,2,5,4,1,9,8,3,6,0,7,2,5,4,1,9,8,3,6,0,7,2,5,4,1,9,8,3,6,0,7,2,5,4,1,9,8,3,6,0,7,2,5,4,1,9,8,
#         3,6,0,7,2,5,4,1,9,8,3,6,0,7,2,5,4,1,9,8,0,6,9,7,8,5,2,3,4,1,0,6,9,7,8,5,2,3,4,1,0,6,9,7,8,5,2,3,4,1,0,6,9,7,8,5,2,3,4,1,
#         0,6,9,7,8,5,2,3,4,1,0,6,9,7,8,5,2,3,4,1,0,6,9,7,8,5,2,3,4,1,0,6,9,7,8,5,2,3,4,1,0,6,9,7,8,5,2,3,4,1,0,6,9,7,8,5,2,3,4,1]
#


# print(df_anno1) # [1500 rows x 6 columns]
type=[]
video1 = []
start = []
end = []
num = -1
for ano in df_anno1.values[:]:
    type.append(ano[5])
    video1.append(ano[4])
    start.append(ano[2])
    end.append(ano[3])
    # print(ano)
startFrame = start
endFrame = end
type_idx = type
print('type_idx',len(type_idx))
# print('video1',video1)
frame1 = pd.DataFrame({'video': video1, 'type': type, 'type_idx':type_idx, 'start': start, 'end': end, 'startFrame': startFrame,'endFrame': endFrame})
print(frame1)
# 创建一个空的 DataFrame
train_annotation = pd.DataFrame(columns=['video', 'type', 'type_idx', 'start','end','startFrame','endFrame'])
test_annotation = pd.DataFrame(columns=['video', 'type', 'type_idx', 'start','end','startFrame','endFrame'])
for i in range(len(train)):
        # print("train.iloc[i,0]",(train.iloc[i,0])[2:])
        train_annotation0 = frame1.loc[frame1['video'] == (train.iloc[i,0])[2:]]
        train_annotation = train_annotation.append(train_annotation0)
for j in range(len(test)):
        # print('test.iloc[j,0]]',(test.iloc[j,0])[2:])
        test_annotation0 = frame1.loc[frame1['video']==(test.iloc[j,0])[2:]]
        test_annotation = test_annotation.append(test_annotation0)
train_annotation.to_csv("train_Behave_Public.csv", index=False, sep=',')
test_annotation.to_csv("test_Behave_Public.csv", index=False, sep=',')
print(train_annotation,len(train_annotation))  # [1250 rows x 7 columns] 1250
print(test_annotation,len(test_annotation))  # [250 rows x 7 columns] 250

#
#
#
#

# '''-----------------------------------------------旧击键数据3-8_annotation_csv文件的制作---------------------------------------------------------'''
# # label = np.load(r'E:\Pacode\My_AFSD_击键识别\DataSet\NumKey3_8\label\\trainlable.npy')
# label = np.load(r'E:\Pacode\My_AFSD_击键识别\DataSet\NumKey3_8\label\\testlable.npy')
# print('label.shape',label.shape)  # (12, 4, 3)
# type = []
# video1 = []
# start = []
# end = []
# num = 48
# for i in range(len(label)):
#         for j in range(len(label[i])):
#             print('label[i][j]',label[i][j])
#             if int(label[i][j][2])== 0 :
#                 print('$')
#                 continue
#             else:
#                 video1.append('keynum3-8_'+str(i+num))
#                 start.append(int(label[i][j][0] * 6000))
#                 end.append(int(label[i][j][1] * 6000))
#                 type.append(int(label[i][j][2]))
# type_idx = type
# startFrame = start
# endFrame = end
# # print('video1',video1)
# frame1 = pd.DataFrame({'video': video1, 'type': type, 'type_idx':type_idx, 'start': start, 'end': end, 'startFrame': startFrame,'endFrame': endFrame})
# # print(frame1)
# # 创建一个空的 DataFrame
# frame1.to_csv("test_num3-8_annotation.csv", index=False, sep=',')
# print(frame1,len(frame1))
#
# '''----------------------------------------------------over--------------------------------------------------------------'''


# '''------------------------------------------------对新击键数据集进行分割-----------------------------------------------------'''
#
# list_segment1 = np.load('E:\Pacode\MAP2\count-mAP-txt-master\list_segment_30.npy')
# place_lable = np.load('E:\Pacode\MAP2\count-mAP-txt-master\place_lable30.npy',allow_pickle=True)  # 包含分段后各段包含的绝对位置标签
# Data30_keystrock = []
# for i in range(len(list_segment1)):
#     Data = np.load('E:\击键数据\第二版击键数据\处理成npy的数据\\3发送天线组合后的数据\\all-antenna-pair_number_'+str(i)+'.npy')
#     print('Data.shape',Data.shape)
#     for j in range(len(list_segment1[i])):
#         start = list_segment1[i][j][0]
#         end = list_segment1[i][j][1]
#         if len(place_lable[i][j]) == 0:
#             print('1u1_i,j:',i,j)
#             continue
#         else:
#             data = Data[start:end]
#             Data30_keystrock.append(data)
# print('Data30_keystrock.shape',np.array(Data30_keystrock).shape)  # (89, 8000, 3, 3, 30)
# '''--------------------------------------------------over----------------------------------------------------------------'''

'''---------------------------------------------对旧的行为数据集进行训练集与测试集的划分----------------------------------------------------------'''
# '''------------------------------------------得info---------------------------------------------'''
# train_info = 'F:\Pycode_2\已完成的项目\行为动作分割识别\未加傅里叶模块\Backbone_I3D-AFSD-main\\thumos_annotations\\val_video_info.csv'
# test_info = 'F:\Pycode_2\已完成的项目\行为动作分割识别\未加傅里叶模块\Backbone_I3D-AFSD-main\\thumos_annotations\\test_video_info.csv'
# video_info_path = 'no'
# train_video_name = []
# test_video_name = []
# train_fps = []
# train_sample_fps = []
# train_count = []
# train_sample_count = []
# test_fps = []
# test_sample_fps = []
# test_count = []
# test_sample_count = []
# for i in range(2):
#         if i ==0:
#                 video_info_path = train_info
#         if i ==1:
#                 video_info_path = test_info
#         df_info = pd.DataFrame(pd.read_csv(video_info_path)).values[:]
#         video_infos = {}
#         N = 1
#         rate = 0.7 # 设置训练集与测试集的比例
#
#         for info in df_info:
#                 if N % int(rate*10) :
#                         # video_infos[info[0]] = {
#                         #         'fps': info[1],
#                         #         'sample_fps': info[2],
#                         #         'count': info[3],
#                         #         'sample_count': info[4]
#                         # }
#                         # print('11',N % int(rate*10))
#                         train_video_name.append(info[0])
#                         train_fps.append(info[1])
#                         train_sample_fps.append(info[2])
#                         train_count.append(info[3])
#                         train_sample_count.append(info[4])
#                         print('train',info[0])
#                 else:
#                         test_video_name.append(info[0])
#                         test_fps.append(info[1])
#                         test_sample_fps.append(info[2])
#                         test_count.append(info[3])
#                         test_sample_count.append(info[4])
#                         print('test', info[0])
#                 N += 1
#
#
# # video = ['keynum_'+str(i) for i in range(89)]
# #
# # fps = [1 for i in range(89)]
# # sample_fps = [1 for i in range(89)]
# # count = [8000 for i in range(89)]
# # sample_count = [8000 for i in range(89)]
#
# # 字典中的key值即为csv中的列名
# train = pd.DataFrame({'video': train_video_name, 'fps': train_fps, 'sample_fps': train_sample_fps, 'count': train_count, 'sample_count': train_sample_count})
# test = pd.DataFrame({'video': test_video_name, 'fps': test_fps, 'sample_fps': test_sample_fps, 'count': test_count, 'sample_count': test_sample_count})
# # train = frame.sample(frac=0.8, replace=False,random_state=30)  # 随机选取一些数做训练集
# # test = frame.drop(train.index,axis=0)
# # print('train',train)
# # print('test',test)
# # # 将DataFrame存储为csv，index表识时候显示行名，default=True
# # train.to_csv("train_behave_info_C.csv", index=False, sep=',')
# # test.to_csv("test_behave_info_C.csv", index=False, sep=',')
# '''---------------------------------------------over------------------------------------------------'''

# '''--------------------------------------得annotation------------------------------------------------------'''
#
# train_annotation_old_path = 'F:\Pycode_2\已完成的项目\行为动作分割识别\未加傅里叶模块\Backbone_I3D-AFSD-main\\thumos_annotations\\val_Annotation_ours.csv'
# test_annotation_old_path = 'F:\Pycode_2\已完成的项目\行为动作分割识别\未加傅里叶模块\Backbone_I3D-AFSD-main\\thumos_annotations\\test_Annotation_ours.csv'
# train_annotation_old = pd.DataFrame(pd.read_csv(train_annotation_old_path))
# test_annotation_old = pd.DataFrame(pd.read_csv(test_annotation_old_path))
# train_info_old_path = 'E:\Pacode\MAP2\count-mAP-txt-master\\train_behave_info_C.csv'
# test_info_old_path = 'E:\Pacode\MAP2\count-mAP-txt-master\\test_behave_info_C.csv'
# train1 = pd.DataFrame(pd.read_csv(train_info_old_path))
# test1 = pd.DataFrame(pd.read_csv(test_info_old_path))
# frame2 = train_annotation_old.merge(test_annotation_old,on=['video', 'type', 'type_idx', 'start','end','startFrame','endFrame'], how='outer') # ,on=['video', 'type', 'type_idx', 'start','end','startFrame','endFrame'], how='outer'
# # print('frame2',frame2) # [2114 rows x 7 columns]
# # 创建一个空的 DataFrame
#
# train_annotation = pd.DataFrame(columns=['video', 'type', 'type_idx', 'start','end','startFrame','endFrame'])
# test_annotation = pd.DataFrame(columns=['video', 'type', 'type_idx', 'start','end','startFrame','endFrame'])
# for i in range(len(train1)):
#         # print("train.iloc[i,0]",train.iloc[i,0])
#         train_annotation0 = frame2.loc[frame2['video'] == train1.iloc[i,0]]
#         train_annotation = train_annotation.append(train_annotation0)
# for j in range(len(test1)):
#         # print('test.iloc[j,0]]',test.iloc[j,0])
#         test_annotation0 = frame2.loc[frame2['video']==test1.iloc[j,0]]
#         test_annotation = test_annotation.append(test_annotation0)
# train_annotation.to_csv("train_behave_annotation_C.csv", index=False, sep=',')
# test_annotation.to_csv("test_behave_annotation_C.csv", index=False, sep=',')
# print(train_annotation,len(train_annotation)) # [1634 rows x 7 columns] 1634
# print(test_annotation,len(test_annotation)) # [271 rows x 7 columns] 271
#
#
# '''--------------------------------------over---------------------------------------------------'''








'''------------------------------------------------------------over-----------------------------------------------------------------'''