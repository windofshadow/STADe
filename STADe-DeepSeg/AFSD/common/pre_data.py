import numpy as np

# data = np.load('/root/tf-logs/Amend_Backbone_I3D-AFSD/Label_30.npy')
# print(data.shape) # (30000, 3, 3, 30)




'''------------------------------------------------原始标签特征分析------------------------------------------------------------'''
# A = []
# B = []
# C = []
# for i in range(29):
#     a = (label[i][10][1]-label[i][10][0])/25000
#     A.append(label[i][10][1]/a-label[i][10][0]/a+5000-30000) # [-300, -170, -64, -55, 50, 143, 220, 290, 300, 200, -150, -150, -250, -100, 0, 0, 100, 200, 290, 250, 300, 150, -300, -300, -100, -250, -100, -150, 0, 200]
#     # B.append(label[i+1][10][1]-label[i][10][1]-30000) #[1095, 1101, 1059, 1055, 1073, 1097, 1130, 1100, 1000, 1200, 1050, 1050, 1100, 1100, 1000, 1100, 1100, 1150, 1000, 1150, 1000, 1100, 1100, 1100, 1050, 1050, 950, 1150, 1200]
#     # C.append(label[i+1][10][0]-label[i][10][1]-5000) # [1265, 1165, 1114, 1005, 930, 877, 840, 800, 800, 1350, 1200, 1300, 1200, 1100, 1000, 1000, 900, 860, 750, 850, 850, 1400, 1400, 1200, 1300, 1150, 1100, 1150, 1000]
# print(A)
'''------------------------------------------------over--------------------------------------------------------------'''
# '''------------将时间标签转换成序列位置标签-------------------'''
# Label = []
# for i in range(30):
#     A = []
#     a = (label[i][10][1]-label[i][10][0])/25000　　＃　按序列长度（开始的５０００个点没有击键动作发生，所以以２５０００）对时间位置进行调整
#     label[i] = label[i]//a
#     label[i] = [x-label[i][10][0] + 5000 for x in label[i]]
#     # for j in range(10):
#     #     A.append(label[i][j][1]-label[i][j][0])
#     # print(label[i])
#     # print(A)
#     Label.append(label[i][0:10])
# # print(Label)
# np.save('Label_30',Label)
# '''-------------------------------------over----------------------------------------------'''

label = np.load('/root/tf-logs/Amend_Backbone_I3D-AFSD/Label_30.npy') # 此标签仅含击键位置
print(label.shape) # (30, 10, 2)
# # print(label-15000)
# list1 = []
# for i in range(30):
#     a = label[i][2][1] -  label[i][0][0]
#     b = label[i][6][1] -  label[i][3][0]
#     c = label[i][9][1] -  label[i][7][0] 
#     list1.append([a,b,c])
# max1 = max(list1)
# min1 = min(list1)
# print(list1,"max:",max1,'min:',min1)

# '''-----------------------------------------------------对序列进行分段---------------------------------------------------------------------------'''
# def Divide(data,Lsegment):
#     segment = []
#     i = 1300   # 设置分割开始位置 1300
#     a = 0
#     b = 0
#     c = 0
#     while i < 30000-Lsegment:
#         data1 = data-i
#         data2 = data-i-Lsegment
#         list2 = []
#         for j in range(len(data)):  # 判断data1，data2中的元素是否同号
#             list2.append(data1[j][1]*data1[j][0])
#             list2.append(data2[j][1]*data2[j][0])
#         if (np.array(list2)>0).all():  #[i for i in list2 if i>0]
#             segment.append([i,i+Lsegment])
#             i = i+Lsegment
#         i +=1
#     for k in range(len(data)):
#         if segment[0][0]<=data[k][0]  and  data[k][1] <= segment[0][1] :
#             a += 1
#         if segment[1][0]<=data[k][0]  and  data[k][1] <= segment[1][1] :
#             b += 1
#         if segment[2][0]<=data[k][0]  and  data[k][1] <= segment[2][1] :
#             c += 1
#     if a==0 or b==0 or c==0:
#         print('有的分段不含有击键动作')
#     if a+b+c != len(data):
#         print('有的击键动作没有被包含到分段中')
#     return segment,[a,b,c]


# Lsegment = 8000 # 设置分段的长度
# list_segment = []
# for i in range(30):
#     segment,num = Divide(label[i],Lsegment)
#     list_segment.append(segment)
#     print(segment,num)

# np.save('list_segment_30',list_segment) # 该文件保存的是分段的断点位置 维度=[序列条数，3，2] 分成3段，每段包括起点位置和终点位置  
# '''------------------------------------------------------------------over-------------------------------------------------------------------------------'''
# list_segment = np.load('/root/tf-logs/Amend_Backbone_I3D-AFSD/list_segment_30.npy')
# print('list_segment[0]',list_segment[2])
# print('label[0]',label[2])

# '''-------------------------------------------------获得序列分段后的标签-------------------------------------------------------------'''
# list4 = []  # 其保存的是所有序列分段后，每段包含的击键绝对位置
# for k in range(len(list_segment)):
#     list3 = [[],[],[]] # 一条序列各段所包含的击键动作标签
#     a = 0
#     b = 0
#     c = 0
#     for i in range(len(label[k])):        
#         if list_segment[k][0][0]<=label[k][i][0]  and  label[k][i][1] <= list_segment[k][0][1] :
#             list3[0].append(label[k][i])
#             a += 1
#         if list_segment[k][1][0]<=label[k][i][0]  and  label[k][i][1] <= list_segment[k][1][1] :
#             list3[1].append(label[k][i])
#             b += 1
#         if list_segment[k][2][0]<=label[k][i][0]  and  label[k][i][1] <= list_segment[k][2][1] :
#             list3[2].append(label[k][i])
#             c += 1
#     print('list3',len(list3[0]),len(list3[1]),len(list3[2]),'K',k)
#     list4.append(list3)
# # print('list4',list4)

# # 将击键的绝对位置变为每段的相对位置
# for m in range(len(list_segment)):
#     for n in range(len(list4[m])):
#         if len(list4[m][n]) == 0 :
#             # print('-------------------')
#             continue
#         else:
#             for o in range(len(list4[m][n])): 
#                 # print('list4[m][n]',list4[m][n])
#                 # print('list3[m]',list3[m])
#                 # print('m',m,'n',n,'o',o)
#                 list4[m][n][o][0] = list4[m][n][o][0] - list_segment[m][n][0]
#                 list4[m][n][o][1] = list4[m][n][o][1] - list_segment[m][n][0]
        # print('list4',list4[m][n])             
# print('list4[0]',list4[2])

# np.save('place_lable30',list4)
'''-----------------------------------------------over------------------------------------------------------------------'''


# '''------------------------------------------------------样本类别标签---------------------------------------------------------------'''
# list1_lable0 = [2,7,9,4,0,3,8,1,5,6] # 该序列敲10次
# list1_lable1 = [3,6,0,7,2,5,4,1,9,8]
# list1_lable2 = [0,6,9,7,8,5,2,3,4,1]




'''--------------------------------------------------对数据进行切割-------------------------------------------------------------------'''
list_segment1 = np.load('/root/tf-logs/Amend_Backbone_I3D-AFSD/list_segment_30.npy')
place_lable = np.load('/root/tf-logs/Amend_Backbone_I3D-AFSD/place_lable30.npy',allow_pickle=True)  # 包含分段后各段包含的绝对位置标签
Data30_keystrock = [] 
num = -1
for i in range(len(list_segment1)):
    Data = np.load('/root/autodl-tmp/Data30/all-antenna-pair_number_'+str(i)+'.npy')
    print('Data.shape',Data.shape)
    for j in range(len(list_segment1[i])):
        start = list_segment1[i][j][0]
        end = list_segment1[i][j][1]
        if len(place_lable[i][j]) == 0:
            print('1u1_i,j:',i,j)
            continue
        else:
            num += 1 
            data = Data[start:end]
            np.save('/root/autodl-tmp/Data30/keynum_'+str(num),data)
            Data30_keystrock.append(data)
# np.save('/root/autodl-tmp/Data30/Data30_keystrock.npy',Data30_keystrock)
print('Data30_keystrock.shape',np.array(Data30_keystrock).shape)


'''--------------------------------------------------------------over--------------------------------------------------------------------'''

'''-------------------------------------------------------CSV文件制作---------------------------------------------------------------------------'''
























