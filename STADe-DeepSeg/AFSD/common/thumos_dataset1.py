import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader,TensorDataset
import tqdm
import sys
# sys.path.append('/root/tf-logs/My_AFSD/AFSD')
# from AFSD.common import videotransforms


# from AFSD.common.config import config
import random
import math


window_L=6000
num3_trage = [[[1250,2100,3],[3700,4500,3],[5300,6000,3]],[[7400,8200,3],[9100,10100,3],[11200,11900,3]],[[13200,13900,3],[15100,15900,3],[17200,17900,3]],[[19400,20200,3],
            [21700,22400,3]],[[24000,24700,3],[26100,26800,3],[28200,29000,3]],[[30500,31000,3],[32400,33100,3],[34700,35200,3]],[[36700,37300,3],[38800,39300,3],[40800,41500,3]],
              [[42600,43400,3],[44600,45300,3],[46300,46800,3]],[[48000,48800,3],[49800,50600,3],[51500,52000,3],[52950,53600,3]],[[54500,55250,3],[56000,56800,3],[57700,58350,3]]]
num4_trage = [[[2100,2900,4],[3900,4500,4],[5700,6000,4]],[[6000,6350,4],[7600,8150,4],[9100,10300,4],[11200,12000,4]],[[12600,13900,4],[14500,15600,4],[16100,17400,4]],[[18000,19400,4],
            [20400,21600,4],[22500,23600,4]],[[24250,25500,4],[26400,27750,4],[28500,29350,4]],[[30450,31650,4],[32600,33600,4],[34600,35700,4]],[[36750,37600,4],[39100,40250,4],[40900,41800,4]],
              [[42750,43750,4],[44600,45700,4],[46400,47400,4]],[[48200,49100,4],[50100,51100,4],[51800,53100,4]],[[54000,54750,4],[55700,56600,4],[57200,58100,4]]]
num5_trage = [[[2650,4200,5],[4400,5300,5]],[[6400,7900,5],[8300,9600,5],[10550,11700,5]],[[12700,13700,5],[14550,15600,5],[16600,17600,5]],[[18650,19700,5],[20650,21500,5],[22450,23500,5]],
              [[24300,25300,5],[26300,27300,5],[28250,29200,5]],[[30000,31100,5],[31900,33100,5],[33850,34900,5]],[[36000,36500,5],[37500,38550,5],[39400,40300,5],[41250,42000,5]],[[43200,44000,5],
            [44900,45900,5],[46800,47500,5]],[[48500,49450,5],[50450,51150,5],[52300,53200,5]],[[54200,54950,5],[55900,56700,5],[58100,58900,5]]]
num6_trage = [[[2800,3750,6],[4700,5600,6]],[[6650,7550,6],[8550,9500,6],[10500,11400,6]],[[12300,13300,6],[14250,15100,6],[16100,16750,6]],[[18000,18950,6],[19850,20800,6],[21850,22750,6]],
              [[24000,24950,6],[26250,27100,6],[28650,29600,6]],[[30900,31550,6],[33100,34150,6],[35100,36000,6]],[[37500,38250,6],[39700,40550,6]],[[42450,43200,6],[44350,45650,6],[46900,47750,6]],
              [[49050,50050,6],[51250,52150,6],[53300,54000,6]],[[55300,56150,6],[57600,58500,6],[59500,60000,6]]]
num7_trage = [[[1750,2650,7],[3350,4150,7],[5100,5950,7]],[[6900,7850,7],[9450,10150,7],[11450,12000,7]],[[13250,14150,7],[15350,16100,7],[17250,18000,7]],[[19300,19900,7],[21250,22150,7],[23050,23700,7]],
              [[24700,25550,7],[26900,27750,7],[28900,29750,7]],[[30650,31350,7],[32500,33200,7],[34300,35300,7]],[[36000,36700,7],[37600,38450,7],[39450,40050,7],[41050,41750,7]],[[42700,43500,7],[44300,45300,7],
               [46100,46900,7],[47400,48000,7]],[[48000,48800,7],[49400,50200,7],[51200,51950,7],[53200,53900,7]],[[54950,56050,7]]]
num8_trage = [[[1900,2850,8],[3550,4550,8],[5300,6000,8]],[[6900,7900,8],[8700,9550,8],[10300,11200,8]],[[12000,12750,8],[13700,14900,8],[15350,16400,8],[17250,18000,8]],[[19100,19800,8],[20900,21550,8],[22850,23500,8]],[[24550,25400,8],[26650,27500,8],
              [28050,29300,8]],[[30550,31300,8],[32400,33200,8],[34100,34950,8]],[[36000,36650,8],[37900,38650,8],[39800,40400,8],[41650,42000,8]],[[42000,42350,8],[43550,44300,8],[45300,46150,8],[47500,48000,8]],
              [[49100,49950,8],[50700,51550,8],[52600,53300,8]],[[54000,55150,8]]]
num_trager = [num3_trage,num4_trage,num5_trage,num6_trage,num7_trage,num8_trage]
for i in range(6):
    for j in range(len(num_trager[i])):
        a = np.array(num_trager[i][j])
        num_trager[i][j] = list(a-[0,0,2])

# np.save('/root/tf-logs/My_AFSD/thumos_annotations/num_trager.npy',num_trager)
def get_class_index_map(class_info_path='thumos_annotations/Class Index_Detection.txt'):
    txt = np.loadtxt(class_info_path, dtype=str)
    originidx_to_idx = {}
    idx_to_class = {}
    for idx, l in enumerate(txt):
        originidx_to_idx[int(l[0])] = idx + 1
        idx_to_class[idx + 1] = l[1]
    return originidx_to_idx, idx_to_class


def get_video_info(video_info_path):
    df_info = pd.DataFrame(pd.read_csv(video_info_path)).values[:]
    video_infos = {}
    for info in df_info:
        video_infos[info[0]] = {
            'fps': info[1],
            'sample_fps': info[2],
            'count': info[3],
            'sample_count': info[4]
        }
    return video_infos


def get_video_anno(video_infos,
                   video_anno_path):
    df_anno = pd.DataFrame(pd.read_csv(video_anno_path)).values[:]
    originidx_to_idx, idx_to_class = get_class_index_map()
    video_annos = {}
    for anno in df_anno:
        video_name = anno[0]
        originidx = anno[2]
        start_frame = anno[-2]
        end_frame = anno[-1]
        count = video_infos[video_name]['count']
        sample_count = video_infos[video_name]['sample_count']
        ratio = sample_count * 1.0 / count
        start_gt = start_frame * ratio
        end_gt = end_frame * ratio
        class_idx = originidx_to_idx[originidx]
        if video_annos.get(video_name) is None:
            video_annos[video_name] = [[start_gt, end_gt, class_idx]]
        else:
            video_annos[video_name].append([start_gt, end_gt, class_idx])
    return video_annos


def annos_transform(annos, clip_length):
    res = []
    for anno in annos:
        res.append([
            anno[0] * 1.0 / clip_length,
            anno[1] * 1.0 / clip_length,
            anno[2]
        ])
    return res


# def split_videos(video_infos,
#                  video_annos,
#                  clip_length=config['dataset']['training']['clip_length'],
#                  stride=config['dataset']['training']['clip_stride']):
#     # video_infos = get_video_info(config['dataset']['training']['video_info_path'])
#     # video_annos = get_video_anno(video_infos,
#     #                              config['dataset']['training']['video_anno_path'])
#     training_list = []
#     min_anno_dict = {}
#     for video_name in video_annos.keys():
#         min_anno = clip_length
#         sample_count = video_infos[video_name]['sample_count']
#         annos = video_annos[video_name]
#         if sample_count <= clip_length:
#             offsetlist = [0]
#             min_anno_len = min([x[1] - x[0] for x in annos])
#             if min_anno_len < min_anno:
#                 min_anno = min_anno_len
#         else:
#             offsetlist = list(range(0, sample_count - clip_length + 1, stride))
#             if (sample_count - clip_length) % stride:
#                 offsetlist += [sample_count - clip_length]
#         for offset in offsetlist:
#             left, right = offset + 1, offset + clip_length
#             cur_annos = []
#             save_offset = False
#             for anno in annos:
#                 max_l = max(left, anno[0])
#                 min_r = min(right, anno[1])
#                 ioa = (min_r - max_l) * 1.0 / (anno[1] - anno[0])
#                 if ioa >= 1.0:
#                     save_offset = True
#                 if ioa >= 0.5:
#                     cur_annos.append([max(anno[0] - offset, 1),
#                                       min(anno[1] - offset, clip_length),
#                                       anno[2]])
#             if len(cur_annos) > 0:
#                 min_anno_len = min([x[1] - x[0] for x in cur_annos])
#                 if min_anno_len < min_anno:
#                     min_anno = min_anno_len
#             if save_offset:
#                 start = np.zeros([clip_length])
#                 end = np.zeros([clip_length])
#                 for anno in cur_annos:
#                     s, e, id = anno
#                     d = max((e - s) / 10.0, 2.0)
#                     start_s = np.clip(int(round(s - d / 2.0)), 0, clip_length - 1)  # 生成激活指导的lose的标签
#                     start_e = np.clip(int(round(s + d / 2.0)), 0, clip_length - 1) + 1
#                     start[start_s: start_e] = 1
#                     end_s = np.clip(int(round(e - d / 2.0)), 0, clip_length - 1)
#                     end_e = np.clip(int(round(e + d / 2.0)), 0, clip_length - 1) + 1
#                     end[end_s: end_e] = 1
#                 training_list.append({
#                     'video_name': video_name,
#                     'offset': offset,
#                     'annos': cur_annos,
#                     'start': start,
#                     'end': end
#                 })
#         min_anno_dict[video_name] = math.ceil(min_anno)
#     return training_list, min_anno_dict


def load_video_data(video_infos, npy_data_path):
    data_dict = {}
    print('loading video frame data ...')
    for video_name in tqdm.tqdm(list(video_infos.keys()), ncols=0):
        data = np.load(os.path.join(npy_data_path, video_name + '.npy'))
        data = np.transpose(data, [3, 0, 1, 2])
        data_dict[video_name] = data
    return data_dict


# class THUMOS_Dataset(Dataset):
#     def __init__(self, data_dict,
#                  video_infos,
#                  video_annos,
#                  clip_length=config['dataset']['training']['clip_length'],
#                  crop_size=config['dataset']['training']['crop_size'],
#                  stride=config['dataset']['training']['clip_stride'],
#                  rgb_norm=True,
#                  training=True,
#                  origin_ratio=0.5):
#         self.training_list, self.th = split_videos(
#             video_infos,
#             video_annos,
#             clip_length,
#             stride
#         )
#         # np.random.shuffle(self.training_list)
#         self.data_dict = data_dict
#         self.clip_length = clip_length
#         self.crop_size = crop_size
#         self.random_crop = videotransforms.RandomCrop(crop_size)
#         self.random_flip = videotransforms.RandomHorizontalFlip(p=0.5)
#         self.center_crop = videotransforms.CenterCrop(crop_size)
#         self.rgb_norm = rgb_norm
#         self.training = training
#
#         self.origin_ratio = origin_ratio

    def __len__(self):
        return len(self.training_list)

    def get_bg(self, annos, min_action):
        annos = [[anno[0], anno[1]] for anno in annos]
        times = []
        for anno in annos:
            times.extend(anno)
        times.extend([0, self.clip_length - 1])
        times.sort()
        regions = [[times[i], times[i + 1]] for i in range(len(times) - 1)]
        regions = list(filter(
            lambda x: x not in annos and math.floor(x[1]) - math.ceil(x[0]) > min_action, regions))
        # regions = list(filter(lambda x:x not in annos, regions))
        region = random.choice(regions)
        return [math.ceil(region[0]), math.floor(region[1])]

    def augment_(self, input, annos, th):
        '''
        input: (c, t, h, w)
        target: (N, 3)
        '''
        try:
            gt = random.choice(list(filter(lambda x: x[1] - x[0] > 2 * th, annos)))
            # gt = random.choice(annos)
        except IndexError:
            return input, annos, False
        gt_len = gt[1] - gt[0]
        region = range(math.floor(th), math.ceil(gt_len - th))
        t = random.choice(region) + math.ceil(gt[0])
        l_len = math.ceil(t - gt[0])
        r_len = math.ceil(gt[1] - t)
        try:
            bg = self.get_bg(annos, th)
        except IndexError:
            return input, annos, False
        start_idx = random.choice(range(bg[1] - bg[0] - th)) + bg[0]
        end_idx = start_idx + th

        new_input = input.clone()
        # annos.remove(gt)
        if gt[1] < start_idx:
            new_input[:, t:t + th, ] = input[:, start_idx:end_idx, ]
            new_input[:, t + th:end_idx, ] = input[:, t:start_idx, ]

            new_annos = [[gt[0], t], [t + th, th + gt[1]], [t + 1, t + th - 1]]
            # new_annos = [[t-math.ceil(th/5), t+math.ceil(th/5)],
            #            [t+th-math.ceil(th/5), t+th+math.ceil(th/5)],
            #            [t+1, t+th-1]]

        else:
            new_input[:, start_idx:t - th] = input[:, end_idx:t, ]
            new_input[:, t - th:t, ] = input[:, start_idx:end_idx, ]

            new_annos = [[gt[0] - th, t - th], [t, gt[1]], [t - th + 1, t - 1]]
            # new_annos = [[t-th-math.ceil(th/5), t-th+math.ceil(th/5)],
            #            [t-math.ceil(th/5), t+math.ceil(th/5)],
            #            [t-th+1, t-1]]

        return new_input, new_annos, True

    def augment(self, input, annos, th, max_iter=10):
        flag = True
        i = 0
        while flag and i < max_iter:
            new_input, new_annos, flag = self.augment_(input, annos, th)
            i += 1
        return new_input, new_annos, flag

    # def __getitem__(self, idx):
    #     sample_info = self.training_list[idx]
    #     video_data = self.data_dict[sample_info['video_name']]
    #     offset = sample_info['offset']
    #     annos = sample_info['annos']
    #     th = self.th[sample_info['video_name']]
    #
    #     input_data = video_data[:, offset: offset + self.clip_length]
    #     c, t, h, w = input_data.shape
    #     if t < self.clip_length:
    #         # padding t to clip_length
    #         pad_t = self.clip_length - t
    #         zero_clip = np.zeros([c, pad_t, h, w], input_data.dtype)
    #         input_data = np.concatenate([input_data, zero_clip], 1)
    #
    #     # random crop and flip
    #     if self.training:
    #         input_data = self.random_flip(self.random_crop(input_data))
    #     else:
    #         input_data = self.center_crop(input_data)
    #
    #     # import pdb;pdb.set_trace()
    #     input_data = torch.from_numpy(input_data).float()
    #     if self.rgb_norm:
    #         input_data = (input_data / 255.0) * 2.0 - 1.0
    #     ssl_input_data, ssl_annos, flag = self.augment(input_data, annos, th, 1)
    #     annos = annos_transform(annos, self.clip_length)
    #     target = np.stack(annos, 0)
    #     ssl_target = np.stack(ssl_annos, 0)
    #
    #     scores = np.stack([
    #         sample_info['start'],
    #         sample_info['end']
    #     ], axis=0)
    #     scores = torch.from_numpy(scores.copy()).float()
    #
    #     return input_data, target, scores, ssl_input_data, ssl_target, flag


def detection_collate(batch):
    targets = []
    clips = []
    scores = []

    ssl_targets = []
    ssl_clips = []
    flags = []
    for sample in batch:
        clips.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        scores.append(sample[2])

        ssl_clips.append(sample[3])
        ssl_targets.append(torch.FloatTensor(sample[4]))
        flags.append(sample[5])
    return torch.stack(clips, 0), targets, torch.stack(scores, 0), \
           torch.stack(ssl_clips, 0), ssl_targets, flags

'''-------------------------------wifi数据处理-------------------------------------------------'''
def scorse_fun(y_train):
    clip_length = window_L
    Score=[]
    for anno1 in y_train:
        start = np.zeros([clip_length])
        end = np.zeros([clip_length])
        for anno in anno1:
            s, e, id = anno
            # print('s',s,'e',e)
            s=s * window_L
            e=e * window_L
            d = max((e - s) / 5.0, 2.0)  # 大部分100~200
            # print("d:",d)
            start_s = np.clip(int(round(s - d / 2.0)), 0, clip_length - 1)
            start_e = np.clip(int(round(s + d / 2.0)), 0, clip_length - 1) + 1
            # print('start_s:',start_s,'start_e:',start_e)
            start[start_s: start_e] = 1
            end_s = np.clip(int(round(e - d / 2.0)), 0, clip_length - 1)
            end_e = np.clip(int(round(e + d / 2.0)), 0, clip_length - 1) + 1
            end[end_s: end_e] = 1
        scores = np.stack([
            start,
            end], axis=0)
        Score.append(scores)  #  (48, 2, 6000)
    Score=np.array(Score)
    print('scores.shape',Score.shape)
    scores = torch.from_numpy(Score.copy()).float()
    return scores

def data_processing(window_L,data_list,num):
    '''
    加载各类击键数据，对数据进行分割生成样本，生成样本对应的标签，生成训练集，测试集
    :param window_L: 多少个点为一个样本
    :type int
    :param data_list: 取哪几个击键数据序列  需注意其位置对应其标签 例如【8，7，6】则数字键8的标签为0，数字键7的标签为1.
    :param num: 决定训练集与测试集的划分  表示击键个数  例如 num=25 表示击键序列第25个击键之前是训练集，第25个击键及其之后的数据为测试集
    :return: 训练集 测试集 及标签
    :type  训练测试集维列表 且元素形状不同
    '''
    key_wave = [ [] for j in range(len(data_list)) ]  # 用于存放多种击键类型的击键波形  type:[len(data_list),击键个数，击键所占的点数，单个点的维度] example:(1, 30, 1477, 270)
    No_key_wave = [ [] for j in range(len(data_list)) ]  # 用于存放非击键波形
    order = 0
    lable_list = [[]for j in range(len(data_list))]  # 用于存放多种击键类型的标签序列
    Train = [[]for j in range(len(data_list))]  # 用于存放多种击键类型的标签序列
    Train_lable = [[]for j in range(len(data_list))]  # 用于存放多种击键类型的标签序列
    Test = [[] for j in range(len(data_list))]  # 用于存放多种击键类型的标签序列
    Test_lable = [[] for j in range(len(data_list))]  # 用于存放多种击键类型的标签序列
    for i in data_list:  # 所要取的击键数据
        # start_end_point_set = np.load('G:\pycode\击键识别\Keystroke_recognization_01\SE_point_2f-1sou_num' + str(i) + '.npy')
        key_wave_all0 = np.load('/root/tf-logs/My_AFSD_击键识别/DataSet/all-antenna-pair_number_'+str(i)+'.npy')  # (6500,3,3,30)
        key_wave_all = np.reshape(key_wave_all0, (len(key_wave_all0), -1,30), order='c')
        # print('train_data.shape', key_wave_all.shape)  # (65000, 9,30)
        # '''----------------------------------------将序列按击键动作切割开---------------------------------------------------------------------'''
        #
        # for k in range(len(start_end_point_set)-1):
        #     b = list(key_wave_all[start_end_point_set[k, 0]:start_end_point_set[k, 1] , :])  # 取出特定天线对的单个击键 # -7 是因为数字键7最后一个击键波形的跨度超出了所采集到的数据长度
        #     d = list(key_wave_all[start_end_point_set[k, 1]:start_end_point_set[k+1, 0] , :])
        #     key_wave[order].append(b)  # 添加击键发生时的波形
        #     No_key_wave[order].append(d)  # 添加击键未发生时的波形
        # b_final = list(key_wave_all[start_end_point_set[len(start_end_point_set)-1, 0]:start_end_point_set[len(start_end_point_set)-1, 1] , :])
        # key_wave[order].append(b_final)
        # d_final = list(key_wave_all[start_end_point_set[len(start_end_point_set)-1, 1]:, :])
        # No_key_wave[order].append(d_final)
        # print('key_wave[i].shape:', np.array(key_wave).shape)
        # print('No_key_wave[i].shape:', np.array(No_key_wave[order][29]).shape)
        # '''---------------------------------------------over--------------------------------------------------'''
        # '''----------------------------------------生成标签序列------------------------------------------------------'''
        # lable_list[order] = [0 for j in range(len(key_wave_all))]
        # for k in range(len(start_end_point_set)):
        #      lable_list[order][start_end_point_set[k, 0]:start_end_point_set[k, 1]] = [order+1 for j in range(start_end_point_set[k, 0],start_end_point_set[k, 1])]  # 击键发生时对应点的标签为1，反之为0
        # print('lable_list[order]',lable_list[order])
        '''--------------------------------------------over------------------------------------------------------'''
        '''------------------------------------------生成训练样本---------------------------------------------------------'''
        # for i in range(0, len(key_wave_all) - window_L, window_L):  # 将数据按窗口大小分成多组
        for i in range(0,window_L*8 , window_L): #start_end_point_set[num, 0]
            data_1 = key_wave_all[i:i + window_L]
            Train[order].append(data_1)
        '''------------------------------------------over-------------------------------------------------------------'''
        '''-------------------------------------------生成测试样本--------------------------------------------------------------'''
        for i in range(window_L*8,len(key_wave_all)-window_L , window_L):  # 为了截取到的数据长度都相同
            data_1 = key_wave_all[i:i + window_L]
            Test[order].append(data_1)
        '''--------------------------------------------over--------------------------------------------------------------'''
        order += 1
        '''-------------------------------------------------------------------------------------------'''
    train = sum(Train, [])
    train = np.array(train)  # (53, 6000, 9, 30)
    train=torch.Tensor(train)
    train = train.unsqueeze(1)
    print('train.shape', train.shape)
    # if (c[15] == np.array(Train[1][7])).all():
    #     print('ok')
    # else:
    #     print('bad')
    test = sum(Test, [])  # 取击键数据8进行测试
    test = np.array(test)
    test = torch.Tensor(test)
    test = test.unsqueeze(1)
    print('test.shape:', test.shape)
    # train = train.numpy()
    # test = test.numpy()
    # for i in range(len(train)):
    #     np.save('E:\Pacode\My_AFSD_击键识别\DataSet\\NumKey3_8\\train\\keynum3-8_'+str(i)+'.npy',train[i])
    # for i in range(len(test)):
    #     np.save('E:\Pacode\My_AFSD_击键识别\DataSet\\NumKey3_8\\test\\keynum3-8_'+str(i+48)+'.npy',test[i])


    return train,test


def data_process0():
    data_list = [3,4,5,6,7,8]  # 8 表示取数字8的击键数据
    window_L = 6000
    num_classes = len(data_list) + 1
    num = 25  # 用于击键序列训练集与测试集的化分，训练第25个击键之前是训练数据，第25个键及之后的键为测试数据
    epoch = 200
    Train , Test = data_processing(window_L=window_L, data_list=data_list, num=num)  # 因为各条序列分割

    target = np.load("/root/tf-logs/My_AFSD_击键识别/My_AFSD/thumos_annotations/num_trager.npy",allow_pickle=True)  # <class 'numpy.ndarray'>
    target = list(target)
    y_train = [[]for i in range(len(target))]
    y_test = [[]for i in range(len(target))]
    for i in range(len(target)):
        for j in range(len(target[i])):
            for k in range(len(target[i][j])):
                target[i][j][k]=list(target[i][j][k])
                l1 = target[i][j][k][0:2] #- j*window_L
                # print("\ntarget1",target[i][j][k])
                l = np.array(l1) - j*window_L
                # print("l",l)
                # print("j*window_L",j*window_L)
                l = np.round(l/window_L,3)
                # print("l1",l)
                target[i][j][k][0:2]=list(l)
                # print("type.target[i][j][k][0:2]",type(target[i][j][k]))
                # print("list(l)",list(l))
                # print("target[i][j][k][0:2]",target[i][j][k][0:2])
                # print("target2",target[i][j][k])
            # print(target[i][j])
            if j <8:
                y_train[i].append(target[i][j])
            else:
                y_test[i].append(target[i][j])
    y_train = sum(y_train, [])  # 变成3维，每个二维对应一段序列，该序列包含多个动作片段
    y_test = sum(y_test, [])
    scorse = scorse_fun(y_train)
    narry = np.zeros([len(y_train), len(max(y_train, key=lambda x: len(x))),3])
    for i, j in enumerate(y_train):
        narry[i][0:len(j)] = j  # (48, 4, 3)
    # np.save('E:\Pacode\My_AFSD_击键识别\DataSet\NumKey3_8\label\\trainlable',narry)
    y_train = torch.tensor(narry)
    narry1 = np.zeros([len(y_test), len(max(y_test, key=lambda x: len(x))),3])
    for i, j in enumerate(y_test):
        narry1[i][0:len(j)] = j  # (48, 4, 3)
    # np.save('E:\Pacode\My_AFSD_击键识别\DataSet\NumKey3_8\label\\testlable', narry1)
    y_test = torch.tensor(narry1)
    print('target:',y_train.shape)
    print('y_test.shape:',y_test.shape)
    return Train, Test, y_train, y_test, scorse


def data_process1():
    '''
        说明：
            将输入的数据处理成DataLoaber型，其可分别输出数据和标签
        :param data:
            经CisProcess_dat~npy程序处理过的数据与标签混在一起的数据   type： 4维数组
        :param batch_size：
            每个batch的大小
        :return e:
            4维数组，3*3对天线所有子载波数据   type 4维数组 2000*3*3*30,第一个3维为接收第二个3维为发送。
        '''
    ssl_input_data = torch.ones([48,1,6000,9,30])
    ssl_target = [np.zeros((3, 2)) for i in range(48)]
    ssl_target = np.array(ssl_target)
    ssl_target=torch.Tensor(ssl_target)
    flag = [True for i in range(48)]
    flag = torch.Tensor(flag)
    input_data, Test, target, y_test,  scores = data_process0()  # 数据预处理
    train_dataset = TensorDataset(input_data, target, scores, ssl_input_data, ssl_target, flag)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)  #  50：train_dataset中的数据以50个为一组，每次参数迭代以一组数据为依据进行。
    return train_dataset   # (clips, targets, scores, ssl_clips, ssl_targets, flags)

if __name__ == '__main__':
    #  data_process1()
    # target = np.load("/root/tf-logs/My_AFSD/thumos_annotations/num_trager.npy",allow_pickle=True)
    # print(target)
    input_data, Test, target, y_test,  scores = data_process0()


    print("y_train",target)
