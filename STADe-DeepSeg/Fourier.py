import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
import math
import matplotlib.pyplot as plt
from AFSD.common.utils import *
import time
import argparse
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
# from sklearn.metrics import recall_score, f1_score, accuracy_score
import tqdm
from AFSD.common.config import config
import pandas as pd
from AFSD.common.thumos_dataset import THUMOS_Dataset, get_video_info, \
     detection_collate, get_video_anno
from torch import squeeze
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FIC(nn.Module):
    def __init__(self, window_size, stride):
        super(FIC, self).__init__()
        self.window_size = window_size
        self.k = int(window_size / 2)
        self.conv = nn.Conv1d(in_channels = 1, out_channels = 2 * int(window_size / 2), kernel_size = window_size,
            stride = stride, padding = 0, bias = False)
        self.init()

    def forward(self, x):
        # x: B * C  * L   
        B, C = x.size(0), x.size(1)

        x = torch.reshape(x, (B * C, -1)).unsqueeze(1)
        # print('x1',x.shape)
        x = self.conv(x)
        # print('\nx.parameters',list(self.conv.parameters()))
        x = torch.reshape(x, (B, C, -1, x.size(-1)))
        return x # B * C * fc * L
    

    def init(self):
        '''
            Fourier weights initialization
        '''
        basis = torch.tensor([math.pi * 2 * j / self.window_size for j in range(self.window_size)])  #
        weight = torch.zeros((self.k * 2, self.window_size))
        for i in range(self.k * 2):
            f = int(i / 2) + 1
            if i % 2 == 0:
                weight[i] = torch.cos(f * basis)
            else:
                weight[i] = torch.sin(-f * basis)
        self.conv.weight = torch.nn.Parameter(weight.unsqueeze(1), requires_grad=True)



class TSEnc(nn.Module):
    def __init__(self, window_size, stride, k):
        super(TSEnc, self).__init__()
        '''
            virtual filter choose 2 * k most important channels
        '''
        self.k = k
        self.window_size = window_size
        self.FIC = FIC(window_size = window_size, stride = stride)#.cuda()
        self.RPC = nn.Conv1d(1, 2*k, kernel_size = window_size, stride = stride)


    def forward(self, x):
        # print('x.shape',x.shape)     # [2, 1, 8000, 9, 30]  # x[batch ,采样点数 ,通道数 ]
        x = x.permute(0, 2, 1)       
        # print('in_put',x.shape)
        # fic #
        h_f = self.FIC(x)
        # print('h_f.shape',h_f.shape,'\nh_f',h_f)
        # virtual filter #
        h_f_abs=torch.abs(h_f)
        # print('h_f_abs.shape', h_f_abs.shape, '\nh_f_abs', h_f_abs)
        h_f_pos, idx_pos = h_f_abs.topk(2*self.k, dim = -2, largest = True, sorted = True)
        # print('挑选后h_f_pos.shape:\n',h_f_pos.shape,'\n挑选后：', h_f_pos,'\n索引号：',idx_pos)
        o_f_pos = torch.cat( (h_f_pos, idx_pos.type(torch.Tensor).to(h_f_pos.device )) , -2)
        # print(' o_f_pos:',  o_f_pos.shape)
        # rpc #
        B, C = x.size(0), x.size(1)
        x = torch.reshape(x, (B*C, -1)).unsqueeze(1)
        o_t = self.RPC(x)
        o_t = torch.reshape(o_t, (B, C, -1, o_t.size(-1)))
        # print(' o_t:', o_t.shape)
        o = torch.cat((o_f_pos, o_t),  -2)
        # print(' o:',o.shape)
        return o
class UniTS(nn.Module):
    def __init__(self, input_size, sensor_num, 
        window_list, stride_list, k_list, hidden_channel = 48):   # k_list：时序信号FIC变换后选几个频率。
        super(UniTS, self).__init__()
        assert len(window_list) == len(stride_list)  # 满足条件则执行接下来的代码。否则终止执行。
        assert len(window_list) == len(k_list)  # len(window_list) 为TS编码器个数
        self.hidden_channel = hidden_channel
        self.window_list = window_list

        self.ts_encoders = nn.ModuleList([
           TSEnc(window_list[i], stride_list[i], k_list[i]) for i in range(len(window_list))
            ]) # 初始化多个TS编码器
        self.num_frequency_channel = [6 * k_list[i] for i in range(len(window_list))]    # ？ 3是啥意思 感觉是用于一次融合3个传感器通道的信息
        self.current_size = [1 + int((input_size - window_list[i]) / stride_list[i])  for i in range(len(window_list))]  # 时间段的个数
        # o.size(): B * C * num_frequency_channel * current_size
        self.multi_channel_fusion = nn.ModuleList([nn.ModuleList() for _ in range(len(window_list))])
        self.conv_branches = nn.ModuleList([nn.ModuleList() for _ in range(len(window_list))])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.hidden_channel) for _ in range(len(window_list))])

        self.multi_channel_fusion = nn.ModuleList([nn.Conv2d(in_channels = sensor_num, out_channels = self.hidden_channel,
              kernel_size =(self.num_frequency_channel[i], 1), stride = (1, 1) ) for i in range(len(window_list) ) ])  # ？ 感觉这是传感器级信号融合，如何融合不同的传感器
        self.end_linear = nn.ModuleList([])
        self.k_list = k_list

    def forward(self, x):
        #x: B * L * C
        # print('fourier.shape:',x.shape) # [2, 8000, 270]
        multi_scale_x = []
        B = x.size(0)
        C = x.size(2)
        for i in range(len(self.current_size)):      # i 表示哪个分支
            tmp = self.ts_encoders[i](x)
            # print('k_list[',i,']:',self.k_list[i],'tmp:',tmp.shape)
            #tmp: B * C * fc * L'
            tmp=self.multi_channel_fusion[i](tmp)
            tmp=torch.squeeze(tmp , 2)    # tmp.squeeze(2) 
            tmp = F.relu(self.bns[i](tmp))         # TS编码器最终输出 [16,48,2832]
            if i == 0:
                interpolate_size = tmp.size()[2] + 2
                tmp = F.interpolate(tmp,size=interpolate_size, mode='linear', align_corners=True)
                # print("interpolate_size", interpolate_size)
            else:
                tmp = F.interpolate(tmp,size=interpolate_size, mode='linear', align_corners=True)
            ''' 如想实现各Ts编码器输出在通道维度叠加后各通道间的融合需改变下面代码
             先实现各Ts编码器的拼接然后融合接着进行维度的调整'''
            tmp = tmp.permute(0, 2, 1)
            # print('维度调整tmp0.shape:', tmp.shape) # [1, 4000, 144]
            tmp = tmp.view(tmp.size()[0],400, 10, tmp.size()[2]) # 序列长度取成一段一段的从而提升维度，通道维度不变
            # print('tmp1.shape:', tmp.shape) # [16, 118, 24, 48]
            tmp = tmp.unsqueeze(1)
            # print('tmp2.shape:', tmp.shape)
            multi_scale_x.append(tmp)
        x = torch.cat(multi_scale_x, -1) # [16,1, 177, 24, 480]
        # print('data_fourier_out.shape:', x.shape) 
        return x

def load_video_data(video_infos, npy_data_path):
        data_dict = {}
        print('loading video frame data ...')
        for video_name in tqdm.tqdm(list(video_infos.keys()), ncols=0):
            data = np.load(os.path.join(npy_data_path, video_name + '.npy'))
            data = torch.from_numpy(data).view(8500, 30, 1)
            # data = torch.from_numpy(data).view(340, 25, 30).unsqueeze(0)

            data = data.numpy()
            data = np.expand_dims(data, 0).repeat(1, axis=0)
            # print(data.shape) #(1, 8500, 30, 1)
            data_dict[video_name] = data
        return data_dict

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


GLOBAL_SEED = 1

def worker_init_fn(worker_id):
    set_seed(GLOBAL_SEED + worker_id)

def parse_args():
    folder_path = ""  # 相对路径
    absolute_path = os.path.abspath(folder_path)
    path_fourier="STADe-DeepSeg/configs/fourier"
    path_log = "log"
    absolute_path_fourier = os.path.join(absolute_path, path_fourier)
    absolute_path_log = os.path.join(absolute_path, path_log)
    parser = argparse.ArgumentParser(description='train and test')
    parser.add_argument('--config', default=absolute_path_fourier, type=str)  # Read UniTS hyperparameters  该值为配置文件所在路径。
    parser.add_argument('--dataset', default='wifi', type=str,
                        choices=['opportunity_lc', 'seizure', 'wifi', 'keti'])
    parser.add_argument('--model', default='UniTS', type=str,
                        choices=['UniTS', 'THAT', 'RFNet', 'ResNet', 'MaDNN', 'MaCNN', 'LaxCat', 'static'])
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--log', default=absolute_path_log, type=str,
                        help="Log directory")    # 保存日志信息的文件名
    parser.add_argument('--exp', default='', type=str,
                        choices=['', 'noise', 'missing_data'])
    parser.add_argument('--ratio', default=0.2, type=float)
    parser.add_argument('--n_gpu', default=0, type=int)

    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=64, type=int)

    parser.add_argument('--save', action='store_true')
    parser.add_argument('--test_only', action='store_true')
    args = parser.parse_args()
    config = read_config(args.config + '.yaml')  # 将 yaml文件转变为字典。
    if not os.path.exists(args.log):
        os.mkdir(args.log)  # log ：路径名
    args.log_path = os.path.join(args.log, args.dataset)
    if not os.path.exists(args.log_path):  # 在log文件路径下创建子文件夹
        os.mkdir(args.log_path)
    # torch.cuda.set_device(args.n_gpu)  # 用于选择几号GPu

    if args.dataset == 'opportunity_lc':
        args.input_size = 256
        args.input_channel = 45
        args.hheads = 9
        args.SENSOR_AXIS = 3
    elif args.dataset == 'seizure':
        args.input_channel = 18
        args.input_size = 256
        args.hheads = 6
        args.SENSOR_AXIS = 1
    elif args.dataset == 'wifi':
        args.input_channel = 90 #30
        args.input_size = 8000 #8500
        args.batch_size = 16
        args.hheads = 9
        args.SENSOR_AXIS = 3
    elif args.dataset == 'keti':
        args.input_channel = 4
        args.input_size = 256
        args.hheads = 4
        args.SENSOR_AXIS = 1
    args.model_save_path = os.path.join(args.log_path, args.model + '_' + args.config + '.pt')  # 模型保存路径为相应数据集下的一个路径
    return args, config


args, config1 = parse_args()
log = set_up_logging(args, config)
args.log = log  # args.log:此时已经不是路径了而是函数，其可向指定路径下写入内容




if __name__ == '__main__':
    batch_size = 16
    train_video_infos = get_video_info(config['dataset']['training']['video_info_path'])
    train_data_dict = load_video_data(train_video_infos,
                                      config['dataset']['training']['video_data_path'])
    train_video_annos = get_video_anno(train_video_infos,
                                       config['dataset']['training']['video_anno_path'])
    train_dataset = THUMOS_Dataset(train_data_dict,
                                   train_video_infos,
                                   train_video_annos)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                   num_workers=4, worker_init_fn=worker_init_fn,
                                   collate_fn=detection_collate, pin_memory=True, drop_last=True)
    epoch_step_num = len(train_dataset) // batch_size

    units_m = UniTS(input_size=args.input_size, sensor_num=args.input_channel, 
                  window_list=config1.window_list, stride_list=config1.stride_list, k_list=config1.k_list,
                   hidden_channel=config1.hidden_channel)
    units_m = units_m.to(device)
    with tqdm.tqdm(train_data_loader, total=epoch_step_num, ncols=0) as pbar:
        for n_iter, (clips, targets, scores, ssl_clips, ssl_targets, flags) in enumerate(pbar):
            print('clips.shape',clips.shape)  # [16, 1, 8500, 30, 1]
            clips = torch.squeeze(clips) 
            # clips = clips.type(torch.FloatTensor)
            clips = clips.cuda()
            # print('clips1.shape',clips.shape) # [16, 8500, 30]
            units_m.train()
            out = units_m(clips)
            print('out.shape',out.shape)
            