import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from AFSD.common.thumos_dataset import THUMOS_Dataset, get_video_info, \
    load_video_data, detection_collate, get_video_anno
from torch.utils.data import DataLoader
from AFSD.thumos14.multisegment_loss import MultiSegmentLoss
from AFSD.common.config import config
import numpy as np
import tqdm
import json
from AFSD.common import videotransforms
from AFSD.common.thumos_dataset import get_class_index_map
from AFSD.thumos14.BDNet import BDNet
from AFSD.common.segment_utils import softnms_v2 ,soft_nms
import argparse
from AFSD.evaluation.eval_detection import ANETdetection
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader,TensorDataset
import time
batch_size = config['training']['batch_size']
learning_rate = config['training']['learning_rate']
weight_decay = config['training']['weight_decay']
max_epoch = config['training']['max_epoch']
num_classes = config['dataset']['num_classes']
checkpoint_path = config['training']['checkpoint_path']
pre_checkpoint_path = config['training']['pre_checkpoint_path']
focal_loss = config['training']['focal_loss']
random_seed = config['training']['random_seed']
ngpu = config['ngpu']
model_name = config['training']['model_name']
pre_model_name = config['training']['pre_model_name']
test_model_name = config['testing']['test_model_name']
save_model_num = config['training']['save_model_num']
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
writer = SummaryWriter("runs/scalar_example")
if __name__ == '__main__':
    """
    Start testing
    该测试程序，对测试数据没有使用RGB归一化，所以在训练时也不能使用RGB归一化
    """
    num_classes = config['dataset']['num_classes']
    conf_thresh = config['testing']['conf_thresh']
    top_k = config['testing']['top_k']
    nms_thresh = config['testing']['nms_thresh']
    nms_sigma = config['testing']['nms_sigma']
    clip_length = config['dataset']['testing']['clip_length']
    stride = config['dataset']['testing']['clip_stride']
    max_epoch = config['training']['max_epoch']
    checkpoint_path = config['testing']['checkpoint_path']
    json_name = config['testing']['output_json']
    output_path = config['testing']['output_path']
    softmax_func = True
    test_num = 400
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    fusion = config['testing']['fusion']
    # /root/tf-logs/models/Amend_num0-9_i3d_keystroke/fourier_num0-9-lw-10_lc-1-206.ckpt
    # for i in range(max_epoch-4, max_epoch+1): # max_epoch-4, max_epoch+1
    for i in save_model_num:
        print(i)
        checkpoint_path = os.path.join(checkpoint_path, test_model_name + "-"+str(i)+".ckpt")
        # checkpoint_path = '/root/tf-logs/models/All-Amend_num0-9_Data60/All-Amend_num0-9_Data60-lw-6_lc-3-420.ckpt'
        print('checkpoint_path',checkpoint_path)
        json_name = str(i)+json_name
        video_infos = get_video_info(config['dataset']['testing']['video_info_path'])
        video_annos = get_video_anno(video_infos,
                                       config['dataset']['testing']['video_anno_path'])
        originidx_to_idx, idx_to_class = get_class_index_map("STADe-DeepSeg/thumos_annotations/Class Index_Detection.txt")
        # print('idx_to_class',idx_to_class)
        npy_data_path = config['dataset']['testing']['video_data_path']

        net = BDNet(in_channels=config['model']['in_channels'],
                training=False)
        # print("checkpoint_path:",checkpoint_path)
        # net = nn.DataParallel(net, device_ids=list(range(ngpu))).cuda()
        net.load_state_dict(torch.load(checkpoint_path)) # ,strict=False
        checkpoint_path = config['testing']['checkpoint_path'] 
        net.eval().cuda()

        if softmax_func:
            score_func = nn.Softmax(dim=-1)
        else:
            score_func = nn.Sigmoid()

        centor_crop = videotransforms.CenterCrop(config['dataset']['testing']['crop_size'])

        result_dict = {}
        for video_name in tqdm.tqdm(list(video_infos.keys()), ncols=0):
            sample_count = video_infos[video_name]['sample_count']
            sample_fps = video_infos[video_name]['sample_fps']
            sample_gt = video_annos[video_name]
            # print('sample_gt:',sample_gt)
            if sample_count < clip_length:
                offsetlist = [0]
            else:
                offsetlist = list(range(0, sample_count - clip_length + 1, stride))
                if (sample_count - clip_length) % stride:
                    offsetlist += [sample_count - clip_length]
            
            data = np.load(os.path.join(npy_data_path, video_name + '.npy'))     # 加载不同的数据集要修改这里的格式
            data = data.transpose(2,0,1)
            data = data[:8000,:,:]
            T,R,C = data.shape
            data_pad = np.zeros((8000,3,30)) 
            data_pad[:T,:R,:C]=data
            # data = np.reshape(data, (len(data), -1,30), order='c')
            data = np.reshape(data_pad, (len(data_pad), -1), order='c')
            # data = np.expand_dims(data, 0).repeat(1, axis=0)
            data = torch.from_numpy(data)
            data = data.type(torch.FloatTensor)
            print('data.shape',data.shape) # [1, 8000, 9, 30]
            output = []
            for cl in range(num_classes):
                output.append([])
            res = torch.zeros(num_classes, top_k, 3)
            clip = data.unsqueeze(0).cuda()
            with torch.no_grad():
                # print('clip',clip.shape)
                # print('Test',clip.shape)
                output_dict = net(clip)

            loc, conf, priors = output_dict['loc'], output_dict['conf'], output_dict['priors'][0]
            prop_loc, prop_conf = output_dict['prop_loc'], output_dict['prop_conf']
            center = output_dict['center']
            loc = loc[0]
            conf = conf[0]
            prop_loc = prop_loc[0]
            prop_conf = prop_conf[0]
            center = center[0]

            pre_loc_w = loc[:, :1] + loc[:, 1:]
            loc = 0.5 * pre_loc_w * prop_loc + loc
            decoded_segments = torch.cat(
                [priors[:, :1] * clip_length - loc[:, :1],
                    priors[:, :1] * clip_length + loc[:, 1:]], dim=-1)
            decoded_segments.clamp_(min=0, max=clip_length)
            # print('decoded_segments[0:10,:]:',decoded_segments.shape)  # [3938, 2]
            conf = score_func(conf)
            # print("conf0[0:10,:]",conf[0:10,:])
            prop_conf = score_func(prop_conf)
            # print('conf1',prop_conf)
            center = center.sigmoid()
            # print("center[0:10,:]",center[0:10,:]) # [2954, 1]
            conf = (conf + prop_conf) / 2.0
            # print("conf1[0:10,:]",conf[0:10,:])
            conf = conf * center
            # print("conf2[:,0:10]",conf.shape)    # [3938, 11]
            conf = conf.view(-1, num_classes).transpose(1, 0)
            conf_scores = conf.clone()

            for cl in range(1, num_classes):
                c_mask = conf_scores[cl] > conf_thresh
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    # print('bad')
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_segments)
                # print('l_mask',l_mask.shape)  # [3938, 2]
                segments = decoded_segments[l_mask].view(-1, 2)
                # decode to original time
                # segments = (segments * clip_length + offset) / sample_fps
                offset = 0
                segments = (segments + offset) / 1 # sample_fps
                segments = torch.cat([segments, scores.unsqueeze(1)], -1)
                output[cl].append(segments)
                # np.set_printoptions(precision=3, suppress=True)
            sum_count = 0
            for cl in range(1, num_classes):
                if len(output[cl]) == 0:
                    continue
                tmp = torch.cat(output[cl], 0)
                # print("tmp.shape",tmp.shape) 
                tmp, count = soft_nms(tmp, sigma=nms_sigma, top_k=top_k) # soft_nms(segments, overlap=0.3, sigma=0.5, top_k=1000):
                # print("nms_sigma",nms_sigma)
                # print("\nclass",cl,"tmp1",tmp.shape,"\ntmp1",tmp)
                res[cl, :count] = tmp
                sum_count += count

            sum_count = min(sum_count, top_k)
            flt = res.contiguous().view(-1, 3)
            flt = flt.view(num_classes, -1, 3)
            proposal_list = []
            for cl in range(1, num_classes):
                class_name = idx_to_class[cl]  # 通过键值来对类别名称进行索引 # {1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 10: '9'}
                # print('class_name',type(class_name))
                tmp = flt[cl].contiguous()
                tmp = tmp[(tmp[:, 2] > 0).unsqueeze(-1).expand_as(tmp)].view(-1, 3)
                if tmp.size(0) == 0:
                    continue
                tmp = tmp.detach().cpu().numpy()
                for i in range(tmp.shape[0]):
                    tmp_proposal = {}
                    tmp_proposal['label'] = class_name
                    tmp_proposal['score'] = float(tmp[i, 2])
                    tmp_proposal['segment'] = [float(tmp[i, 0]),
                                                float(tmp[i, 1])]
                    proposal_list.append(tmp_proposal)

            result_dict[video_name] = proposal_list
            # print("len(proposal_list)",len(proposal_list))
        output_dict = {"version": "THUMOS14", "results": dict(result_dict), "external_data": {}}

        with open(os.path.join(output_path, json_name), "w") as out:
            json.dump(output_dict, out)
        json_name = config['testing']['output_json']
    writer.close()

    """
    Start evaluating
    """
    writer = SummaryWriter("runs/eval")
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    max_epoch = config['training']['max_epoch']
    max = 0
    max_i = 0
    x = []
    y = []
    for i in save_model_num:   # max_epoch-4, max_epoch + 1
        
        gt_json = 'STADe-DeepSeg/thumos_annotations/Behave_Public_25.json'
        output_json = os.path.join(output_path, str(i) + json_name)        
        tious = [0.3, 0.4, 0.5, 0.6, 0.7]
        anet_detection = ANETdetection(
            ground_truth_filename=gt_json,
            prediction_filename=output_json,
            subset='test', tiou_thresholds=tious)
        mAPs, average_mAP, ap = anet_detection.evaluate()
        # print(mAPs, "\n", average_mAP, "\n", ap)
        print("epoch", i)
        for (tiou, mAP) in zip(tious, mAPs):
            print("mAP at tIoU {} is {}".format(tiou, mAP))
        print(average_mAP, "\n")

        if average_mAP > max:
            max = average_mAP
            max_i = i

        writer.add_scalar('average_mAP', round(average_mAP, 4), i)
        x.append(i)
        y.append(round(average_mAP, 4))

 