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
# from AFSD.common.thumos_dataset1 import get_video_info,data_process0
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
train_state_path = os.path.join(checkpoint_path, 'training')
pre_train_state_path = os.path.join(pre_checkpoint_path, 'training')
if not os.path.exists(train_state_path):
    os.makedirs(train_state_path)

resume = config['training']['resume']


def print_training_info():
    print('batch size: ', batch_size)
    print('learning rate: ', learning_rate)
    print('weight decay: ', weight_decay)
    print('max epoch: ', max_epoch)
    print('checkpoint path: ', checkpoint_path)
    print('loc weight: ', config['training']['lw'])
    print('cls weight: ', config['training']['cw'])
    print('ssl weight: ', config['training']['ssl'])
    print('piou:', config['training']['piou'])
    print('resume: ', resume)
    print('gpu num: ', ngpu)


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


def get_rng_states():
    states = []
    states.append(random.getstate())
    states.append(np.random.get_state())
    states.append(torch.get_rng_state())
    if torch.cuda.is_available():
        states.append(torch.cuda.get_rng_state())
    return states


def set_rng_state(states):
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(states[3])


def save_model(epoch, model, optimizer):
    # torch.save(model.module.state_dict(),
    #            os.path.join(checkpoint_path, 'checkpoint-{}.ckpt'.format(epoch)))
    torch.save(model.state_dict(),
               os.path.join(checkpoint_path, model_name+'-{}.ckpt'.format(epoch)))
    torch.save(model.state_dict(),
               os.path.join(checkpoint_path, model_name+'-{}.ckpt'.format(epoch)))
    torch.save({'optimizer': optimizer.state_dict(),
                'state': get_rng_states()},
               os.path.join(train_state_path, model_name+'_{}.ckpt'.format(epoch)))


def resume_training(resume, model, optimizer):
    start_epoch = 1
    if resume > 0:
        start_epoch += resume
        model_path = os.path.join(pre_checkpoint_path, pre_model_name+'-{}.ckpt'.format(resume))
        # model.module.load_state_dict(torch.load(model_path))
        # /root/tf-logs/models/fourier_num0-9-lw-10_lc-1-206.ckpt
        model.load_state_dict(torch.load(model_path))
        train_path = os.path.join(pre_train_state_path, pre_model_name+'_{}.ckpt'.format(resume))
        state_dict = torch.load(train_path)
        optimizer.load_state_dict(state_dict['optimizer'])
        set_rng_state(state_dict['state'])
        # '''----------------------------加载对照组模型-----------------------------------'''
        # model_path = os.path.join(checkpoint_path, 'checkpoint-60-lw-5_lc-{}.ckpt'.format(resume))
        # model.load_state_dict(torch.load(model_path),strict=False) # ,strict=False
        # '''--------------------------over-------------------------------'''
    return start_epoch


def calc_bce_loss(start, end, scores):
    start = torch.tanh(start).mean(-1)
    end = torch.tanh(end).mean(-1)
    # print('start',len(start.view(-1))) # len(start): 12000
    # print('scores[:, 1].contiguous()',len(scores[:, 1].contiguous().view(-1)))
    # scores = scores.numpy()
    # if 1>=scores.all() >= 0:
    #     print("0")
    # else:
    #     print("有不符合的")
    # # print('scores[:, 1].contiguous().view(-1)',scores[:, 1].contiguous().view(-1))
    # scores = torch.from_numpy(scores)
    loss_start = F.binary_cross_entropy(start.view(-1),
                                        scores[:, 0].contiguous().view(-1).cuda(),
                                        reduction='mean')
    loss_end = F.binary_cross_entropy(end.view(-1),
                                      scores[:, 1].contiguous().view(-1).cuda(),
                                      reduction='mean')
    # loss_start = F.binary_cross_entropy_with_logits(start.view(-1),
    #                                     scores[:, 0].contiguous().view(-1).cuda(),
    #                                     reduction='mean')
    # loss_end = F.binary_cross_entropy_with_logits(end.view(-1),
    #                                   scores[:, 1].contiguous().view(-1).cuda(),
    #                                   reduction='mean')
    return loss_start, loss_end


def forward_one_epoch(net, clips, targets, scores=None, training=True, ssl=False):
    clips = clips.cuda()
    # print('clips.shape:',clips.shape) # [2, 1, 256, 25, 30]
    targets = [t.cuda() for t in targets]
    # print('targets:',targets)
    if training:
        if ssl:
            # output_dict = net.module(clips, proposals=targets, ssl=ssl)
            output_dict = net(clips, proposals=targets, ssl=ssl)
            # print('you ssl')
        else:
            output_dict = net(clips, ssl=False)
            # print('wu ssl')
    else:
        with torch.no_grad():
            output_dict = net(clips)

    if ssl:
        anchor, positive, negative = output_dict
        loss_ = []
        weights = [1, 0.1, 0.1]
        for i in range(3):
            loss_.append(nn.TripletMarginLoss()(anchor[i], positive[i], negative[i]) * weights[i])
        trip_loss = torch.stack(loss_).sum(0)
        return trip_loss
    else:
        loss_l, loss_c, loss_prop_l, loss_prop_c, loss_ct = CPD_Loss(
            [output_dict['loc'], output_dict['conf'],
             output_dict['prop_loc'], output_dict['prop_conf'],
             output_dict['center'], output_dict['priors'][0]],
            targets)
        # print('scores[:, 0].contiguous().view(-1)',scores[:, 0].contiguous().view(-1).shape)
        # print('start.view(-1):',output_dict['start'].shape)
        loss_start, loss_end = calc_bce_loss(output_dict['start'], output_dict['end'], scores)
        scores_ = F.interpolate(scores, scale_factor=1.0 / 80) #  scale_factor=1.0 / 4
        loss_start_loc_prop, loss_end_loc_prop = calc_bce_loss(output_dict['start_loc_prop'],
                                                               output_dict['end_loc_prop'],
                                                               scores_)
        loss_start_conf_prop, loss_end_conf_prop = calc_bce_loss(output_dict['start_conf_prop'],
                                                                 output_dict['end_conf_prop'],
                                                                 scores_)
        loss_start = loss_start + 0.1 * (loss_start_loc_prop + loss_start_conf_prop)
        loss_end = loss_end + 0.1 * (loss_end_loc_prop + loss_end_conf_prop)
    return loss_l, loss_c, loss_prop_l, loss_prop_c, loss_ct, loss_start, loss_end


def run_one_epoch(epoch, net, optimizer, data_loader, epoch_step_num, training=True):
    if training:
        net.train()
    else:
        net.eval()

    loss_loc_val = 0
    loss_conf_val = 0
    loss_prop_l_val = 0
    loss_prop_c_val = 0
    loss_ct_val = 0
    loss_start_val = 0
    loss_end_val = 0
    loss_trip_val = 0
    loss_contras_val = 0
    cost_val = 0
    with tqdm.tqdm(data_loader, total=epoch_step_num, ncols=0) as pbar:
        for n_iter, (clips, targets, scores, ssl_clips, ssl_targets, flags) in enumerate(pbar):
            # print('clips.shape',clips.shape) #[1, 1, 8000, 9, 30]
            loss_l, loss_c, loss_prop_l, loss_prop_c,\
            loss_ct, loss_start, loss_end = forward_one_epoch(
                net, clips, targets, scores, training=training, ssl=False)
            loss_l = loss_l * config['training']['lw']
            loss_c = loss_c * config['training']['cw']
            loss_prop_l = loss_prop_l * config['training']['lw']
            loss_prop_c = loss_prop_c * config['training']['cw']
            loss_ct = loss_ct * config['training']['cw']
            cost = loss_l + loss_c + loss_prop_l + loss_prop_c + loss_ct + loss_start + loss_end

            ssl_count = 0
            loss_trip = 0
            for i in range(len(flags)):
                if flags[i] and config['training']['ssl'] > 0:
                    loss_trip += forward_one_epoch(net, ssl_clips[i].unsqueeze(0), [ssl_targets[i]],
                                                   training=training, ssl=True) * config['training']['ssl']
                    loss_trip_val += loss_trip.cpu().detach().numpy()
                    ssl_count += 1
            if ssl_count:
                loss_trip_val /= ssl_count
                loss_trip /= ssl_count
            cost = cost + loss_trip
            if training:
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()
            loss_loc_val += loss_l.cpu().detach().numpy()
            loss_conf_val += loss_c.cpu().detach().numpy()
            loss_prop_l_val += loss_prop_l.cpu().detach().numpy()
            loss_prop_c_val += loss_prop_c.cpu().detach().numpy()
            loss_ct_val += loss_ct.cpu().detach().numpy()
            loss_start_val += loss_start.cpu().detach().numpy()
            loss_end_val += loss_end.cpu().detach().numpy()
            cost_val += cost.cpu().detach().numpy()
            pbar.set_postfix(loss='{:.5f}'.format(float(cost.cpu().detach().numpy())))

    loss_loc_val /= (n_iter + 1)
    loss_conf_val /= (n_iter + 1)
    loss_prop_l_val /= (n_iter + 1)
    loss_prop_c_val /= (n_iter + 1)
    loss_ct_val /= (n_iter + 1)
    loss_start_val /= (n_iter + 1)
    loss_end_val /= (n_iter + 1)
    loss_trip_val /= (n_iter + 1)
    cost_val /= (n_iter + 1)
    print('本次迭代平均lose',cost_val)
    if training:
        prefix = 'Train'
        if epoch in save_model_num:
            save_model(epoch, net, optimizer)
    else:
        prefix = 'Val'

    writer.add_scalar('Total Loss', cost_val, epoch)
    writer.add_scalar('loc', loss_loc_val, epoch)
    writer.add_scalar('conf', loss_conf_val, epoch)



if __name__ == '__main__':
    print_training_info()
    set_seed(random_seed)
    ngpu = 1
    test_model_num = 380
    """
    Setup model 
    """
    net = BDNet(in_channels=1,
                backbone_model=config['model']['backbone_model'],
                training=True)
    # for para in net.backbone.parameters():
    #     para.requires_grad = False

    # net = nn.DataParallel(net, device_ids=list(range(ngpu))).cuda()
    net = net.cuda()

    # for k, v in net.named_parameters():
    #     print('{}: {}'.format(k, v.requires_grad))

    """
    Setup optimizer
    """
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    """
    Setup loss
    """
    piou = config['training']['piou']
    CPD_Loss = MultiSegmentLoss(num_classes, piou, 1.0, use_focal_loss=focal_loss)

    """
    Setup dataloader
    """
   
    train_video_infos = get_video_info(config['dataset']['training']['video_info_path'])
    train_video_annos = get_video_anno(train_video_infos,
                                       config['dataset']['training']['video_anno_path'])
    train_data_dict = load_video_data(train_video_infos,
                                      config['dataset']['training']['video_data_path'])
    train_dataset = THUMOS_Dataset(train_data_dict,
                                   train_video_infos,
                                   train_video_annos)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                   num_workers=4, worker_init_fn=worker_init_fn,
                                   collate_fn=detection_collate, pin_memory=True, drop_last=True)
    print('len(train_dataset)',len(train_dataset))
    epoch_step_num = len(train_dataset) // batch_size
    
    """
    Start training
    # """
    start_epoch = resume_training(resume, net, optimizer)

    for i in range(start_epoch, max_epoch + 1):
        print('train_epoch:',i)
        start = time.perf_counter()
        run_one_epoch(i, net, optimizer, train_data_loader, len(train_dataset) // batch_size)
        end = time.perf_counter()
        print('train_epoch:',i,'use-time:',str(end-start))

    

