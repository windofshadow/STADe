import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from AFSD.common.utils import *
from AFSD.common.i3d_backbone import InceptionI3d
from AFSD.common.config import config
from AFSD.common.layers import Unit1D, Unit3D
from AFSD.prop_pooling.boundary_pooling_op import BoundaryMaxPooling, BoundaryConPooling0, BoundaryConPooling1, BoundaryConPooling2, BoundaryConPooling3, BoundaryConPooling4, BoundaryConPooling5, BoundaryConPooling6
from Fourier import UniTS,args, config1
import argparse

num_classes = config['dataset']['num_classes']
freeze_bn = config['model']['freeze_bn']
freeze_bn_affine = config['model']['freeze_bn_affine']
ConfDim_num = config['model']['ConfDim_num']
LocDim_num = config['model']['LocDim_num']
ConPoolingKernal = config['model']['ConPoolingKernal']
frame_num = config['dataset']['training']['clip_length']
backbone_model=config['model']['backbone_model']

layer_num = 6
conv_channels = 512
feat_t = 256 // 4


class I3D_BackBone(nn.Module):
    def  __init__(self, final_endpoint='Mixed_5c', name='inception_i3d', in_channels=1,
                 freeze_bn=freeze_bn, freeze_bn_affine=freeze_bn_affine):
        super(I3D_BackBone, self).__init__()
        self._model = InceptionI3d(final_endpoint=final_endpoint,
                                   name=name,
                                   in_channels=in_channels)
        self._model.build()
        self._freeze_bn = freeze_bn
        self._freeze_bn_affine = freeze_bn_affine

    def load_pretrained_weight(self, model_path='/root/tf-logs/Amend_Backbone_I3D-num0-9/models/thumos14/checkpoint-15.ckpt'):
        folder_path = ""  # 相对路径
        absolute_path = os.path.abspath(folder_path)
        path_model_path=backbone_model
        absolute_path_path_model_path = os.path.join(absolute_path, path_model_path)
        self._model.load_state_dict(torch.load(absolute_path_path_model_path), strict=False)

    def train(self, mode=True):
    # def train(self, mode=False):
        super(I3D_BackBone, self).train(mode)
        if self._freeze_bn and mode:
            print('freeze all BatchNorm3d in I3D backbone.')
            for name, m in self._model.named_modules():
                if isinstance(m, nn.BatchNorm3d):
                    # print('freeze {}.'.format(name))
                    m.eval()
                    if self._freeze_bn_affine:
                        m.weight.requires_grad_(False)
                        m.bias.requires_grad_(False)

    def forward(self, x):
        return self._model.extract_features(x)


class ScaleExp(nn.Module):
    def __init__(self, init_value=1.0):
        super(ScaleExp, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return torch.exp(input * self.scale)



class ProposalBranch_conf(nn.Module):
    def __init__(self, in_channels, proposal_channels,kernel):
        super(ProposalBranch_conf, self).__init__()
        self.cur_point_conv = nn.Sequential(
            Unit1D(in_channels=in_channels,
                   output_channels=proposal_channels,
                   kernel_shape=1,
                   activation_fn=None),
            nn.GroupNorm(32, proposal_channels),
            nn.ReLU(inplace=True)
        )
        self.lr_conv = nn.Sequential(
            Unit1D(in_channels=in_channels,
                   output_channels=proposal_channels * 2,
                   kernel_shape=1,
                   activation_fn=None),
            nn.GroupNorm(32, proposal_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.kernel = kernel
        self.boundary_max_pooling = BoundaryMaxPooling()
        '''-------------------------------------------锚框内容卷积的初始化--------------------------------------------------'''
        self.BoundaryConPooling0 = BoundaryConPooling0(in_channels=proposal_channels ,proposal_channels=proposal_channels,kernel=kernel[0],padding='no') 
        self.BoundaryConPooling1 = BoundaryConPooling1(in_channels=proposal_channels * 2,proposal_channels=proposal_channels,kernel=kernel[1],padding='no')
        self.BoundaryConPooling2 = BoundaryConPooling2(in_channels=proposal_channels * 2,proposal_channels=proposal_channels,kernel=kernel[2],padding='no')
        self.BoundaryConPooling3 = BoundaryConPooling3(in_channels=proposal_channels * 2,proposal_channels=proposal_channels,kernel=kernel[3],padding='no')
        self.BoundaryConPooling4 = BoundaryConPooling4(in_channels=proposal_channels * 2,proposal_channels=proposal_channels,kernel=kernel[4],padding='no')
        self.BoundaryConPooling5 = BoundaryConPooling5(in_channels=proposal_channels * 2,proposal_channels=proposal_channels,kernel=kernel[5],padding='no')
        self.BoundaryConPooling6 = BoundaryConPooling6(in_channels=proposal_channels * 2,proposal_channels=proposal_channels,kernel=kernel[6],padding='no')


        '''-----------------------------------------------over-------------------------------------------------------------'''
        # self.BoundaryConPooling = BoundaryConPooling(in_channels=proposal_channels * 2,proposal_channels=proposal_channels) 
        self.roi_conv = nn.Sequential(
            Unit1D(in_channels=proposal_channels,
                   output_channels=proposal_channels,
                   kernel_shape=1,
                   activation_fn=None),
            nn.GroupNorm(32, proposal_channels),
            nn.ReLU(inplace=True)
        )
        self.proposal_conv = nn.Sequential(
            Unit1D(
                in_channels=proposal_channels * 6+500,
                output_channels=in_channels,
                kernel_shape=1,
                activation_fn=None
            ),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feature, frame_level_feature, segments, frame_segments,feature_lastclass,anchor,frame_anchor,order,LocResult_feature):
        fm_short = self.cur_point_conv(feature)
        feature = self.lr_conv(feature)
        prop_feature = self.boundary_max_pooling(feature, segments)
        prop_roi_feature = self.boundary_max_pooling(frame_level_feature, frame_segments)
        prop_roi_feature = self.roi_conv(prop_roi_feature)
       
         # '''----------------------------对预测出的锚框中的数据进行卷积提取特征-----------------------------'''
        # print('juanji anchor',anchor,anchor.shape,'juanji frame_anchor',frame_anchor.shape) # anchor.shape = [2, 2000, 2]
        L1 = len(anchor[0,:,0]) # 
        L2 = len(frame_level_feature[0,0,:]) # 8000
        # print('L1',L1,'L2',L2)
        a1 = np.ones([1, 512])
        b1 = torch.from_numpy(a1).cuda()

        ConFeature = [] # 用于存放卷积特征
        Frame_confeature = [] # 用于存放帧水平的卷积特征
        Kernals=self.kernel
        for i in range(L1):
            Data1 = []
            Data2 = []
            # if hasattr(torch.cuda, 'empty_cache'):
            #  torch.cuda.empty_cache()
            # if hasattr(torch.cuda, 'empty_cache'):
	        #     torch.cuda.empty_cache()
            for j in range(anchor.shape[0]):
                a,b,c = feature.shape
                A,B,C = frame_level_feature.shape
                left1 = torch.clamp(anchor[j,i,0],0,L1)
                right1 = torch.clamp(anchor[j,i,1],0,L1)
                if right1 <=left1:
                    data1 = torch.from_numpy(np.zeros([b,Kernals[order+1]]))
                    data1 = data1.type(torch.FloatTensor).cuda()
                else:
                    # print('left1',left1,'right1',right1)
                    data1 = feature[j,:,int(left1):int(right1)+1]
                left2 = torch.clamp(frame_anchor[j,i,0],0,L2)
                right2 = torch.clamp(frame_anchor[j,i,1],0,L2)
                if right2 <=left2:
                    data2 = torch.from_numpy(np.zeros([B,Kernals[0]]))
                    data2 = data2.type(torch.FloatTensor).cuda()

                else:
                    data2 = frame_level_feature[j,:,int(left2):int(right2)+1]
                # print('data1.shape',data1.shape)
                data1 =data1.unsqueeze(0)
                # print('data1.shape',data1.shape)
                data2 =data2.unsqueeze(0) # [1, 1024, 1]
                # print('data2.shape',data2.shape)
                data1 = F.interpolate(data1, size=Kernals[order+1])
                # print('data1.shape',data1.shape) # [1, 1024, 200]
                data2 = F.interpolate(data2, size=Kernals[0])
                # print('data2.shape',data2.shape)
                data1 =data1.squeeze(0)
                # print('data1.shape',data1.shape) # [1024, 200]
                data2 =data2.squeeze(0)
                # print('data2.shape',data2.shape) # [512, 800]
                Data1.append(data1)
                Data2.append(data2)
            Data1 = torch.stack(Data1,dim = 0)
            Data2 = torch.stack(Data2,dim = 0)
            # print('Data1.shape',Data1.shape) # [2, 1024, 200]
            # print('Data2.shape',Data2.shape) # [2, 512, 800]
            if order == 0:
                con_feature = self.BoundaryConPooling1(Data1)
            if order == 1:
                con_feature = self.BoundaryConPooling2(Data1)
            if order == 2:
                con_feature = self.BoundaryConPooling3(Data1)
            if order == 3:
                con_feature = self.BoundaryConPooling4(Data1)
            if order == 4:
                con_feature = self.BoundaryConPooling5(Data1)
            if order == 5:
                con_feature = self.BoundaryConPooling6(Data1)
            prop_Con_feature = self.BoundaryConPooling0(Data2) # [1, 512, 1]
            con_feature = con_feature.squeeze(-1)
            prop_Con_feature = prop_Con_feature.squeeze(-1)
            # with torch.no_grad():
            # b1 = torch.cat([b1,prop_Con_feature],dim = 0)
            ConFeature.append(con_feature)
            Frame_confeature.append(prop_Con_feature)
            # torch.cuda.empty_cache()
            # torch.backends.cudnn.enabled = True
            # torch.backends.cudnn.benchmark = True
            # print('b1.shape:',b1.shape)
        ConFeature = torch.stack(ConFeature,dim = 2)
        Frame_confeature = torch.stack(Frame_confeature,dim = 2)
        # print('ConFeature.shape:',ConFeature.shape,'Frame_confeature.shape:',Frame_confeature.shape)
        prop_Con_feature = self.roi_conv(Frame_confeature)
        # print('最大池化前feature.shape',feature.shape,'hou:',prop_feature.shape) # 最大池化前feature.shape torch.Size([2, 1024, 63]) hou: torch.Size([2, 1024, 63])
        # print('最大池化前frame_level_feature',frame_level_feature.shape,'hou:',frame_level_feature.shape) # 最大池化前frame_level_feature torch.Size([2, 512, 8000]) hou: torch.Size([2, 512, 8000])
        '''------------------------------------over-----------------------------------------'''
        prop_feature = torch.cat([prop_roi_feature, prop_feature, fm_short,feature_lastclass,ConFeature,prop_Con_feature,LocResult_feature], dim=1)
        prop_feature = self.proposal_conv(prop_feature)
        return prop_feature, feature

class ProposalBranch_loc(nn.Module):
    def __init__(self, in_channels, proposal_channels):
        super(ProposalBranch_loc, self).__init__()
        self.cur_point_conv = nn.Sequential(
            Unit1D(in_channels=in_channels,
                   output_channels=proposal_channels,
                   kernel_shape=1,
                   activation_fn=None),
            nn.GroupNorm(32, proposal_channels),
            nn.ReLU(inplace=True)
        )
        self.lr_conv = nn.Sequential(
            Unit1D(in_channels=in_channels,
                   output_channels=proposal_channels * 2,
                   kernel_shape=1,
                   activation_fn=None),
            nn.GroupNorm(32, proposal_channels * 2),
            nn.ReLU(inplace=True)
        )

        self.boundary_max_pooling = BoundaryMaxPooling()
        # self.BoundaryConPooling = BoundaryConPooling(in_channels=proposal_channels * 2,proposal_channels=proposal_channels) 
        self.roi_conv = nn.Sequential(
            Unit1D(in_channels=proposal_channels,
                   output_channels=proposal_channels,
                   kernel_shape=1,
                   activation_fn=None),
            nn.GroupNorm(32, proposal_channels),
            nn.ReLU(inplace=True)
        )
        self.proposal_conv = nn.Sequential(
            Unit1D(
                in_channels=proposal_channels * 4+500,
                output_channels=in_channels,
                kernel_shape=1,
                activation_fn=None
            ),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feature, frame_level_feature, segments, frame_segments,feature_lastclass,ConfResult_feature):
        fm_short = self.cur_point_conv(feature)
        feature = self.lr_conv(feature)
        prop_feature = self.boundary_max_pooling(feature, segments)
        prop_roi_feature = self.boundary_max_pooling(frame_level_feature, frame_segments)
        prop_roi_feature = self.roi_conv(prop_roi_feature)
        '''-----------------------对边界范围的数据进行卷积提取特征-----------------------------'''
        # Con_feature = self.BoundaryConPooling(feature, anchor) # [2, 2000, 4]
        # prop_Con_feature = self.BoundaryConPooling(frame_level_feature, frame_anchor)
        # prop_Con_feature = self.roi_conv(prop_Con_feature)
        # print('最大池化前feature.shape',feature.shape,'hou:',prop_feature.shape) # 最大池化前feature.shape torch.Size([2, 1024, 63]) hou: torch.Size([2, 1024, 63])
        # print('prop_roi_feature',prop_roi_feature.shape,'prop_feature:',prop_feature.shape,'fm_short',fm_short.shape) # prop_roi_feature torch.Size([2, 512, 23]) prop_feature: torch.Size([2, 1024, 23]) fm_short torch.Size([2, 512, 23])
        '''------------------------------------over-----------------------------------------'''
        prop_feature = torch.cat([prop_roi_feature, prop_feature, fm_short,feature_lastclass,ConfResult_feature], dim=1)
        prop_feature = self.proposal_conv(prop_feature)
        return prop_feature, feature

class CoarsePyramid(nn.Module):
    def __init__(self, feat_channels, frame_num,ConPoolingKernal): # 118为视频帧的长度
        super(CoarsePyramid, self).__init__()
        out_channels = conv_channels
        self.pyramids = nn.ModuleList()
        self.loc_heads = nn.ModuleList()
        self.frame_num = frame_num
        self.layer_num = layer_num
        self.pyramids.append(nn.Sequential(
            Unit3D(
                in_channels=feat_channels[0],
                output_channels=out_channels,
                kernel_shape=[1, 1, 30],
                padding='spatial_valid',
                use_batch_norm=False,
                use_bias=True,
                activation_fn=None
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        ))

        self.pyramids.append(nn.Sequential(
            Unit3D(
                in_channels=feat_channels[1],
                output_channels=out_channels,
                kernel_shape=[1, 1, 15],
                use_batch_norm=False,
                padding='spatial_valid',
                use_bias=True,
                activation_fn=None
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        ))
        for i in range(2, layer_num):
            self.pyramids.append(nn.Sequential(
                Unit1D(
                    in_channels=out_channels,
                    output_channels=out_channels,
                    kernel_shape=3,
                    stride=2,
                    use_bias=True,
                    activation_fn=None
                ),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True)
            ))

        loc_towers = []
        for i in range(2):
            loc_towers.append(
                nn.Sequential(
                    Unit1D(
                        in_channels=out_channels,
                        output_channels=out_channels,
                        kernel_shape=3,
                        stride=1,
                        use_bias=True,
                        activation_fn=None
                    ),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        self.loc_tower = nn.Sequential(*loc_towers)
        conf_towers = []
        for i in range(2):
            conf_towers.append(
                nn.Sequential(
                    Unit1D(
                        in_channels=out_channels,
                        output_channels=out_channels,
                        kernel_shape=3,
                        stride=1,
                        use_bias=True,
                        activation_fn=None
                    ),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        self.conf_tower = nn.Sequential(*conf_towers)

        self.loc_head = Unit1D(
            in_channels=out_channels,
            output_channels=2,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None
        )
        self.conf_head = Unit1D(
            in_channels=out_channels,
            output_channels=num_classes,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None
        )

        # self.loc_proposal_branch = ProposalBranch(out_channels, 512)
        # self.conf_proposal_branch = ProposalBranch(out_channels, 512)
        self.loc_proposal_branch = ProposalBranch_loc(out_channels, 512)
        self.conf_proposal_branch = ProposalBranch_conf(out_channels, 512, kernel=ConPoolingKernal)
        # '''------------------------------------------------分类和回归特征的形成-----------------------------------------------------------------'''
        self.conf_linear = nn.Linear(in_features = num_classes, out_features = ConfDim_num) # bias = ture  
        self.loc_linear = nn.Linear(in_features = 2, out_features = LocDim_num)
        self.relu = nn.ReLU()

        # '''-----------------------------------------------------------OVER---------------------------------------------------------------'''
        self.prop_loc_head = Unit1D(
            in_channels=out_channels,
            output_channels=2,
            kernel_shape=1,
            activation_fn=None
        )
        self.prop_conf_head = Unit1D(
            in_channels=out_channels,
            output_channels=num_classes,
            kernel_shape=1,
            activation_fn=None
        )

        self.center_head = Unit1D(
            in_channels=out_channels,
            output_channels=1,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None
        )

        self.deconv = nn.Sequential(
            Unit1D(out_channels, out_channels, 3, activation_fn=None),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            Unit1D(out_channels, out_channels, 3, activation_fn=None),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            Unit1D(out_channels, out_channels, 1, activation_fn=None),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )

        self.priors = []
        t = feat_t
        for i in range(layer_num):
            self.loc_heads.append(ScaleExp())
            if i == 0 : # 击键识别有这个，该层长度有点出入，调整一下
                t=100
            if i == 1 :
                t = 50
            if i == 2 :
                t = 25
            if i == 3 :
                t = 13
            if i == 4 :
                t=7
            if i == 5 :
                t=4
            self.priors.append(
                torch.Tensor([[(c + 0.5) / t] for c in range(t)]).view(-1, 1)
            )
            # if t % 2 != 0:
            #     t = t // 2 + 1
            # else:
            #     t = t // 2

    def forward(self, feat_dict, ssl=False):
        pyramid_feats = []
        locs = []
        confs = []
        centers = []
        prop_locs = []
        prop_confs = []
        trip = []
        x2 = feat_dict['Mixed_5c']
        x1 = feat_dict['Mixed_4f']
        batch_num = x1.size(0)
        for i, conv in enumerate(self.pyramids):
            if i == 0:
                # print('x01.shape',x1.shape)
                x = conv(x1)
                # print('x0.shape',x.shape) # [1, 512, 50, 1, 1]
                x = x.squeeze(-1).squeeze(-1)
            elif i == 1:
                # print('x02.shape',x2.shape)
                x = conv(x2)
                # print('x1.shape',x.shape) # [1, 512, 25, 1, 1]
                x = x.squeeze(-1).squeeze(-1)
                x0 = pyramid_feats[-1]
                y = F.interpolate(x, x0.size()[2:], mode='nearest')
                pyramid_feats[-1] = x0 + y
            else:
                x = conv(x)    # [2, 512, 23, 1, 15]
            pyramid_feats.append(x)

        frame_level_feat = pyramid_feats[0].unsqueeze(-1)
        frame_level_feat = F.interpolate(frame_level_feat, [self.frame_num, 1]).squeeze(-1)
        frame_level_feat = self.deconv(frame_level_feat)
        trip.append(frame_level_feat.clone())
        start_feat = frame_level_feat[:, :256]
        end_feat = frame_level_feat[:, 256:]
        start = start_feat.permute(0, 2, 1).contiguous()
        end = end_feat.permute(0, 2, 1).contiguous()

        for i, feat in enumerate(pyramid_feats):
            loc_feat = self.loc_tower(feat)
            conf_feat = self.conf_tower(feat)

            loc_data =  self.loc_heads[i](self.loc_head(loc_feat)).view(batch_num, 2, -1).permute(0, 2, 1).contiguous()
                    
                    
            locs.append( loc_data)
            conf_data = self.conf_head(conf_feat).view(batch_num, num_classes, -1).permute(0, 2, 1).contiguous()
                    
            confs.append(conf_data ) 
            t = feat.size(2)
            # print('t',t)
            with torch.no_grad():
                segments = locs[-1] / self.frame_num * t
                # print('self.priors[i]:',self.priors[i].shape)
                priors = self.priors[i].expand(batch_num, t, 1).to(feat.device)
                new_priors = torch.round(priors * t - 0.5)
                plen = segments[:, :, :1] + segments[:, :, 1:]
                in_plen = torch.clamp(plen / 4.0, min=1.0)
                out_plen = torch.clamp(plen / 10.0, min=1.0)

                l_segment = new_priors - segments[:, :, :1]
                r_segment = new_priors + segments[:, :, 1:]
                anchor = torch.cat([l_segment,r_segment],dim=-1)
                segments = torch.cat([
                    torch.round(l_segment - out_plen),
                    torch.round(l_segment + in_plen),
                    torch.round(r_segment - in_plen),
                    torch.round(r_segment + out_plen)
                ], dim=-1)

                decoded_segments = torch.cat(
                    [priors[:, :, :1] * self.frame_num - locs[-1][:, :, :1],
                     priors[:, :, :1] * self.frame_num + locs[-1][:, :, 1:]],
                    dim=-1)
                frame_anchor = decoded_segments
                plen = decoded_segments[:, :, 1:] - decoded_segments[:, :, :1] + 1.0
                in_plen = torch.clamp(plen / 4.0, min=1.0)
                out_plen = torch.clamp(plen / 10.0, min=1.0)
                frame_segments = torch.cat([
                    torch.round(decoded_segments[:, :, :1] - out_plen),
                    torch.round(decoded_segments[:, :, :1] + in_plen),
                    torch.round(decoded_segments[:, :, 1:] - in_plen),
                    torch.round(decoded_segments[:, :, 1:] + out_plen)
                ], dim=-1)
            '''-------------------------------------分类或回归结果转成特征的第一方法----------------'''
            # print('locs:',loc_data.shape,'confs.shape',conf_data.shape)      # locs: torch.Size([2, 23, 2]) confs.shape torch.Size([2, 23, 8])
            ConfResult_feature = self.conf_linear(conf_data).permute(0, 2, 1).contiguous()
            LocResult_feature = self.loc_linear(loc_data).permute(0, 2, 1).contiguous()
            ConfResult_feature = self.relu(ConfResult_feature)
            LocResult_feature = self.relu(LocResult_feature)
            # print('ConfResult_feature:',ConfResult_feature.shape,'LocResult_feature.shape',LocResult_feature.shape) # ConfResult_feature: torch.Size([2, 10, 23]) LocResult_feature.shape torch.Size([2, 10, 23])
            '''------------------------------------------------------over------------------------------------------'''
            '''-------------------------------------将前一个动作的种类及回归作为特征送到下一个动作中作预测----------------'''
            # print('locs:',loc_data.shape,'confs.shape',conf_data.shape)      # locs: torch.Size([2, 23, 2]) confs.shape torch.Size([2, 23, 8])
            conf_data1 = torch.zeros_like(conf_data).cuda()
            Lconf = len(conf_data[0,:,0])-1
            # print('conf_data1',conf_data1.shape,'Lconf',Lconf)
            conf_data1[:,1:,:] = conf_data[:,:Lconf,:]
            
            loc_data1 = torch.zeros_like(loc_data).cuda()
            loc_data1[:,1:,:] = loc_data[:,:Lconf,:]
    
            ConfResult_feature1 = self.conf_linear(conf_data1).permute(0, 2, 1).contiguous()
            LocResult_feature1 = self.loc_linear(loc_data1).permute(0, 2, 1).contiguous()
            ConfResult_feature1 = self.relu(ConfResult_feature1)
            LocResult_feature1 = self.relu(LocResult_feature1)
            # print('ConfResult_feature:',ConfResult_feature.shape,'LocResult_feature.shape',LocResult_feature.shape) # ConfResult_feature: torch.Size([2, 10, 23]) LocResult_feature.shape torch.Size([2, 10, 23])
            '''------------------------------------------------------over------------------------------------------'''
            loc_prop_feat, loc_prop_feat_ = self.loc_proposal_branch(loc_feat, frame_level_feat,
                                                                     segments, frame_segments,LocResult_feature1,ConfResult_feature)
            conf_prop_feat, conf_prop_feat_ = self.conf_proposal_branch(conf_feat, frame_level_feat,
                                                                        segments, frame_segments,ConfResult_feature1,anchor,frame_anchor,order=i,LocResult_feature=LocResult_feature)
            if i == 0:
                trip.extend([loc_prop_feat_.clone(), conf_prop_feat_.clone()])
                ndim = loc_prop_feat_.size(1) // 2
                start_loc_prop = loc_prop_feat_[:, :ndim, ].permute(0, 2, 1).contiguous()
                end_loc_prop = loc_prop_feat_[:, ndim:, ].permute(0, 2, 1).contiguous()
                start_conf_prop = conf_prop_feat_[:, :ndim, ].permute(0, 2, 1).contiguous()
                end_conf_prop = conf_prop_feat_[:, ndim:, ].permute(0, 2, 1).contiguous()
                if ssl:
                    return trip
            prop_locs.append(self.prop_loc_head(loc_prop_feat).view(batch_num, 2, -1)
                             .permute(0, 2, 1).contiguous())
            prop_confs.append(self.prop_conf_head(conf_prop_feat).view(batch_num, num_classes, -1)
                              .permute(0, 2, 1).contiguous())
            centers.append(
                self.center_head(loc_prop_feat).view(batch_num, 1, -1)
                    .permute(0, 2, 1).contiguous()
            )

        loc = torch.cat([o.view(batch_num, -1, 2) for o in locs], 1)
        conf = torch.cat([o.view(batch_num, -1, num_classes) for o in confs], 1)
        prop_loc = torch.cat([o.view(batch_num, -1, 2) for o in prop_locs], 1)
        prop_conf = torch.cat([o.view(batch_num, -1, num_classes) for o in prop_confs], 1)
        center = torch.cat([o.view(batch_num, -1, 1) for o in centers], 1)
        priors = torch.cat(self.priors, 0).to(loc.device).unsqueeze(0)
        return loc, conf, prop_loc, prop_conf, center, priors, start, end, \
               start_loc_prop, end_loc_prop, start_conf_prop, end_conf_prop


class BDNet(nn.Module):
    def __init__(self, in_channels=1, backbone_model=None, training=True):
        super(BDNet, self).__init__()

        self.coarse_pyramid_detection = CoarsePyramid([832, 1024],frame_num=frame_num,ConPoolingKernal=ConPoolingKernal)
        self.reset_params()
        self.fourier =  UniTS(input_size=args.input_size, sensor_num=args.input_channel, 
                  window_list=config1.window_list, stride_list=config1.stride_list, k_list=config1.k_list,
                   hidden_channel=config1.hidden_channel)
        self.backbone = I3D_BackBone(in_channels=in_channels)
        self.boundary_max_pooling = BoundaryMaxPooling()
        self._training = training

        if self._training:
            if backbone_model is None:
                self.backbone.load_pretrained_weight()
            else:
                self.backbone.load_pretrained_weight(backbone_model)
        self.scales = [1, 4, 4]

    @staticmethod
    def weight_init(m):
        def glorot_uniform_(tensor):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
            scale = 1.0
            scale /= max(1., (fan_in + fan_out) / 2.)
            limit = np.sqrt(3.0 * scale)
            return nn.init._no_grad_uniform_(tensor, -limit, limit)

        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) \
                or isinstance(m, nn.ConvTranspose3d):
            glorot_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x, proposals=None, ssl=False):
        # x should be [B, C, 256, 96, 96] for THUMOS14
        # print('傅里叶x.shape',x.shape) # [1, 8000, 270]
        x_fourier = self.fourier(x) # [2, 1, 177, 24, 48]
        # print('x_fourier',x_fourier.shape)
        feat_dict = self.backbone(x_fourier)
        # print(feat_dict.keys())
        if ssl:
            top_feat = self.coarse_pyramid_detection(feat_dict, ssl)
            decoded_segments = proposals[0].unsqueeze(0)
            plen = decoded_segments[:, :, 1:] - decoded_segments[:, :, :1] + 1.0
            in_plen = torch.clamp(plen / 4.0, min=1.0)
            out_plen = torch.clamp(plen / 10.0, min=1.0)
            frame_segments = torch.cat([
                torch.round(decoded_segments[:, :, :1] - out_plen),
                torch.round(decoded_segments[:, :, :1] + in_plen),
                torch.round(decoded_segments[:, :, 1:] - in_plen),
                torch.round(decoded_segments[:, :, 1:] + out_plen)
            ], dim=-1)
            anchor, positive, negative = [], [], []
            for i in range(3):
                bound_feat = self.boundary_max_pooling(top_feat[i], frame_segments / self.scales[i])
                # for triplet loss
                ndim = bound_feat.size(1) // 2
                anchor.append(bound_feat[:, ndim:, 0])
                positive.append(bound_feat[:, :ndim, 1])
                negative.append(bound_feat[:, :ndim, 2])

            return anchor, positive, negative
        else:
            loc, conf, prop_loc, prop_conf, center, priors, start, end, \
            start_loc_prop, end_loc_prop, start_conf_prop, end_conf_prop = \
                self.coarse_pyramid_detection(feat_dict)
            return {
                'loc': loc,
                'conf': conf,
                'priors': priors,
                'prop_loc': prop_loc,
                'prop_conf': prop_conf,
                'center': center,
                'start': start,
                'end': end,
                'start_loc_prop': start_loc_prop,
                'end_loc_prop': end_loc_prop,
                'start_conf_prop': start_conf_prop,
                'end_conf_prop': end_conf_prop
            }


def test_inference(repeats=3, clip_frames=256):
    model = BDNet(training=False)
    model.eval()
    model.cuda()
    import time
    run_times = []
    x = torch.randn([1, 3, clip_frames, 96, 96]).cuda()
    warmup_times = 2
    for i in range(repeats + warmup_times):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            y = model(x)
        torch.cuda.synchronize()
        run_times.append(time.time() - start)

    infer_time = np.mean(run_times[warmup_times:])
    infer_fps = clip_frames * (1. / infer_time)
    print('inference time (ms):', infer_time * 1000)
    print('infer_fps:', int(infer_fps))
    # print(y['loc'].size(), y['conf'].size(), y['priors'].size())


if __name__ == '__main__':
    test_inference(20, 256)
