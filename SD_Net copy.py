import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
import sys
sys.path.append('/root/autodl-tmp/Code/stable_diffusion')
sys.path.append('/root/autodl-tmp/Code/stable_diffusion/taming-transformers')

from einops import rearrange, repeat
from unetwarpper import UNetWrapper, TextAdapter
from utils.util import load_stable_diffusion
from pointnet2_cls_ssg import Pointnet2
from torch.cuda.amp import autocast, GradScaler

class filter_net(nn.Module):
    def __init__(self,cfg):
        super(filter_net, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, cfg.data.pc_num_category)
    
    def forward(self,x):
        
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
    
        return x


class sd_net(nn.Module):

    def __init__(self, cfg, unet_config=dict()):
        super(sd_net, self).__init__()

        self.cfg = cfg
        #* Stable Diffusion params
        self.sd_yaml = cfg.model.sd_yaml
        self.sd_ckpt = cfg.model.sd_ckpt
        #* Text Adapter params
        self.text_dim = cfg.model.text_dim
        self.text_adapter_gamma = cfg.model.text_adapter_gamma
        self.text_embeddings = cfg.data.text_embedding
        #* Point cloud encoders params
        self.pc_class_num = cfg.model.pc_class_num
        self.pc_use_normal = cfg.model.pc_use_normal
        self.pc_encoder = cfg.model.pc_encoder
        
        #* stable diffusion model
        self.sd_model = load_stable_diffusion(sd_yaml=self.sd_yaml,sd_ckpt=self.sd_ckpt)
        self.encoder_vq = self.sd_model.first_stage_model
        
        self.u_net = self.sd_model.model
        for param in self.u_net.parameters():
            param.requires_grad = False
        
        # self.unet = UNetWrapper(self.sd_model.model, **unet_config)
        # self.unet.freeze()
        #! keep consistent to https://github.com/wl-zhao/VPD/tree/main/segmentation
        self.sd_model.model = None
        self.sd_model.first_stage_model = None
        del self.sd_model.cond_stage_model
        del self.encoder_vq.decoder

        #* text adapter
        
        self.text_adapter = TextAdapter(text_dim=self.text_dim)
    
        
        
        #* point cloud encoder 
        # if self.pc_encoder == "PointNet++":
        #     self.pc_encoder = Pointnet2(num_class=self.pc_class_num,normal_channel=self.pc_use_normal)
        # else:
        #     # ToDo: add more point cloud backbones
        #     pass 
        
        #* sd_feature_filter module, refer MLP-Pixer mlp block
        self.f_net = filter_net(cfg=cfg)
        
        

    def forward(self, img, text,flag='train'):

        # print(img.device)
        # print(t.device)
        # print(text_latents.device)
        #* processing point cloud
        # cls_output, point_feats = self.pc_encoder(points)
        
        # if flag!='train': 
        #     #* test
        #     return cls_output
        # else:            
        #* train
        # print(img.device)
        #* processing image data
        with torch.no_grad():
            img_latents = self.encoder_vq.encode(img)
        #? keep consistent to https://github.com/wl-zhao/VPD/tree/main/segmentation, 为什么是取众数
        img_latents = img_latents.mode().detach()
        t = torch.ones((img.shape[0],)).long()
        t = t.to(img.device)
        # print(img_latents.device)
        
        #* processing text data
        
        text_latents = self.text_adapter(img_latents, text, self.text_adapter_gamma)
        # print(text_latents.device)
        
        #* Stable Diffusion outputs
        
        # print(t.device)
        with torch.no_grad():
        
            sd_outs = self.u_net(img_latents, t, c_crossattn=[text_latents])
                
        sd_outs = torch.reshape(sd_outs,[sd_outs.shape[0],-1,1024])
        sd_outs = torch.mean(sd_outs,dim=1).reshape(-1,1024)
        
        
        #* filtering Stable Diffusion outputs                  
        sd_outs_filtering = self.f_net(sd_outs)
        
        # return cls_output, sd_outs_filtering, point_feats
        return  sd_outs_filtering
            













