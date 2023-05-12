# from clip import 
import clip
from torch.nn import functional as F

import torch
import torch.nn as nn


def load_clip_to_cpu(cfg):
    '''
    load clip pre-trained model
    '''
    
    vision_encoder_name = cfg.model.vision_encoder
    url = clip._MODELS[vision_encoder_name]
    model_path = clip._download(url,root='')
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')
    
    model = clip.build_model(state_dict or model.state_dict())

    return model



class Textual_Encoder(nn.Module):
    
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.cfg = cfg
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
        
    def forward(self,x):
        
        x = torch.squeeze(x,dim=1)
        text_feat = self.clip_model.encode_text(x)
        return text_feat

 

class Image_Text_Embedding(nn.Module):
    
    def __init__(self, cfg, clip_model):
        super().__init__()
        
        self.dtype = clip_model.dtype
        self.cfg = cfg
        self.vision_encoder = clip_model.visual
        self.text_encoder = Textual_Encoder(cfg=cfg,clip_model=clip_model)

       

    #! 返回值需要灵活 提供两个返回值
    def forward(self,img,text):
        
        if text!=None:
            text_feat = self.text_encoder(text)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        if img!=None:
            img = img.type(self.dtype)
            img_feat = self.vision_encoder(img)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        
        if text==None:
            return img_feat
        elif img == None:
            return text_feat
        else:
            return img_feat,text_feat
        
     

class Image_Text_CLIP():
    def __init__(self,cfg):
        
        self.cfg = cfg
        self.device = torch.device(cfg.train.device)
    
    def build_model(self):
        
        clip_model = load_clip_to_cpu(self.cfg)
        model = Image_Text_Embedding(self.cfg,clip_model)
        model = model.to(self.device)
        
        return model