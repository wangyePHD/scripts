import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import random
import time
import shutil
import glob

import sys
sys.path.append('/home/wangye/Code/3D_SD_knowledge_base/Code/stable_diffusion')
sys.path.append('/home/wangye/Code/3D_SD_knowledge_base/Code/stable_diffusion/taming-transformers')
sys.path.append('/home/wangye/Code/3D_SD_knowledge_base/backbones/Pointnet_Pointnet2_pytorch')
sys.path.append('/home/wangye/Code/3D_SD_knowledge_base/Code/scripts')
from omegaconf import  OmegaConf
from ldm.util import instantiate_from_config
from transformers import CLIPTokenizer
from unetwarpper import  FrozenCLIPEmbedder
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from config.config_info import get_default_config



def set_model_mode(flag,model):
    if flag=='train':
        model.train()
    else:
        model.eval()

def init_writer(exp_output_dir,flag):
    temp_dir = os.path.join(exp_output_dir,flag)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    writer = SummaryWriter(temp_dir)
    return writer


def create_results_dir(exp_output_dir):
    '''
    create the exp results output dir
    '''
    output = os.path.join(exp_output_dir,'output')
    if not os.path.exists(output):
        os.makedirs(output)
    return output


def record_optimizer_params(optimizer,logger):
    for group in optimizer.param_groups:
        for key, value in group.items():
            if key == 'params':
                logger.info('optimizing params number:{}', len(value))
            else:
                logger.info('{}, {}', key, value)
                

def use_multi_gpus(flag,model,gpus):
    if flag:
        #* multi-gpus
        model = torch.nn.DataParallel(model,device_ids=gpus)
        return model
    else:
        #* single-gpu
        return model





def record_trainable_params(model):
    trainable_params = []
    for name,para in model.named_parameters():
        if para.requires_grad:
            trainable_params.append(name)
    return trainable_params



def create_tensorboard_dir(exp_output_dir):
    
    
    path = os.path.join(exp_output_dir,"tensorboard")
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass
    return path



def backup_yaml(cfg,exp_output_dir):
        
    file = os.path.join(exp_output_dir,"config.yaml")    
    f = open(file,'a+')
    print(cfg,file=f)



        
    
def backup_file(src_list,exp_output_dir):
    '''
    backup the current experiment code and config files
    '''
    
    # src_list = ['/root/autodl-tmp/z_indi_avg_feat_project/Code/']
    backup_code = "backup_code"
    target = os.path.join(exp_output_dir,backup_code)
    for src in src_list:
        if not os.path.exists(target):
            if os.path.isdir(src):
                shutil.copytree(src, target)
            elif os.path.isfile(src):
                os.makedirs(target)
                shutil.copy(src, target)
        else:
            if os.path.isdir(src):
                dirs = glob.glob(src+'/*')
                for dir_i in dirs:
                    subdir = os.path.basename(dir_i)
                    tar_dir = target+'/'+ subdir
                    if not os.path.exists(tar_dir):
                        if os.path.isdir(dir_i):
                            shutil.copytree(dir_i, tar_dir)
                        elif os.path.isfile(dir_i):
                            shutil.copy(dir_i, tar_dir)
            elif os.path.isfile(src):
                shutil.copy(src, target)
    current_file = os.path.abspath(__file__)
    
    shutil.copy(current_file,target)
    
    
   



def setup_cfg(args):
    # load base config params
    cfg = get_default_config()
    
    if args.exp_config:
        cfg.merge_from_file(args.exp_config)
    
    cfg.freeze()
    
    return cfg






def get_time():
    now = int(round(time.time()*1000))
    now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(now/1000))
    return str(now)

def create_dir(args):
    temp_time = get_time()
    temp_time_ = '-'.join(temp_time.split(' '))
    # create the exp output dir if it is not exist
    exp_output_dir = os.path.join(args.results_dir,temp_time_+"-"+str(args.exp_name))
    if not os.path.exists(exp_output_dir):
        os.makedirs(exp_output_dir)
    return exp_output_dir


#* define a logger 
def get_logger(path):
    
    logger.add(os.path.join(path,'exp_{time}.log'),format="{time} | {level} | {message}",level="INFO")
    return logger


#* save best model
def save_checkpoint(best_loss,test_loss,cfg,model,exp_output_dir,epoch):
    
    ckpt = os.path.join(exp_output_dir,'ckpt')
    if not os.path.exists(ckpt):
        os.makedirs(ckpt)
    
    if test_loss<=best_loss:
        logger.info("epoch = {}, the best loss have been updated!!!",epoch)
        best_loss = test_loss
        path = os.path.join(ckpt,'model-best.pth')
        torch.save(model.state_dict(),path)
        logger.info("The best model ckpt saved!!!")

    else:
        logger.info("This is not the best model!!!")
        
    return best_loss


#* load pre-trained Stable Diffusion
def load_stable_diffusion(sd_yaml,sd_ckpt):

    # ! keep consistency to https://github.com/wl-zhao/VPD/tree/main/segmentation
    config = OmegaConf.load(sd_yaml)
    config.model.params.ckpt_path = sd_ckpt
    config.model.params.cond_stage_config.target = 'ldm.modules.encoders.modules.AbstractEncoder'

    sd_model = instantiate_from_config(config.model)
    # sd_model = sd_model.cuda()
    

    return sd_model



#* convert text to embedding using CLIP
def text_to_embedding():
    '''
    convert text to clip embeddings
    '''
    text_file = "/root/autodl-tmp/ModelNet/modelnet40_normal_resampled/modelnet10_shape_names.txt"
    text_list = []
    f = open(text_file,'r')
    for line in f.readlines():
        text_list.append(line.strip())
    text_template = "a photo of a"
    text_clip_tokenizer = FrozenCLIPEmbedder().cuda()


    text_embedding_list = []
    for category in text_list:
        text = text_template + ' ' + category
        text_latents = text_clip_tokenizer.encode(text)
        text_embedding_list.append(text_latents)

    text_embedding = torch.concat(text_embedding_list,dim=0)
    text_embedding = text_embedding.cpu().numpy()
    print(text_embedding)
    np.savetxt('./class_embedding.txt', text_embedding)




#* set random seed for reproduce results
def setup_seed(seed=2023):
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    np.random.seed(seed)
    random.seed(seed)
    
    torch.backends.cudnn.deterministic = True  # 为了确定算法，保证得到一样的结果。
    torch.backends.cudnn.benchmark = False  # 为了加速。因为大小不定，所以关闭，防止降低效率
    torch.backends.cudnn.enabled = False



#* load checkpoints trained on multi-gpus
def load_mul_gpu_model(model,path):
    device = torch.device('cuda:0')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    state_dict = torch.load(path)
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    return model

#* load checkpoints trained on single-gpu
def load_one_gpu_model(model,path):
    device = torch.device('cuda:0')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    state_dict = torch.load(path)
    for k, v in state_dict.items():
        name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    return model

def load_textembedding(cfg):    
    text_latents = np.loadtxt(cfg.data.text_embedding).astype(np.float32)
    text_latents = torch.from_numpy(text_latents)
    
    text_list = []
    for i in range(len(cfg.train.gpus)):
        text_list.append(text_latents)
    text = torch.concat(text_list,dim=0)
    return text


if __name__ == '__main__':

    # text_to_embedding()
    text_latents = np.loadtxt("/root/autodl-tmp/Code/scripts/class_embedding.txt").astype(np.float32)
    text_latents = torch.from_numpy(text_latents)
    print(text_latents.shape)




