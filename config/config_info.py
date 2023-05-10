from pydoc import importfile
from yacs.config import CfgNode as CN



def get_default_config():
    
    # root
    cfg = CN()
    
    cfg.src_path = "/home/wangye/Code/3D_SD_knowledge_base/Code/scripts/"
    
    # model
    cfg.model = CN()
    cfg.model.name = 'Stable Diffusion knowledge Extraction Model' 
    cfg.model.sd_version = 'v1-5'
    cfg.model.sd_ckpt = '/root/autodl-tmp/Code/checkpoints/v1-5-pruned-emaonly.ckpt' 
    cfg.model.sd_yaml = '/root/autodl-tmp/Code/scripts/config/v1-inference.yaml'
    cfg.model.text_encoder = 'CLIP_Text_Encoder_Fixed'
    cfg.model.text_dim = 768
    cfg.model.text_adapter_gamma = 1e-4
    cfg.model.pc_encoder = 'PointNet++'
    cfg.model.pc_class_num = 10
    cfg.model.pc_use_normal = False #* 默认不使用法线向量
     
    
    # data
    cfg.data = CN()
    cfg.data.text_embedding = '/root/autodl-tmp/Code/scripts/class_embedding.txt'
    cfg.data.pc_train_path = '/root/autodl-tmp/ModelNet/modelnet40_normal_resampled' #! data path
    cfg.data.pc_test_path = '/root/autodl-tmp/ModelNet/modelnet40_normal_resampled'
    cfg.data.pc_num_points = 1024
    cfg.data.pc_num_category = 10
    cfg.data.pc_process_data = False
    cfg.data.pc_use_normal = cfg.model.pc_use_normal
    cfg.data.pc_use_uniform_sample = False
    cfg.data.img_path = '/root/autodl-tmp/ModelNet/modelnet40_images_new_12x'
    cfg.data.img_height = 512
    cfg.data.img_width = 512
    
    
    
    # train
    cfg.train = CN()
    cfg.train.flag = True #! train (True) or test (False)
    cfg.train.random_seed = 2023
    cfg.train.optim = 'adam'
    cfg.train.epoch = 50
    cfg.train.batch_size = 12
    cfg.train.lr_scheduler = ''
    cfg.train.use_multi_gpus = False
    cfg.train.gpus = [0,1]
    cfg.train.device = 'cuda'
   
    
    
    
    cfg.adam = CN()
    cfg.adam.lr = 1e-3
    cfg.adam.beta1 = 0.9 # exponential decay rate for first moment
    cfg.adam.beta2 = 0.999 # exponential decay rate for second moment
    cfg.adam.eps = 1e-8
    cfg.adam.weight_decay = 1e-4
    
    
    
    
    cfg.schedule = CN()
    cfg.schedule.name = 'cosine'
    cfg.schedule.min_lr = 1e-5
    cfg.schedule.warmup_epoch = 5
    cfg.schedule.warmup_type = 'linear'
    cfg.schedule.warmup_cons_lr = 1e-5
    cfg.schedule.warmup_min_lr = 1e-5
    
    # loss
    cfg.loss = CN()
    cfg.loss.temperature = 0.03    
    cfg.loss.infonce_weight = 0.1

    
    
    # test
    cfg.test = CN()
    cfg.test.batch_size = 6
    

    
    return cfg.clone()
    
    