import torch
import torch.nn as nn

#* define optimizer
def optimizer_setup(cfg,model):
   
    
    parameters = model.parameters()
    
    if cfg.train.optim == 'adam':
        optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad,parameters),
            lr=cfg.adam.lr,
            betas=(cfg.adam.beta1, cfg.adam.beta2),
            eps=cfg.adam.eps,
            weight_decay=cfg.adam.weight_decay
        )
        
    
    # TODO config中没有配置参数
    if cfg.train.optim == 'sgd':
        optimizer = torch.optim.SGD(
            params=filter(lambda p: p.requires_grad,parameters),
            lr=cfg.train.lr,
            momentum=cfg.train.momentum,
            dampening=cfg.train.sgd_dampning,
            weight_decay=cfg.train.weight_decay,
            nesterov=cfg.train.sgd_nesterov
        )
        
    return optimizer
    