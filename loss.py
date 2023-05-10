import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Classification_Loss(nn.Module):
    def __init__(self):
        super(Classification_Loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)

        return total_loss


# TODO add more details 
class InfoNCE_Loss(nn.Module):
    def __init__(self,cfg):
        super(InfoNCE_Loss, self).__init__()
        self.cfg = cfg
        
    def calculate_loss(self,logits,mask):
        return -torch.log((F.softmax(logits,dim=1)*mask).sum(1))
        
    def forward(self, pc_feats, sd_feats):
        
        pc_feats = F.normalize(pc_feats.float(), dim=1)
        sd_feats = F.normalize(sd_feats.float(), dim=1)
        
        pc_sd_matrix = pc_feats @ sd_feats.T
        sd_pc_matrix = sd_feats @ pc_feats.T
        
        pc_sd_matrix = pc_sd_matrix / self.cfg.loss.temperature
        sd_pc_matrix = sd_pc_matrix / self.cfg.loss.temperature
        
        mask = torch.from_numpy(np.eye(pc_sd_matrix.shape[0]))
        device = torch.device(self.cfg.train.device)
        mask = mask.to(device)
        
        loss_pc2sd = self.calculate_loss(pc_sd_matrix, mask).mean()
        loss_sd2pc = self.calculate_loss(sd_pc_matrix, mask).mean()
        
        loss = (loss_sd2pc + loss_pc2sd) / 2
        return loss 
        
        
        
        