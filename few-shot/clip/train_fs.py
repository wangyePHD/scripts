
import os
import torch
import numpy as np
import datetime
import logging
import shutil
import argparse
import sys
import time
sys.path.append("/home/wangye/Code/3D_SD_knowledge_base/Code/scripts/utils")

from pathlib import Path
from tqdm import tqdm
# from ModelNetDataLoader import ModelNetDataLoader

from util import *
from metric import *
from loss import *

from torch.utils.tensorboard import SummaryWriter
from fs_clip import Image_Text_CLIP
from optimizer import optimizer_setup
from torch.cuda.amp import autocast, GradScaler
from lr_schedule import build_lr_scheduler
import torch.cuda.amp as amp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))



def forward_backward(cfg, target, image,model,optimizer,cls_loss,scaler,text):
    #* process data
    device = torch.device(cfg.train.device)
    target = target.to(device)
    image  = image.to(device)
    text   = text.to(device)

    with autocast():
        #* model forward and calculate loss
        sd_outs_filtering = model(image,text)
        classification_loss = cls_loss(sd_outs_filtering,target.long())
        
    loss = classification_loss
    
    #* model backward and update
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    loss_summary = {
        'cls_loss': classification_loss.item()
    }
    
    return loss_summary,sd_outs_filtering
    
    
@torch.no_grad()
def model_inferrence(cfg, target, image, model, cls_loss, text):
    #* process data
    device = torch.device(cfg.train.device)
    target = target.to(device)
    image  = image.to(device)
    text   = text.to(device)
        
    #* model forward and calculate loss
    cls_output = model(image,text)
    classification_loss = cls_loss(cls_output,target.long())
        
    loss_summary = {
        'cls_loss': classification_loss.item(),
    }
    
    return loss_summary,cls_output
    


def get_current_lr(optimizer):
    
    return optimizer.param_groups[0]['lr']
    
    

def run_epoch(cfg, model, train_loader, optimizer, scheduler, epoch ,cls_loss,scaler,text):
    
    logger.info("set the model mode as train")
    set_model_mode(flag='train',model=model)
    
    loss_summary = MetricMeter()
    ins_accuracy_meter = AverageMeter()
    batch_time = AverageMeter()
    num_batches = len(train_loader)
    class_acc = np.zeros((cfg.data.pc_num_category, 3))
    
    for batch_idx, batch in enumerate(train_loader):
        
        image, target = batch
        batch_start_time = time.time()
        loss,pred = forward_backward(cfg,target,image,model,optimizer,cls_loss,scaler,text)
        batch_end_time = time.time()
        batch_cost = batch_end_time - batch_start_time
        batch_time.update(batch_cost)

        loss_summary.update(loss)
        num_batch_current = num_batches - (batch_idx+1)
        num_batch_future  = (cfg.train.epoch - (epoch+1)) * num_batches
        eta_seconds = batch_time.avg * (num_batch_current + num_batch_future)
        eta = str(datetime.timedelta(seconds=int(eta_seconds)))
        
        #* calculate ins accuracy
        pred_choice = pred.cpu().data.max(1)[1]
        target = target.cpu()
        correct = pred_choice.eq(target.long().data).cpu().sum()
        accuracy = correct.item() / float(image.size()[0])
        ins_accuracy_meter.update(accuracy)
        
        #* calculate class accuracy 
        
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(image[target == cat].size()[0])
            class_acc[cat, 1] += 1
    
        logger.info(
            'epoch [{0}/{1}][{2}/{3}] time {batch_time.val:.3f} ({batch_time.avg:.3f}) eta {eta}  {losses} Accuracy {accuracy_meter.val:.4f} ({accuracy_meter.avg:.4f})  lr {lr:.6f}', 
            epoch + 1,
            cfg.train.epoch,
            batch_idx + 1,
            num_batches,
            batch_time=batch_time,
            eta=eta,
            losses=loss_summary,
            accuracy_meter=ins_accuracy_meter,
            lr=get_current_lr(optimizer)
            )
      
        
    
    scheduler.step()

    avg_ins_acc = ins_accuracy_meter.avg
    avg_class_acc = 0.0
    cls_loss_avg = 0.0
    infonce_loss_avg = 0.0
    for name, meter in loss_summary.meters.items():
        if name =='cls_loss':
            cls_loss_avg = meter.avg
        if name =='nce_loss':
            infonce_loss_avg = meter.avg
    
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    avg_class_acc = np.mean(class_acc[:,2])
    
    logger.info("-------------------------------------------------------------------------------------------------------------------------------------------------------")
    logger.info("Epoch: {}, Train Ins Avg Acc: {}, Train Class Avg Acc: {}, Train Cls Avg Loss: {}, Train Nce Avg Loss: {}",epoch,round(avg_ins_acc,4),round(avg_class_acc,4),round(cls_loss_avg,4),round(infonce_loss_avg,4))
    logger.info("-------------------------------------------------------------------------------------------------------------------------------------------------------")
        
    return [avg_ins_acc, avg_class_acc,cls_loss_avg,infonce_loss_avg]
    
        
        
def test_after_epoch(cfg, model, test_loader,optimizer, epoch,cls_loss,text):
    
    logger.info("set the model mode as test")
    set_model_mode(flag='test',model=model)
    
    loss_summary = MetricMeter()
    ins_accuracy_meter = AverageMeter()
    class_accuracy_meter = AverageMeter()
    batch_time = AverageMeter()
    num_batches = len(test_loader)
    class_acc = np.zeros((cfg.data.pc_num_category, 3))
    
    for batch_idx, batch in enumerate(test_loader):
        
        image,target = batch
        batch_start_time = time.time()
        loss,pred = model_inferrence(cfg,target,image,model,cls_loss,text)
        batch_end_time = time.time()
        batch_cost = batch_end_time - batch_start_time
        batch_time.update(batch_cost)

        loss_summary.update(loss)
        num_batch_current = num_batches - (batch_idx+1)
        num_batch_future  = (cfg.train.epoch - (epoch+1)) * num_batches
        eta_seconds = batch_time.avg * (num_batch_current + num_batch_future)
        eta = str(datetime.timedelta(seconds=int(eta_seconds)))
        
        #* calculate ins accuracy
        pred_choice = pred.cpu().data.max(1)[1]
        target = target.cpu()
        correct = pred_choice.eq(target.long().data).cpu().sum()
        accuracy = correct.item() / float(image.size()[0])
        ins_accuracy_meter.update(accuracy)
        
        #* calculate class accuracy 
        
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(image[target == cat].size()[0])
            class_acc[cat, 1] += 1
        
        
        
        
        logger.info(
            'epoch [{0}/{1}][{2}/{3}] time {batch_time.val:.3f} ({batch_time.avg:.3f}) eta {eta}  {losses} Accuracy {accuracy_meter.val:.4f} ({accuracy_meter.avg:.4f})  lr {lr:.5f}', 
            epoch + 1,
            cfg.train.epoch,
            batch_idx + 1,
            num_batches,
            batch_time=batch_time,
            eta=eta,
            losses=loss_summary,
            accuracy_meter=ins_accuracy_meter,
            lr=get_current_lr(optimizer)
            )
        
    
    
    avg_ins_acc = ins_accuracy_meter.avg
    avg_class_acc = 0.0
    cls_loss_avg = 0.0
    infonce_loss_avg = 0.0
    for name, meter in loss_summary.meters.items():
        if name =='cls_loss':
            cls_loss_avg = meter.avg
        if name =='nce_loss':
            infonce_loss_avg = meter.avg
    
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    avg_class_acc = np.mean(class_acc[:,2])
    
    logger.info("-------------------------------------------------------------------------------------------------------------------------------------------------------")
    logger.info("Epoch: {}, Test Ins Avg Acc: {}, Test Class Avg Acc: {}, Test Cls Avg Loss: {}",epoch,round(avg_ins_acc,4),round(avg_class_acc,4),round(cls_loss_avg,4))
    logger.info("-------------------------------------------------------------------------------------------------------------------------------------------------------")
        
    return [avg_ins_acc, avg_class_acc,cls_loss_avg,infonce_loss_avg]



def main():
    import argparse
    parser = argparse.ArgumentParser("Parameters")
    parser.add_argument('--base-config',default='/home/wangye/Code/3D_SD_knowledge_base/Code/scripts/config/config_info_fs.py',help='the yasc base config')
    parser.add_argument('--exp-config',default='/home/wangye/Code/3D_SD_knowledge_base/Code/scripts/config/yaml/sd_pointnet2.yaml',help='the exp config file path')
    parser.add_argument('--results-dir',default='/home/wangye/Code/3D_SD_knowledge_base/Output',help='the exp output results dir path')
    parser.add_argument('--exp-name',default='clip_fs_modelnet10',help='the experiment name')
    
    args = parser.parse_args()
    
    #* create experiment results folder
    exp_output_dir = create_dir(args=args)
    logger = get_logger(exp_output_dir)
    logger.info("Creating Experiment Results Folder: {}",exp_output_dir)
    
    #* setup config
    cfg = setup_cfg(args)
    logger.info("Finished loading the config params, the detailed infomation is as follows:")
    logger.info(cfg)
    
    # #* backup current config and code files
    logger.info(" backup current config and code files")
    backup_yaml(cfg=cfg,exp_output_dir=exp_output_dir)
    backup_file(src_list=[cfg.src_path],exp_output_dir=exp_output_dir)
    
    # #* create tensorboard save folder
    logger.info("create tensorboard folder")
    tensorboard_path = create_tensorboard_dir(exp_output_dir=exp_output_dir)
    
    # #* set random seed for random, numpy, torch, etc.
    logger.info("set random seed:{}", cfg.train.random_seed)
    setup_seed(cfg.train.random_seed)
    
    # #* define dataset and dataloader
    # train_dataset = ModelNetDataLoader(root=cfg.data.pc_train_path,root_img=cfg.data.img_path,cfg=cfg,split='train',process_data=cfg.data.pc_process_data)
    # test_dataset  = ModelNetDataLoader(root=cfg.data.pc_test_path, root_img=cfg.data.img_path,cfg=cfg,split='test', process_data=cfg.data.pc_process_data)
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, cfg.train.batch_size, shuffle=True,  num_workers=10, drop_last=True)
    # test_dataloader  = torch.utils.data.DataLoader(test_dataset,  cfg.test.batch_size, shuffle=False, num_workers=10)
    # logger.info("The train dataset length is {}",train_dataset.__len__())
    # logger.info("The test dataset length is {}",test_dataset.__len__())
    # logger.info("The train dataloader contain {} batch, batch size is {}",len(train_dataloader),cfg.train.batch_size)
    # logger.info("The test dataloader contain {} batch, batch size is {}",len(test_dataloader),cfg.train.batch_size)
    
    # #* define torch device
    # device = torch.device(cfg.train.device)
    
    #* define model and multi-gpus training
    # model = sd_net(cfg=cfg)
    model = Image_Text_CLIP(cfg=cfg)
    print(model)
    
    
    # model = use_multi_gpus(flag=cfg.train.use_multi_gpus, model=model, gpus=cfg.train.gpus)
    # model.to(device)
    # logger.info("define model and set training mode | single gpu:{}, multi gpus:{}",not cfg.train.use_multi_gpus, cfg.train.use_multi_gpus)
    
    # #* record trainable parameters
    # trainable_params = record_trainable_params(model)
    # logger.info("The trainable params are {},{}",trainable_params,len(trainable_params))
    
    # #* define loss function
    # cls_loss = Classification_Loss()
    
    # #* define optimizer
    # logger.info("Build optimizer {}, record optimizer params", cfg.train.optim)
    # optimizer = optimizer_setup(cfg=cfg,model=model)
    # record_optimizer_params(optimizer=optimizer,logger=logger)
    
    # #TODO define lr scheduler
    # scheduler = build_lr_scheduler(optimizer=optimizer,cfg=cfg)
    
    
    # #* create results and ckpt saved dir
    # logger.info("Create the exp results output dir....")
    # sub_outputs_dir = create_results_dir(exp_output_dir=exp_output_dir)
    # logger.info("The current results output dir is {}",sub_outputs_dir)
    
    # #* init tensorboard writer for train and test
    # summary_writer = init_writer(exp_output_dir=tensorboard_path,flag='train')
   
    # #* define a GradScaler 
    # scaler = GradScaler()

    # #* define train process
    # logger.info("Starting training!")
    # best_instance_acc = 0.0
    # best_class_acc = 0.0

    # #* load text embedding
    # text_embeddings = load_textembedding(cfg=cfg)
    # #* training
    # for epoch in range(0,cfg.train.epoch):
    #     train_results = run_epoch(cfg=cfg,model=model,train_loader=train_dataloader,optimizer=optimizer,scheduler=scheduler,epoch=epoch,cls_loss=cls_loss,scaler=scaler,text=text_embeddings)
    #     test_results = test_after_epoch(cfg=cfg,model=model,test_loader=test_dataloader,epoch=epoch,cls_loss=cls_loss,text=text_embeddings,optimizer=optimizer)
    #     train_avg_ins_acc, train_avg_class_acc,train_cls_loss_avg,train_infonce_loss_avg = train_results
    #     test_avg_ins_acc, test_avg_class_acc,test_cls_loss_avg,test_infonce_loss_avg = test_results
        
    #     #* tensorboard save information
    #     summary_writer.add_scalars('ins_acc',{'train':train_avg_ins_acc,"test":test_avg_ins_acc},epoch)
    #     summary_writer.add_scalars('class_acc',{'train':train_avg_class_acc,"test":test_avg_class_acc},epoch)
    #     summary_writer.add_scalars('cls_loss',{'train':train_cls_loss_avg,'test':test_cls_loss_avg},epoch)
        
        
    #     if (test_avg_ins_acc >= best_instance_acc):
    #             best_instance_acc = test_avg_ins_acc
    #             best_epoch = epoch + 1

    #     if (test_avg_class_acc >= best_class_acc):
    #         best_class_acc = test_avg_class_acc
            
    #     logger.info('Test Instance Accuracy: {}, Class Accuracy: {}',test_avg_ins_acc, test_avg_class_acc)
    #     logger.info('Best Instance Accuracy: {}, Class Accuracy: {}',best_instance_acc, best_class_acc)
        
    #     #* save best model and other states 
    #     if (test_avg_ins_acc >= best_instance_acc):
    #         logger.info('Save model...')
    #         savepath = sub_outputs_dir + '/best_model.pth'
    #         logger.info('Saving at {}', savepath)
    #         state = {
    #             'epoch': best_epoch,
    #             'instance_acc': test_avg_ins_acc,
    #             'class_acc': test_avg_class_acc,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #         }

    #         torch.save(state, savepath)
        
    #     logger.info("Saved Finished, Continue To Train......")
        
        
        

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    
    
    main()
    
    
    
    
    
    