"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
"""
import torch
from torch.optim.lr_scheduler import _LRScheduler

AVAI_SCHEDS = ["single_step", "multi_step", "cosine", "OnPlateau"]


class _BaseWarmupScheduler(_LRScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        last_epoch=-1,
        verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)


class ConstantWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        cons_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]


class LinearWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        min_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.min_lr = min_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        if self.last_epoch == 0:
            return [self.min_lr for _ in self.base_lrs]
        return [
            lr * self.last_epoch / self.warmup_epoch for lr in self.base_lrs
        ]


def build_lr_scheduler(optimizer, cfg):
    """A function wrapper for building a learning rate scheduler.

    Args:
        optimizer (Optimizer): an Optimizer.
        optim_cfg (CfgNode): optimization config.
    """
    lr_scheduler = cfg.schedule.name
    
    max_epoch = cfg.train.epoch

    if lr_scheduler not in AVAI_SCHEDS:
        raise ValueError(
            "Unsupported scheduler: {}. Must be one of {}".format(
                lr_scheduler, AVAI_SCHEDS
            )
        )
    

    if lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(max_epoch),eta_min=cfg.schedule.min_lr
        )
    # TODO add more scheduler settings

    # ! 目前没有设置热启动    
    if cfg.schedule.warmup_epoch > 0:
    
        if cfg.schedule.warmup_type == "constant":
            scheduler = ConstantWarmupScheduler(
                optimizer, scheduler, cfg.schedule.warmup_epoch,
                cfg.schedule.warmup_cons_lr
            )

        elif cfg.schedule.warmup_type == "linear":
            scheduler = LinearWarmupScheduler(
                optimizer, scheduler, cfg.schedule.warmup_epoch,
                cfg.schedule.warmup_min_lr
            )

        else:
            raise ValueError

    return scheduler