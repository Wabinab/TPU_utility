# %% [code]
"""
Contains utilities that does not requires TPU to run. 
Works together with tpu_utility_1.py which contains functions
requiring TPU to run. 
Separating them allow tests to be run. 
Created on: 27 August 2021.
"""
import copy
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.utils.data as D
from torch.optim import lr_scheduler

from pathlib import Path
from tqdm.auto import tqdm

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-25T05:58:03.379271Z","iopub.execute_input":"2021-08-25T05:58:03.379563Z","iopub.status.idle":"2021-08-25T05:58:03.388903Z","shell.execute_reply.started":"2021-08-25T05:58:03.379535Z","shell.execute_reply":"2021-08-25T05:58:03.387263Z"}}
def dataloader(train_ds, val_ds, flags, test_ds=None, distributed=False):
    """
    flags requirement: (python dict)
        "bs": (int) batch_size,
        "num_workers": (int) number of workers.

    :args have_test: (bool) return test dataloader? Default: False
        If True: requires test_ds to be defined. 
    """
    if distributed:
        import torch_xla.core.xla_model as xm
        train_sampler = D.distributed.DistributedSampler(
            train_ds, num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(), shuffle=True)
        train_loader = D.DataLoader(train_ds, batch_size=flags["bs"], sampler=train_sampler,
                                   num_workers=flags["num_workers"], drop_last=True)
    else:
        train_loader = D.DataLoader(train_ds, batch_size=flags["bs"], shuffle=True,
                                   num_workers=flags["num_workers"], drop_last=False)
        
    val_loader = D.DataLoader(val_ds, batch_size=flags["bs"], shuffle=False,
                             num_workers=flags["num_workers"],
                             drop_last=True if distributed else False)
    
    if not test_ds: test_loader = None
    else: test_loader = D.DataLoader(test_ds, batch_size=flags["bs"], shuffle=False, 
                                    num_workers=flags["num_workers"])
    
    return {"train": train_loader, "val": val_loader, "test": test_loader}


# %% [markdown]
# ## Show one batch

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-25T05:58:03.440365Z","iopub.execute_input":"2021-08-25T05:58:03.440666Z","iopub.status.idle":"2021-08-25T05:58:03.451694Z","shell.execute_reply.started":"2021-08-25T05:58:03.440634Z","shell.execute_reply":"2021-08-25T05:58:03.450924Z"}}
def show_one_batch(dls, nrows=2, ncols=4, aspect=None):
    """
    Show image of a single batch up to nrows * ncols images. 
    Will raise error if nrows * ncols is larger than the number of batches. 
    The title of each subplots are their labels (same as fastai). 
    
    Args:
        dls: (python dict) Containing keys "train" and "test" corresponding to 
            PyTorch DataLoader. Use `dataloader` function to get this. 
        nrows: (int) matplotlib subplot args. 
        ncols: (int) matplotlib subplot args. 
        aspect: (str/others) matplotlib imshow args. Default: None. 
    For matplotlib args, check matplotlib respective docs for what is supported. 
    """
    for data, target in dls["train"]: break
    
    count = 0
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    for row in range(nrows):
        for col in range(ncols):
            ax[row, col].imshow(data[count].permute(1, 2, 0), aspect=aspect)
            ax[row, col].axis("off")
            ax[row, col].title.set_text(target[count])
            count += 1


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-25T05:58:03.453140Z","iopub.execute_input":"2021-08-25T05:58:03.453784Z","iopub.status.idle":"2021-08-25T05:58:03.471632Z","shell.execute_reply.started":"2021-08-25T05:58:03.453717Z","shell.execute_reply":"2021-08-25T05:58:03.470572Z"}}
class LinearScheduler(lr_scheduler._LRScheduler):
    """
    Linearly increases lr between two boundaries over a number of iterations. 
    """
    def __init__(self, opt, end_lr, num_iter):  # original `optimizer`
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(LinearScheduler, self).__init__(opt)
        
    def get_lr(self):
        """Formula: start + pct * (end - start)"""
        curr_iter = self.last_epoch + 1
        pct = curr_iter / self.num_iter  # ratio
        return [base_lr + pct * (self.end_lr - base_lr) for base_lr in self.base_lrs]
    
    
class ExponentialScheduler(lr_scheduler._LRScheduler):
    """
    Exponentially increases lr between two boundaries over a number of iterations. 
    """
    def __init__(self, opt, end_lr, num_iter, last_epoch = -1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialScheduler, self).__init__(opt, last_epoch=last_epoch) ## 
        
    def get_lr(self):
        curr_iter = self.last_epoch + 1
        pct = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** pct for base_lr in self.base_lrs]
    
    
class CosineScheduler(lr_scheduler._LRScheduler):
    """
    Cosine increases lr between two boundaries over a number of iterations. 
    """
    def __init__(self, opt, end_lr, num_iter, last_epoch = -1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(CosineScheduler, self).__init__(opt, last_epoch=last_epoch) ## 
        
    def get_lr(self):
        curr_iter = self.last_epoch + 1
        pct = curr_iter / self.num_iter
        cos_out = np.cos(np.pi * pct) + 1
        return [self.end_lr + (base_lr - self.end_lr) / 2 * cos_out for base_lr in self.base_lrs]


# %% [code] {"execution":{"iopub.status.busy":"2021-08-13T07:53:08.882115Z","iopub.execute_input":"2021-08-13T07:53:08.882454Z","iopub.status.idle":"2021-08-13T07:53:08.888884Z","shell.execute_reply.started":"2021-08-13T07:53:08.882424Z","shell.execute_reply":"2021-08-13T07:53:08.887981Z"},"jupyter":{"outputs_hidden":false}}
def annealing_no(start, end, pct):
    """No annealing, always return 'start'."""
    return start


def annealing_linear(start, end, pct):
    """Linearly anneal from start to end as pct goes from 0.0 to 1.0."""
    return start + (pct * (end - start))


def annealing_exp(start, end, pct):
    """Exponentially anneal from start to end as pct goes from 0.0 to 1.0."""
    return start * (end / start) ** pct


def annealing_cos(start, end, pct):
    """Cosine anneal from start and end as pct goes from 0.0 to 1.0."""
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start - end) / 2 * cos_out


# %% [markdown]
# `train_tpu` normalize function gotten from https://github.com/albumentations-team/albumentations/blob/300ee99386ad27f482387047dac4f6dddff11ac2/albumentations/augmentations/functional.py#L131

# %% [code] {"jupyter":{"outputs_hidden":false}}
def normalize_fn(data=None, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), 
              max_pixel=255, calculated_input=False):
    """
    Normalize image function. 
    
    :input requirement: (PyTorch Tensor) of shape (Batch, channel, height, width)
    
    :args:
        :data: (The input). If passed in, will not return mean and std. If None, return
            mean and std instead. 
        :mean: (int/tuple) If int, will be broadcasted in the channel dimension. 
            If tuple, must have same number of values as number of channels. 
            Mean of values. 
        :std: (int/tuple) Check mean for explanation. Standard deviation of values. 
        :max_pixel: (int/float) This is the max pixel values. Default: 255 (so image are from 0-255).
        :calculated input: (bool) Whether the input are already calculated, as in 
            they are tensors to be used directly with the correct shape. 
        
    :return: 
        (data != None) normalized data, same shape as input. 
        (data is None) mean, std 
    """
    if not calculated_input: 
        if type(mean) in [float, int]: mean = [mean]
        if type(std) in [float, int]: std = [std]

        mean = np.array(mean) * max_pixel
        std = np.array(std) * max_pixel

        mean = torch.from_numpy(mean).view(1, mean.shape[0], 1, 1).type(torch.float32)
        std = torch.from_numpy(std).view(1, std.shape[0], 1, 1).type(torch.float32)
        
    if data == None: return mean, std
    
    assert type(mean) == torch.Tensor
    assert type(std) == torch.Tensor
    assert data.shape[1] == mean.shape[1] == std.shape[1]
    return (data - mean.to(data.device)) / std.to(data.device)