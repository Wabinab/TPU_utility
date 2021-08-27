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