# %% [code]
from .tpu_utility_1 import *

import torch_xla  # as a decoration here. 
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu
import torch_xla.utils.cached_dataset as xcd

import torch
import torch.nn as nn
import torch.utils.data as D
from torch.optim import lr_scheduler

import numpy as np 
import pandas as pd 
import os
from IPython.display import clear_output

# %% [code] {"jupyter":{"outputs_hidden":false}}
def cache_dataset(train_ds=None, val_ds=None, get_dataset=None, compress=False):
    """
    train_ds: (PyTorch Dataset) training dataset. 
    val_ds: (PyTorch Dataset) validation dataset.
    get_dataset (Python Function) (only used if train_ds and val_ds are not passed.)
        Function that returns train_ds and val_ds as tuple. 
    compress: Whether to .tar.gz the output folder. Defaults: False. 
    kwargs: keyword arguments that are passed into get_dataset. 
    """
    if (train_ds is get_dataset is None) or (val_ds is get_dataset is None): 
        raise ValueError("Please pass in (train_ds AND val_ds) OR get_dataset. ")
    
    def _mp_fn(index, train_ds, val_ds):
        if get_dataset: train_ds, val_ds = get_dataset()
        train_ds = xcd.CachedDataset(train_ds, "./cached-train")
        val_ds = xcd.CachedDataset(val_ds, "./cached-val")
        print("Creating training dataset.")
        train_ds.warmup()
        print("Done\nCreating validationd dataset.")
        val_ds.warmup()
        print("Done")
        
    xmp.spawn(_mp_fn, args=(train_ds, val_ds,), start_method="fork", nprocs=1)
    
    clear_output()
    
    if compress:
        print("Compressing validation dataset (before training dataset)")
        os.system("tar czf cached-val.tar.gz ./cached-val")
        print("Deleting original folder (to save disk space)")
        os.system("rm -r ./cached-val")
        print("Compressed validation dataset")
        
        print("Compressing training dataset")
        os.system("tar czf cached-train.tar.gz ./cached-train")
        print("Deleting original folder to save disk space")
        os.system("rm -r ./cached-train")
        print("Compresed training dataset.")