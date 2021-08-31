# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-25T05:58:01.697697Z","iopub.execute_input":"2021-08-25T05:58:01.698083Z","iopub.status.idle":"2021-08-25T05:58:03.377702Z","shell.execute_reply.started":"2021-08-25T05:58:01.698048Z","shell.execute_reply":"2021-08-25T05:58:03.376787Z"}}
import copy
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from IPython.display import clear_output

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

from pathlib import Path
from tqdm.auto import tqdm

from sklearn.metrics import f1_score

from .other_utils import *
from ._lr_finder import *

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-25T05:58:03.390186Z","iopub.execute_input":"2021-08-25T05:58:03.390486Z","iopub.status.idle":"2021-08-25T05:58:03.408775Z","shell.execute_reply.started":"2021-08-25T05:58:03.390457Z","shell.execute_reply":"2021-08-25T05:58:03.407563Z"}}
def distrib_dataloader(get_dataset, flags, cached=False):
    """
    Will run without test_ds available.
    get_dataset: (function) The function that returns train_ds, val_ds as tuple. 
    flags: (python dict): Requires: 
        "bs": (int) batch_size,
        "num_workers": (int) number of workers.
    cached: Not yet implemented. 

    To assign test_ds, you can do dls["test"] = test_ds later on. 
    """
    serial_exec = xmp.MpSerialExecutor()
    
    train_ds, val_ds = serial_exec.run(get_dataset)
    dls = dataloader(train_ds, val_ds, flags, distributed=True)
    return dls

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-25T05:58:03.410078Z","iopub.execute_input":"2021-08-25T05:58:03.410580Z","iopub.status.idle":"2021-08-25T05:58:03.421279Z","shell.execute_reply.started":"2021-08-25T05:58:03.410544Z","shell.execute_reply":"2021-08-25T05:58:03.420437Z"}}
def train_cycle_distrib(dls, flags, train_loop_fn, val_loop_fn, device=None):
    """
    dls: dataloaders, use distrib_dataloader() to get this. 
    flags: (python dict). Required:
        "num_epochs": number of epochs to train for. 
        "metrics_debug": whether to print metrics report of TPU. 
    train_loop_fn: (function) The training loop function.
    val_loop_fn: (function) The validation loop function. 
    device = device. Defaults: xm.xla_device()
    """
    for epoch in range(1, flags["num_epochs"] + 1):
        para_loader = pl.ParallelLoader(dls["train"], [device])
        train_loop_fn(para_loader.per_device_loader(device))
        clear_output(wait=True)
        xm.master_print(f"Finished training epoch {epoch}")

        para_loader = pl.ParallelLoader(dls["val"], [device])
        returned_val = val_loop_fn(para_loader.per_device_loader(device))
        if flags["metrics_debug"]: xm.master_print(met.metrics_report(), flush=True)
        
    return returned_val

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-25T05:58:03.422940Z","iopub.execute_input":"2021-08-25T05:58:03.423371Z","iopub.status.idle":"2021-08-25T05:58:03.439128Z","shell.execute_reply.started":"2021-08-25T05:58:03.423341Z","shell.execute_reply":"2021-08-25T05:58:03.438030Z"}}
def cached_dataset(cache_train_loc=None, cache_val_loc=None):
    """
    NOTE: This SHOULD BE CALLED inside the `_mp_fn` function of distributed. 
    Will fetch the cached dataset that is already preprocessed. 
    """
    if cache_train_loc is None: cache_train_loc = "./cache_train"
    if cache_val_loc is None: cache_val_loc = "./cache_val"
        
    train_ds = xcd.CachedDataset(None, cache_train_loc)
    val_ds = xcd.CachedDataset(None, cache_val_loc)
    
    return train_ds, val_ds

# %% [markdown]
# # LR Finder TPU

# Originally taken from https://nbviewer.jupyter.org/github/aman5319/Multi-Label/blob/master/Classify_scenes.ipynb
# Some information are gotten from https://github.com/davidtvs/pytorch-lr-finder/blob/master/torch_lr_finder/lr_finder.py

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-25T06:04:09.838292Z","iopub.execute_input":"2021-08-25T06:04:09.838826Z","iopub.status.idle":"2021-08-25T06:04:09.845127Z","shell.execute_reply.started":"2021-08-25T06:04:09.838793Z","shell.execute_reply":"2021-08-25T06:04:09.844020Z"}}
def lr_finder(model, opt, criterion, dls=None, train_loader=None, device=None):
    """
    Learning Rate Finder with default values. 
    model: your model.
    opt: torch.optim optimizers.
    criterion: torch.nn loss function. 
    dls: dataloaders. Check out `dataloader` function. Don't pass train_loader if this is used.
    train_loader: train_ds. PyTorch Dataset. Don't pass dls if this is used. 
    device: device.
    
    Return: steepest_point, model, best_loss
    """
    lrfinder = LRFinder(model, opt, criterion, device=device, barrier=True)
    if dls is train_loader is None: raise ValueError("One of dls or train_loader must be passed.")
    return lrfinder.lr_find(dls["train"] if dls else train_loader)

# %% [code] {"jupyter":{"outputs_hidden":false}}
def distrib_lr_finder(model, opt, criterion, dls=None, train_loader=None, device=None):
    """
    Learning Rate Finder with default values. 
    Will use distributed method to allow definition of `xm.xla_device()` in training
    without requiring restart of jupyter notebook. 
    
    :Args:
        model: your model.
        opt: torch.optim optimizers.
        criterion: torch.nn loss function. 
        dls: dataloaders. Check out `dataloader` function. Don't pass train_loader if this is used.
        train_loader: PyTorch DataLoader. Using this if dls not detected. 
        device: device.
        
    :return: (Nothing is returned)
    """
    SERIAL_EXEC = xmp.MpSerialExecutor()
    WRAPPED_MODEL = xmp.MpModelWrapper(model)

    def train_fn(lrfinder, train_loader):
        device = xm.xla_device()
        model = WRAPPED_MODEL.to(device)
        if not xm.is_master_ordinal(): xm.rendezvous("lrfind_only_once")
        model, opt = SERIAL_EXEC.run(lambda: lrfinder.lr_find(train_loader, model=model, device=device))
        if xm.is_master_ordinal(): xm.rendezvous("lrfind_only_once")
        return model, opt

    def _mp_fn(rank, model, opt, criterion, device):
        lrfinder = LRFinder(model, opt, criterion, device=device, barrier=False)
        if dls is train_loader is None: raise ValueError("One of dls or train_loader must be passed.")
        model, opt = train_fn(lrfinder, dls["train"] if dls else train_loader)
        if rank == 0: 
            torch.save({
                "model": model.cpu().state_dict(),
                "opt": opt.cpu().state_dict(),
            }, "/kaggle/working/temp.pth")
        
    xmp.spawn(_mp_fn, args=(WRAPPED_MODEL, opt, criterion, device,), start_method="fork", nprocs=8)

# %% [markdown]
# # One Cycle Policy
# Taken from https://nbviewer.jupyter.org/github/aman5319/Multi-Label/blob/master/Classify_scenes.ipynb

# %% [code] {"execution":{"iopub.status.busy":"2021-08-13T07:53:04.364632Z","iopub.execute_input":"2021-08-13T07:53:04.365059Z","iopub.status.idle":"2021-08-13T07:53:04.372010Z","shell.execute_reply.started":"2021-08-13T07:53:04.365022Z","shell.execute_reply":"2021-08-13T07:53:04.370905Z"},"jupyter":{"outputs_hidden":false}}
class Stepper():
    """Step through n_iter on a schedule defined by func."""
    def __init__(self, val, n_iter: int, func):
        self.start, self.end = val
        self.n_iter = max(1, n_iter)
        self.func = func
        self.n = 0
        
    def step(self):
        """Returned next value along annealed schedule."""
        self.n += 1
        return self.func(self.start, self.end, self.n/self.n_iter)
    
    @property
    def is_done(self):
        """Return True if schedule complted."""
        return self.n >= self.n_iter
        

# %% [code] {"jupyter":{"outputs_hidden":false}}
class OneCyclePolicy_TPU:
    def __init__(self, model, opt, criterion, FLAGS, num_iter=None, train_ds=None,
                 momentum=(0.95, 0.85), div_factor=25, pct_start=0.4,
                # train_transform=None, val_transform=None, 
                channels_last=True, get_dataset=None,
                cache_train_loc=None, cache_val_loc=None):
        """
        :args:
        
        model: model.
        opt: optimizer.
        criterion: loss function. 
        FLAGS: (python dict) FLAGS containing information to pass in. Refer to FLAGS tutorial
             on what's required to put inside flag. 
        num_iter: (int) number of iterations. please use len(train_dataset) // batch_size
        train_ds: (IF num_iter IS NONE): (PyTorch Dataset) pass in training dataset. 
        momentum: Default: (0.95, 0.85)  # momentum for optimizer.
        div_factor: (int) Minimum learning rate: max_learning_rate / div_factor. 
        pct_start: (float) starting percentage. Defaults 0.4.
        # train_transform: Not used. 
        # val_transform: Not used. 
        channels_last: (bool) Whether to have NHWC instead of NCHW format. 
        get_dataset: (python function). Should returns train_ds, val_ds. Required if 
            you don't define cache_train_loc and cache_val_loc. 
        
        """
        self.SERIAL_EXEC = xmp.MpSerialExecutor()
        self.WRAPPED_MODEL = xmp.MpModelWrapper(model)
        self.opt = opt
        self.criterion = criterion
        self.flags = FLAGS
        self.chls = channels_last
        self.get_dataset = get_dataset
        self.cache_train_loc = cache_train_loc
        self.cache_val_loc = cache_val_loc
        
        if get_dataset is cache_train_loc is None:
            assert os.path.exists("./cache_train"), "Folder does not exist. Please put cached train dataset in this folder."
        if get_dataset is cache_val_loc is None:
            assert os.path.exists("./cache_val"), "Folder does not exist. Please put cached val dataset in this folder."
        
        if num_iter is train_ds is None: 
            raise ValueError("One of num_iter or train_ds must be defined")
        if num_iter is None: num_iter = len(train_ds) // self.flags["bs"]
        
        n = num_iter * self.flags["num_epochs"]
        a1 = int(n * pct_start)
        a2 = n - a1
        self.phases = ((a1, annealing_linear), (a2, annealing_cos))
        
        self.div_factor = div_factor
        self.momentum = momentum
        self.idx_s = 0
        
        max_lr = self.flags["lr"] * xm.xrt_world_size()
        min_lr = max_lr / self.div_factor
        self.lr_scheds = self.steps((min_lr, max_lr), (max_lr, min_lr / 1e4))
        self.mom_scheds = self.steps(self.momentum, self.momentum[::-1])
        self.update_lr_mom(self.lr_scheds[0].start, self.mom_scheds[0].start)

        # Future development
#         self.train_transform = train_transform
#         self.val_transform = val_transform
        
    def steps(self, *steps):
        """Build anneal schedule for all of the parameters. """
        return [Stepper(step, n_iter, func=func)
               for (step, (n_iter, func)) in zip(steps, self.phases)]
    
    def lrfinder(self, model, train_loader):
        if not xm.is_master_ordinal(): xm.rendezvous("lrfinder_once")
        lrfinder = LRFinder(model, self.opt, self.criterion, train_loader=train_loader)
        steepest, _, best_loss = lrfinder.lr_find()
        if xm.is_master_ordinal(): xm.rendezvous("lrfinder_once")
        
        max_lr = steepest * xm.xrt_world_size()
        min_lr = max_lr / self.div_factor
        self.lr_scheds = self.steps((min_lr, max_lr), (max_lr, min_lr / 1e4))
        self.mom_scheds = self.steps(self.momentum, self.momentum[::-1])
        self.update_lr_mom(self.lr_scheds[0].start, self.mom_scheds[0].start)
        
    def update_lr_mom(self, lr=0.001, mom=0.99):
        for l in self.opt.param_groups:
            l["lr"] = lr
            
            if isinstance(self.opt, (torch.optim.Adamax, torch.optim.Adam)):
                l["betas"] = (mom, 0.999)
            elif isinstance(self.opt, torch.optim.SGD):
                l["momentum"] = mom
    
    def train_tpu(self, train_ds, val_ds, fixed_lr=True, normalize=False, average=None):
        """
        :args fixed_lr: (bool) Whether to use a fixed_lr defined in FLAGS. 
            Defaults: True. 
        :args normalize: (bool/set of values) Whether to normalize the dataset 
            based on imagenet values. If passing in set of values, pass in as
            ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 255) for example, 
            corresponding to (mean, std, max_pixel) respectively. 
        :args average: (str/None) f1_score arguments on average. 
        """
        torch.manual_seed(self.flags["seed"])
        
        if train_ds is None: train_ds, val_ds = self.SERIAL_EXEC.run(get_dataset)
        dls = dataloader(train_ds, val_ds, self.flags, distributed=True)
        
        device = xm.xla_device()
        model = self.WRAPPED_MODEL.to(device)
        
        
        if not fixed_lr: self.SERIAL_EXEC.run(lambda: self.lrfinder(model, dls["train"]))
        
#         def train_loop_fn(loader):
#             tracker = xm.RateTracker()
#             model.train()
            
#             running_loss = 0.0
#             total_samples = 0
            
#             for data, target in tqdm(loader):
#                 self.opt.zero_grad()
#                 data, target = data.to(device), target.to(device)
#                 if self.chls: data = data.to(memory_format=torch.channels_last)
                
#                 output = model(data)
#                 loss = self.criterion(output, target.to(torch.float32))
# #                 preds = (torch.sigmoid(output).data > 0.5).to(torch.float32)
                
#                 loss.backward()
#                 xm.optimizer_step(self.opt)
#                 self.update_lr_mom(self.lr_scheds[self.idx_s].step(),
#                                     self.mom_scheds[self.idx_s].step())
                            
#                 if self.lr_scheds[self.idx_s].is_done: self.idx_s += 1
#                 tracker.add(self.flags["bs"])
                
#                 running_loss += loss.item() * data.size(0)
#                 total_samples += data.size(0)

#             return running_loss, total_samples
            
#         def test_loop_fn(loader):
#             total_samples = 0
#             running_loss, f1Score = 0.0, 0.0
#             model.eval()
            
#             for data, target in tqdm(loader):
#                 data, target = data.to(device), target.to(device)
                
#                 output = model(data)
#                 loss = self.criterion(output, target.to(torch.float32))
#                 preds = (torch.sigmoid(output).data > 0.5).to(torch.float32)
                
#                 total_samples += data.size(0)
#                 running_loss += loss.item() * data.size(0)
                
#                 target = target.cpu().to(torch.int).numpy()
#                 preds = preds.cpu().to(torch.int).numpy()
                
#                 f1Score += f1_score(target, preds, average="weighted") * data.size(0)
                
#             epoch_loss = running_loss / total_samples
#             epoch_f1score = f1Score / total_samples
            
# #             print(f"""
# #                 Val loss: {epoch_loss} | 
# #                 Val F1Score: {epoch_f1score} | 
# #             """, flush=True)
#             return epoch_loss, epoch_f1score, data, preds, target

        mean, std = normalize_fn() if normalize else (None, None)
#         if mean is not None: mean, std = mean.to(device), std.to(device)
        
        for epoch in range(1, self.flags["num_epochs"] + 1):
            para_loader = pl.ParallelLoader(dls["train"], [device])
            running_loss, total_samples = self.train_loop_fn(
                                    para_loader.per_device_loader(device), model, mean, std)
            clear_output(wait=True)
            xm.master_print(f"Finished training epoch {epoch}")
            xm.master_print(f"Train loss: {running_loss / total_samples}", flush=True)
            
            para_loader = pl.ParallelLoader(dls["val"], [device])
            test_loss, f1score, data, pred, targ = self.test_loop_fn(
                                    para_loader.per_device_loader(device), model, average,
                                    mean, std)
            xm.master_print(f"""
                Val loss: {test_loss} | 
                Val F1Score: {f1score} | 
            """, flush=True)
            if self.flags["metrics_debug"]: xm.master_print(met.metrics_report(), flush=True)
                
        return test_loss, f1score, data, pred, targ, model
    
    def _mp_fn(self, rank, normalize):
        flags = self.flags
        torch.set_default_tensor_type(torch.FloatTensor)
        
        if self.get_dataset is None: 
            train_ds, val_ds = cached_dataset(self.cache_train_loc, self.cache_val_loc)
        else: train_ds, val_ds = None, None
            
        loss, f1score, data, pred, targ, model = self.train_tpu(train_ds, val_ds, normalize=normalize)
        if rank == 0: torch.save(model.state_dict(), self.flags["save_path"])
            
    def train(self, normalize=False):
        import gc
        gc.collect()
        xmp.spawn(self._mp_fn, args=(normalize, ), nprocs=self.flags["num_cores"], start_method="fork")

    def train_loop_fn(self, loader, model, mean, std):
        tracker = xm.RateTracker()
        model.train()
        
        running_loss = 0.0
        total_samples = 0
        
        # get one batch and put onto same device. Alternatively can put in loop. 
        for data, target in loader: break  
        if mean is not None: mean, std = mean.to(data.device), std.to(data.device)
        
        for data, target in tqdm(loader):
            self.opt.zero_grad()
#             data, target = data.to(device), target.to(device)
            if mean is not None: data = (data - mean) / std
            if self.chls: data = data.to(memory_format=torch.channels_last)
            
            output = model(data)
            loss = self.criterion(output, target.to(torch.float32))
#                 preds = (torch.sigmoid(output).data > 0.5).to(torch.float32)
            
            loss.backward()
            xm.optimizer_step(self.opt)
            self.update_lr_mom(self.lr_scheds[self.idx_s].step(),
                                self.mom_scheds[self.idx_s].step())
                        
            if self.lr_scheds[self.idx_s].is_done: self.idx_s += 1
            tracker.add(self.flags["bs"])
            
            running_loss += loss.item() * data.size(0)
            total_samples += data.size(0)

        return running_loss, total_samples

    def test_loop_fn(self, loader, model, average, mean, std):
        total_samples = 0
        running_loss, f1Score = 0.0, 0.0
        model.eval()
        
        # get one batch and put onto same device. Alternatively can put in loop. 
        for data, target in loader: break
        if mean is not None: mean, std = mean.to(data.device), std.to(data.device)
        
        for data, target in tqdm(loader):
#             data, target = data.to(device), target.to(device)
            if mean is not None: data = (data - mean) / std
            if self.chls: data = data.to(memory_format=torch.channels_last)
            
            output = model(data)
            loss = self.criterion(output, target.to(torch.float32))
            preds = (torch.sigmoid(output).data > 0.5).to(torch.float32)
            
            total_samples += data.size(0)
            running_loss += loss.item() * data.size(0)
            
            target = target.cpu().to(torch.int).numpy()
            preds = preds.cpu().to(torch.int).numpy()
            
            f1Score += f1_score(target, preds, average=average) * data.size(0)
            
        epoch_loss = running_loss / total_samples
        epoch_f1score = f1Score / total_samples
        
#             print(f"""
#                 Val loss: {epoch_loss} | 
#                 Val F1Score: {epoch_f1score} | 
#             """, flush=True)
        return epoch_loss, epoch_f1score, data, preds, target

# %% [markdown]
# `weighted` can result in F-score **that is not between precision and recall**. URL: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

# %% [markdown]
# # Predict Batch
# For batch prediction

# %% [code] {"jupyter":{"outputs_hidden":false}}
def predict_batch(model, dl_test, paths=None, normalize=False, device=None):
    """
    Batch prediction. 
    File predicted will be according to how the data is input. It should be in the order 
    of your input paths. 
    :args: 
        model (PyTorch model): state dict should already be loaded by yourself. 
        dl_test: (PyTorch Dataloaders) Dataloaders containing test files. 
        paths (Python List of PosixPaths/str): paths to test files. (currently not used)
        normalize (bool/tuple): Whether to normalize data or not. If you want
            True, please pass in tuple of mean and std for non-Imagenet normalization. 
    :return: (Tensor) list of predictions in the order of paths. 
    """
    with torch.no_grad():
        model = model.to(device)
        model.eval()
        if normalize == True: mean, std = normalize_fn()
        elif type(normalize) == tuple: 
            mean, std = normalize
            normalize = True
            
        total = None
        
        for k, data in enumerate(tqdm(dl_test)):
            if normalize: data = (data - mean) / std
            output = model(data.to(device))
            if total is None: total = copy.deepcopy(output.cpu())
            else: total = torch.cat([total, output.cpu()], axis=0)
            
    return total

# CHanges with multiprocessing: if the class to fetch data can return the idx, then we can 
# use it to search back the filename based on the idx. Idx as integers can be collated together 
# unlike file name whom are str/object. 
