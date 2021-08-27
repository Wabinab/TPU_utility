"""
Test cases for other utilities
Created on: 27 August 2021.
"""
import albumentations
import jpeg4py as jpeg
import glob
import numpy as np 
import pandas as pd 
import os 
from pathlib import Path
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score
from tqdm.auto import tqdm

import torch
import torch.nn as nn 
import torch.utils.data as D 
from torchvision import models

# Setup for the whole test case process
# Code taken from https://www.kaggle.com/wabinab/submission-gnet
# and another code taken from a private notebook https://www.kaggle.com/wabinab/gnet-tpu-train
# and another private notebook https://www.kaggle.com/wabinab/gnet-create-ds
FLAGS = {
    "data_dir": Path("../sample"),
    "num_workers": os.cpu_count(),
    "lr": 0.001,
    "bs": 8,
    "num_epochs": 2,
    "seed": 1447,
}


class GnetDataset(D.Dataset):
    def __init__(self, parent_path, df=None, shuffle=False, seed=None, 
                transforms=None, split=None):
        """
        parent_path: (pathlib.Path) Parent path of images. 
        df: (numpy) (This haven't changed the 'df' name, but is a numpy array of test files)
        shuffle: (Boolean) Shuffle dataset? Default: False. 
        seed: (integer) seed. Default: None. 
        transforms: (Albumentations) Transformation to images, default: None. 
        split: (python list) Splits index for train test split. 
        """
        self.df_np = df
        self.parent_path = parent_path
        self.seed = seed
        self.transforms = transforms

        if seed:
            np.random.seed(seed)
            torch.manual_seed(seed)

        if type(split) != type(None): self.df_np = self.df_np[split]

        def __len__(self): return len(self.df_np)

        def __getitem__(self, idx):
            item_path = self.parent_path / self.df_np[idx]  # image filenames
            image = jpeg.JPEG(item_path).decode()

            if self.transforms is not None:
                image = self.transforms(**{"image": image})
                image = image["image"]

            image = torch.from_numpy(image).permute(2, 0, 1)
            
            return torch.Tensor([idx]).type(torch.uint8), image


test_path = [Path(path) for path in sorted(glob.glob("../sample/*.jpg"))]