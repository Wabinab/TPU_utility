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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from tqdm.auto import tqdm

import torch
import torch.nn as nn 
import torch.utils.data as D 
from torchvision import models

import pytest
from tpu_util.other_utils import *

# Setup for the whole test case process
# Code taken from https://www.kaggle.com/wabinab/submission-gnet
# and another code taken from a private notebook https://www.kaggle.com/wabinab/gnet-tpu-train
# and another private notebook https://www.kaggle.com/wabinab/gnet-create-ds
FLAGS = {
    "data_dir": Path("./sample"),
    "num_workers": os.cpu_count(),
    "lr": 0.001,
    "bs": 8,
    "num_epochs": 2,
    "seed": 1447,
}


class GnetTestDataset(D.Dataset):
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

    def __len__(self): 
        return len(self.df_np)

    def __getitem__(self, idx):
        # item_path = self.parent_path / self.df_np[idx]  # image filenames
        item_path = self.df_np[idx]  # this requires changing depending on different cases. 
        image = jpeg.JPEG(item_path).decode()

        if self.transforms is not None:
            image = self.transforms(**{"image": image})
            image = image["image"]

        image = torch.from_numpy(image).permute(2, 0, 1)
        
        return torch.Tensor([idx]).type(torch.uint8), image


class GnetDataset(D.Dataset):
    def __init__(self, parent_path, df=None, shuffle=False,
                 seed=None, transforms=None, split=None):
        """
        parent_path: (pathlib.Path) Parent path of images. 
        df: (pandas.DataFrame) Dataframe containing labels and file path. 
        shuffle: (Boolean) Shuffle dataset? Default: False. 
        seed: (integer) seed. Default: None. 
        transforms: (Albumentations) Transformation to images, default: None. 
        split: (python list) Splits index for train test split. 
        """
        self.df_np = df.to_numpy()
        self.parent_path = parent_path
        self.seed = seed
        self.transforms = transforms
        
        if seed:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        if type(split) != type(None): self.df_np = self.df_np[split]
            
    def __len__(self): 
        return len(self.df_np)
    
    def __getitem__(self, idx):
        item = self.df_np[idx]
        item_path = self.parent_path / item[0]  # image filenames
        image = jpeg.JPEG(item_path).decode()
        
        if self.transforms is not None:
            image = self.transforms(**{"image": image})
            image = image["image"]
            
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW format
        target = torch.Tensor([item[1]])
        
        return image, target


test_path = np.array([Path(path) for path in sorted(glob.glob("./sample/*.jpg"))])
test_ds = GnetTestDataset(FLAGS["data_dir"], test_path, seed=FLAGS["seed"])

df = pd.read_csv("./sample/training_labels.csv")
df["id"] = df["id"] + ".jpg"

skf = StratifiedKFold(n_splits=5)
tdf_np = df.to_numpy()
X = tdf_np[:, 0]
y = tdf_np[:, 1].astype(np.uint8)
for train_index, test_index in skf.split(X, y): pass

def get_dataset():
    parent_path = FLAGS["data_dir"]

    train_ds = GnetDataset(parent_path, df, shuffle=True, seed=FLAGS["seed"],
                            split=train_index)
    val_ds = GnetDataset(parent_path, df, shuffle=False, seed=FLAGS["seed"],
                            split=test_index)

    return train_ds, val_ds


### ---------------------------------------------------- ###
### Starting test case ###
### ---------------------------------------------------- ###
@pytest.fixture
def call_dl(tmpdir):
    train_ds, val_ds = get_dataset()
    return dataloader(train_ds, val_ds, FLAGS, test_ds)


def test_dataloader_returns_correct_data_type():
    """For distributed=False. Unfortunately we cannot test distributed=True without TPU."""
    train_ds, val_ds = get_dataset()
    dls = dataloader(train_ds, val_ds, FLAGS)

    assert type(dls) == dict


def test_dataloader_have_correct_items(call_dl):
    """Should have train, val, and optionally test"""

    assert "train" in call_dl.keys()
    assert "val" in call_dl.keys()
    assert "test" in call_dl.keys()


def test_dataloader_each_child_item_correct_datatype(call_dl):
    for key in call_dl.keys():
        assert type(call_dl[key]) == torch.utils.data.dataloader.DataLoader


def test_dataloader_no_exception_raise_calling_one_batch_train(call_dl):
    try: 
        for one_batch in call_dl["train"]: break
    except Exception: pytest.fail("cannot call one batch")


def test_dataloader_no_exception_raise_calling_one_batch_val(call_dl):
    try:
        for one_batch in call_dl["val"]: break
    except Exception: pytest.fail("cannot call one batch")


def test_dataloader_no_exception_raise_calling_one_batch_test(call_dl):
    try:
        for one_batch in call_dl["test"]: break
    except Exception: pytest.fail("cannot call one batch")