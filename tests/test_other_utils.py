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


model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
for param in model.parameters(): param.requires_grad_(False)
model.fc = nn.Linear(num_features, 1)  
# actually requires sigmoid but we're not using model here so ignore. 

criterion = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=FLAGS["lr"])


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


def test_linear_scheduler_correct_value():
    steps = 10
    lin_sched = LinearScheduler(opt, 10, steps)
    all_lr = np.zeros((steps, ))

    for k, i in enumerate(range(steps)):
        all_lr[k] = lin_sched.get_lr()[0]
        lin_sched.step()

    assert (np.round(all_lr).astype(np.uint8) == np.arange(10) + 1).all()


def test_exponential_scheduler_correct_value():
    steps, end_lr = 10, 10
    exp_sched = ExponentialScheduler(opt, end_lr, steps)
    all_lr = np.zeros((steps, ))

    comparable = np.array([FLAGS["lr"] * ((end_lr / FLAGS["lr"]) ** (pct / steps)) for pct in range(1, steps + 1)])

    for k, i in enumerate(range(steps)):
        all_lr[k] = exp_sched.get_lr()[0]
        exp_sched.step()

    assert (all_lr == comparable).all()


def test_cosine_scheduler_correct_value():
    steps, end_lr = 10, 10
    cos_sched = CosineScheduler(opt, end_lr, steps)
    all_lr = np.zeros((steps, ))

    comparable = np.array([end_lr + (FLAGS["lr"] - end_lr) / 2 * (np.cos(np.pi * (pct / steps)) + 1) for pct in range(1, steps + 1)])

    for k, i in enumerate(range(steps)):
        all_lr[k] = cos_sched.get_lr()[0]
        cos_sched.step()

    assert (all_lr == comparable).all()


def test_annealing_no_functions_correctly():
    assert annealing_no(1, 10, 0.1) == 1
    assert annealing_no(2, 10, 0.1) == 2
    assert annealing_no(1, 2, 0.1) == 1
    assert annealing_no(1, 10, 0.5) == 1


@pytest.fixture
def start_pct_end(tmpdir): return 0.001, 0.2, 10


def test_annealing_linear_functions_correctly(start_pct_end):
    start, pct, end = start_pct_end
    assert annealing_linear(start, end, pct) == pytest.approx(2, 0.01)
    assert annealing_linear(1, end, pct) == 2.8
    assert annealing_linear(start, 100, pct) == pytest.approx(20, 0.001)
    assert annealing_linear(start, end, 0.5) == pytest.approx(5, 0.01)


def test_annealing_exp_functions_correctly(start_pct_end):
    start, pct, end = start_pct_end
    assert annealing_exp(start, end, pct) == start * ((end / start) ** pct)
    assert annealing_exp(1, end, pct) == 1 * ((end / 1) ** pct)
    assert annealing_exp(start, 100, pct) == start * ((100 / start) ** pct)
    assert annealing_exp(start, end, 0.5) == start * ((end / start) ** 0.5)


def test_annealing_cos_functions_correctly(start_pct_end):
    start, pct, end = start_pct_end
    assert annealing_cos(start, end, pct) == end + (((start - end) / 2) * (1 + np.cos(np.pi * pct)))
    assert annealing_cos(1, end, pct) == end + (((1 - end) / 2) * (1 + np.cos(np.pi * pct)))
    assert annealing_cos(start, 100, pct) == 100 + (((start - 100) / 2) * (1 + np.cos(np.pi * pct)))
    assert annealing_cos(start, end, 0.5) == end + (((start - end) / 2) * (1 + np.cos(np.pi * 0.5)))


# normalize_fn
def test_normalize_fn_default_correct_value():
    mean, std = normalize_fn()

    assert (mean.squeeze() == torch.Tensor([123.6750, 116.2800, 103.5300])).all()
    assert (std.squeeze() == torch.Tensor([58.3950, 57.1200, 57.3750])).all()


def test_normalize_fn_default_correct_shape():
    mean, std = normalize_fn()

    assert mean.shape == torch.Size([1, 3, 1, 1])
    assert std.shape == torch.Size([1, 3, 1, 1])


def test_normalize_fn_with_data(call_dl):
    for data, target in call_dl["train"]: break
    data_out = normalize_fn(data)
    mean, std = normalize_fn()

    assert (data_out == (data - mean) / std).all()


def test_normalize_fn_with_data_shape_retains(call_dl):
    for data, target in call_dl["train"]: break
    data_out = normalize_fn(data)

    assert data_out.shape == data.shape


def test_normalize_fn_calculated_input_false(call_dl):
    for data, target in call_dl["train"]: break
    mean, std = normalize_fn()

    data_out = normalize_fn(data, mean, std, calculated_input=True)
    data_baseline = normalize_fn(data)

    assert (data_out == data_baseline).all()


def test_normalize_fn_float_mean_works():
    mean, std = normalize_fn(mean=0.495)

    one_mean = 0.495 * 255
    assert (mean.squeeze() == torch.Tensor([one_mean, one_mean, one_mean])).all()


def test_normalize_fn_float_std_works():
    mean, std = normalize_fn(std=0.38)
    one_std = 0.38 * 255

    assert (std.squeeze() == torch.Tensor([one_std, one_std, one_std])).all()


def test_normalize_fn_int_mean_works():
    mean, std = normalize_fn(mean=3)

    one_mean = 3 * 255
    assert (mean.squeeze() == torch.Tensor([one_mean, one_mean, one_mean])).all()


def test_normalize_fn_int_std_works():
    mean, std = normalize_fn(std=3)

    one_std = 3 * 255
    assert (std.squeeze() == torch.Tensor([one_std, one_std, one_std])).all()


def test_normalize_fn_max_pixel_works():
    max_pixel = 2**16
    mean, std = normalize_fn(max_pixel=max_pixel)

    mean_comp = np.array([0.485, 0.456, 0.406], dtype=np.float32) * max_pixel
    std_comp = np.array([0.229, 0.224, 0.225], dtype=np.float32) * max_pixel

    assert mean.squeeze().numpy() == pytest.approx(mean_comp, 1e-6)
    assert std.squeeze().numpy() == pytest.approx(std_comp, 1e-6)


def test_max_pixel_can_be_float():
    mean, std = normalize_fn(max_pixel=0.496)
    compare = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 0.496
    assert mean.squeeze().numpy() == pytest.approx(compare, 1e-6)


def test_normalize_fn_str_doesnt_work_mean(): 
    with pytest.raises(np.core._exceptions.UFuncTypeError):
        mean, std = normalize_fn(mean="0.64")


def test_normalize_fn_str_doesnt_work_std(): 
    with pytest.raises(np.core._exceptions.UFuncTypeError):
        mean, std = normalize_fn(std="0.64")


def test_normalize_fn_str_doesnt_work_max_pixel(): 
    with pytest.raises(np.core._exceptions.UFuncTypeError):
        mean, std = normalize_fn(max_pixel="0.64")


def test_normalize_fn_with_another_shape_mean_works():
    mean, std = normalize_fn(mean=(0.5, 0.6, 0.7, 0.8, 0.9))

    assert mean.shape == torch.Size([1, 5, 1, 1])


