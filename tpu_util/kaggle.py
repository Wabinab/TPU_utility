# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-25T05:56:43.862555Z","iopub.execute_input":"2021-08-25T05:56:43.863001Z","iopub.status.idle":"2021-08-25T05:56:43.875244Z","shell.execute_reply.started":"2021-08-25T05:56:43.862917Z","shell.execute_reply":"2021-08-25T05:56:43.874047Z"}}
def setup_kaggle():
    os.system("curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py")
    print("Download complete step 1 of 2")
    os.system("python pytorch-xla-env-setup.py --version 1.7 --apt-packages libomp5 libopenblas-dev")
    print("Setup complete step 2 of 2")
    
    clear_output()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-25T05:56:43.877682Z","iopub.execute_input":"2021-08-25T05:56:43.878075Z","iopub.status.idle":"2021-08-25T05:58:01.695632Z","shell.execute_reply.started":"2021-08-25T05:56:43.878040Z","shell.execute_reply":"2021-08-25T05:58:01.694297Z"}}
import os
from IPython.display import clear_output

try: import torch_xla
except Exception: setup_kaggle()

from .tpu_utility_1 import *
from .tpu_cache_ds_utils import *
from .other_utils import *
from ._lr_finder import *