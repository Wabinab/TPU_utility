def setup_colab():
    os.system("pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl")
    print("Download complete.")
    clear_output()


import os
from IPython.display import clear_output

try: import torch_xla
except Exception: setup_colab()

from .tpu_utility_1 import *
from .tpu_cache_ds_utils import *
from .other_utils import *
from ._lr_finder import *