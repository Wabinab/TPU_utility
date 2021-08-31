# TPU_utility
Utility for TPU with PyTorch XLA.

One more examples please check [this kaggle notebook](https://www.kaggle.com/wabinab/test-tpu-training).

We have move the packages to folder `tpu_util`. For installation, call `!python -m pip install git+https://github.com/Wabinab/TPU_utility.git@main` on colab for nightly releases. 

Update: currently we have published the packages on PyPI for stable versions. You could install with `pip install tpu-util`. 

Currently, if by any chance you are running TPU on Google Cloud Platform, the dependencies **are not installed automatically**. One doesn't have any access to GCP so one isn't sure how it works, and hence you are responsible for installing requirements.txt and all other requirements by yourself, or clone colab/kaggle environment to your machine and setup from there. 

Sample contains data taken from [this](https://www.kaggle.com/wabinab/gnet-cqt-shannon-alt) dataset. This image is a spectrogram, it's not easy to understand; however it is small enough to be use as test cases. 
