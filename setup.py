import setuptools

setuptools.setup(
    name="tpu_util",
    version="0.1.3",
    url="https://github.com/Wabinab/TPU_utility",
    description="TPU utility with PyTorch XLA",
    long_description=open("README.md").read(),
    packages=setuptools.find_packages(),
    install_requires= ["pip", "packaging"],
    python_requires=">=3.6",
)
