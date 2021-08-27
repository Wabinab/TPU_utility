import setuptools

setuptools.setup(
    name="tpu_util",
    version=1,
    description="TPU utility with PyTorch XLA",
    long_description=open("README.md").read(),
    package=["tpu_util"],
    install_requires= ["pip", "packaging"],
    python_requires=">=3.6",
)