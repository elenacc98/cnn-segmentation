from setuptools import setup

setup(
        name="CNN-Segmentation",
        version="0.0.1",
        author="Davide Marzorati",
        author_email="davide.marzorati@polimi.iyìt",
        packages=["segmentation"],
        package_dir={"segmentation":"segmentation"},
        url="https://github.com/ltebs-polimi/cnn-segmentation",
        license="MIT",
        install_requires=[  "tensorflow >= 2.3"]
)