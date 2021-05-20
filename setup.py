import setuptools

requirements = ["tensorflow >= 2.3", "matplotlib", "pandas", "seaborn", "nibabel", "scikit-learn", "opencv-python", "scipy", "numpy==1.19.2"]

config = {
        'name': "CNN-Segmentation",
        'version': "0.1",
        'author': "Davide Marzorati",
        'author_email': "davide.marzorati@polimi.it",
        'packages': setuptools.find_packages(),
        'url': "https://github.com/ltebs-polimi/cnn-segmentation",
        'install_requires': requirements,
        'tests_require': ['nose']
}

setuptools.setup(**config)