import setuptools

config = {
        'name': "CNN-Segmentation",
        'version': "0.1",
        'author': "Davide Marzorati",
        'author_email': "davide.marzorati@polimi.it",
        'packages': setuptools.find_packages(),
        'url': "https://github.com/ltebs-polimi/cnn-segmentation",
        'install_requires': ["tensorflow >= 2.3", "matplotlib", "pandas"],
        'tests_require': ['nose']
}

setuptools.setup(**config)