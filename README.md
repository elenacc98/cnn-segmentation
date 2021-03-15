# CNN Segmentation Package
This repository contains the source code of the segmentation package. This package provides utility functions to be used for CNN-based segmentation of medical images.

## Submodules:
#### Segmentation 
- `segmentation.callbacks`: callbacks used during the training of the network
- `segmentation.cnn`: this module allows to define CNN-based models in a dynamic way
- `segmentation.losses`: this module allows to define loss functions to be used for segmentation tasks
- `segmentation.metrics`: in this module, metrics to be used during the training of the network are implemented
- `segmentation.utils`: utils functions used for various purposes

#### Segmentation.preprocess
Submodule to preprocess dicom or Nifti files to adapt data for training and optimize memory
- `segmentation.preprocess.main_preprocess`: this module contains the main preprocessing functions 
- `segmentation.preprocess.Cycles`: this module contains example functions to loop over files and call the main functions
- `segmentation.preprocess.utils`: utils functions used for various purposes

Current lab students working on the project:
    - Alberto Faglia: [AlbiFag](https://github.com/AlbiFag)

## Installation
You can install the package directly from GitHub:
`pip install +git https://github.com/ltebs-polimi/cnn-segmentation`ß

## Documentation
You can generate your local documentation by running: `make docs` to update the documentation of the project. Updated documentation can be found in the `docs` folder.
Documentation of the package can be found at the following link: [https://ltebs-polimi.github.io/cnn-segmentation/](https://ltebs-polimi.github.io/cnn-segmentation/)