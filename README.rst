CNN Segmentation Package
==========================
This repository contains the source code of the segmentation package. This package provides utility functions to be used for CNN-based segmentation of medical images.
The code contained in the repository was used for data processing of the following paper:

"D. Marzorati, M. Sarti, L. Mainardi, A. Manzotti and P. Cerveri, "*Deep 3D Convolutional Networks to Segment Bones Affected by Severe Osteoarthritis in CT Scans for PSI-Based Knee Surgical Planning*," in IEEE Access, vol. 8, pp. 196394-196407, 2020, doi: [10.1109/ACCESS.2020.3034418](10.1109/ACCESS.2020.3034418)."

## Submodules:
#### Segmentation 
- `segmentation.callbacks`: callbacks used during the training of the network
- `segmentation.cnn`: CNN-based models to be used for network architecture
- `segmentation.losses`: loss functions for segmentation tasks
- `segmentation.metrics`: metrics to be used during the training of the network
- `segmentation.mesh`: 
- `segmentation.utils`: utils functions 

#### Segmentation.preprocess
The `segmentation.preprocess` is a submodule to preprocess dicom or Nifti (.nii) files to adapt data for training and optimize memory consumption:
- `segmentation.preprocess.main_preprocess`: this module contains the main preprocessing functions 
- `segmentation.preprocess.Cycles`: this module contains example functions to loop over files and call the main functions
- `segmentation.preprocess.utils`: utils functions used for various purposes

## Installation
You can install the package directly from GitHub:
`pip install +git https://github.com/dado93/cnn-segmentation`

## Documentation
You can generate your local documentation by running: `make docs` to update the documentation of the project. Updated documentation can be found in the `docs` folder.
Documentation of the package can be found at the following link: [https://dado93.github.io/cnn-segmentation/](https://dado93.github.io/cnn-segmentation/)

## TODO
- [ ] Documentation with Read The Docs
- [ ] New format for loss functions