"""
Cycles function that call the main_ functions to perform cropping, reshaping,
labeling and merging across all files of the directory.
"""

import os
from segmentation.preprocess.main_preprocess import Merge_Labels, Volume_Crop, Volume_Reshape, Volume_Label


def Cycle_Volume_Crop():
    """
    It calls cropping function for each patient volume to crop.
    Specify in range(_) the correct number of patient volumes (e.g. number of nii files) to preprocess.
    Returns:

    """

    for ik in range(5):
        os.chdir(mainInputDataDirectoryNAS)

        fp = open('case.txt', 'r+')
        fp.write('{}'.format(ik+1))
        fp.close()

        Volume_Crop()


def Cycle_Volume_Reshape():
    """
    It calls reshaping function for each patient volume to reshape.
    Specify in range(_) the correct number of patient volumes (e.g. number of nii files) to preprocess.
    Returns:

    """

    for ik in range(5):
        os.chdir(mainInputDataDirectoryLoc)

        fp = open('case.txt', 'r+')
        fp.write('{}'.format(ik+1))
        fp.close()

        Volume_Reshape()


def Cycle_Volume_Label():
    """
    It calls labeling function for each patient volume to be labeled.
    Specify in range(_) the correct number of patient volumes (e.g. number of nii files) to preprocess.
    Returns:

    """

    for ik in range(5):
        os.chdir(mainInputDataDirectoryLoc)

        fp = open('case.txt', 'r+')
        fp.write('{}'.format(ik+1))
        fp.close()

        Volume_Label()


def Cycle_Merge_Labels():
    """
    It calls merging function for each patient to which merge labels.
    Specify in range(_) the correct number of patient volumes (e.g. number of nii files) to preprocess.
    Returns:

    """

    for ik in range(5):
        os.chdir(mainInputDataDirectoryLoc)

        fp = open('case.txt', 'r+')
        fp.write('{}'.format(ik+1))
        fp.close()

        Merge_Labels()
