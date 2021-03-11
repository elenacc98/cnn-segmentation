
from __future__ import print_function
import os
import numpy as np
import trimesh
from stl import mesh
import matplotlib.pyplot as plt
import pandas
import xlrd
from scipy.io import loadmat
import pydicom
import nibabel as nib
import plotly.graph_objects as go
from scipy.ndimage.morphology import binary_fill_holes, binary_closing, binary_erosion, binary_dilation
from skimage.transform import resize
from skimage.morphology.selem import square, disk
def Volume_Reshape():

    setDirVariables()

    # Options
    b_display = 0
    export_cubic_voxel = 0
    # Resolution
    resolution = [1, 1, 1]
    # Dimension
    Dimension = [128, 128, 128]

    # Sets
    InDCMmSet = 'knee'
    InNiiSet = 'crop_knee'
    OutSurfSet = 'reshape_knee_' + str(Dimension[0]) + '_' + str(Dimension[1]) + '_' + str(Dimension[2])
    OutSurfSet1 = 'reshape_knee_cubic_' + str(Dimension[0]) + '_' + str(Dimension[1]) + '_' + str(Dimension[2])

    os.chdir(mainInputDataDirectory)

    fp = open('case.txt', 'r+')
    casePatient = fp.read()
    casePatient = int(casePatient)
    # casePatient = 3
    fp.close()
    print('Patient no. {}'.format(casePatient))

    os.chdir(mainCodeDirectory)

    patientDirectory = 'Patient{:03d}'.format(casePatient)
    mainPatientDirectory = mainInputDataDirectory + '/' + patientDirectory + '/'
    mainInputDicomDirectory = mainPatientDirectory + '/' + InDCMmSet + '/'

    os.chdir(mainPatientDirectory)

    # Read NII
    niiFilename = 'volumeCT_' + InNiiSet + '_{:03d}.nii'.format(casePatient)
    VolumeCT = loadNiiVolume(niiFilename, mainInputDicomDirectory)
    # Normalize
    VolumeCT = normVolumeScan(VolumeCT)

    os.chdir(mainCodeDirectory)

    # Get data
    volumeDim = VolumeCT.volumeDim
    volumeVoxSize = VolumeCT.voxelSize
    dimX = (volumeDim[0] - 1) * volumeVoxSize[0]
    dimY = (volumeDim[1] - 1) * volumeVoxSize[1]
    dimZ = (volumeDim[2] - 1) * volumeVoxSize[2]
    nVolumeDim = [round(dimX / resolution[0] + 1), round(dimX / resolution[1] + 1), round(dimY / resolution[2] + 1)]

    # Resample to new dimension
    VolumeResampled = resampleVolumeScan(VolumeCT, Dimension) ##### THIS COMMAND ALSO CHANGES VolumeCT !!!! ******
    # Normalize between 0 and 1
    VolumeResampled.volumeData = VolumeResampled.volumeData - VolumeResampled.volumeData.min()
    VolumeResampled.volumeData = VolumeResampled.volumeData/VolumeResampled.volumeData.max()
    volumeOffset = VolumeResampled.volumeOffset

    affine = np.eye(4)
    niiResampledVolumeCT = nib.Nifti1Image(VolumeResampled.volumeData, affine)
    niiResampledVolumeCT.header.set_slope_inter(VolumeResampled.rescaleSlope, VolumeResampled.rescaleIntercept)
    niiResampledVolumeCT.header.set_qform(affine, 1)
    niiResampledVolumeCT.header.set_zooms(VolumeResampled.voxelSize)
    niiResampledVolumeCT.header['qoffset_x'] = volumeOffset[0]
    niiResampledVolumeCT.header['qoffset_y'] = volumeOffset[1]
    niiResampledVolumeCT.header['qoffset_z'] = volumeOffset[2]

    # Cubic Voxel Volume with resolution defined above
    if export_cubic_voxel == 1:
        VolumeResampled1 = resampleVolumeScan(VolumeCT, nVolumeDim)

        VolumeResampled1.volumeData = VolumeResampled1.volumeData - VolumeResampled1.volumeData.min()
        VolumeResampled1.volumeData = VolumeResampled1.volumeData / VolumeResampled1.volumeData.max()
        volumeOffset1 = VolumeResampled1.volumeOffset

        niiResampledVolumeCT1 = nib.Nifti1Image(VolumeResampled1.volumeData, affine)
        niiResampledVolumeCT1.header.set_slope_inter(VolumeResampled1.rescaleSlope, VolumeResampled1.rescaleIntercept)
        niiResampledVolumeCT1.header.set_qform(affine, 1)
        niiResampledVolumeCT1.header.set_zooms(VolumeResampled1.voxelSize)
        niiResampledVolumeCT1.header['qoffset_x'] = volumeOffset1[0]
        niiResampledVolumeCT1.header['qoffset_y'] = volumeOffset1[1]
        niiResampledVolumeCT1.header['qoffset_z'] = volumeOffset1[2]

    os.chdir(mainPatientDirectory)

    # Save nii
    filenameCT = 'volumeCT_' + OutSurfSet + '_{:03d}.nii'.format(casePatient)
    nib.nifti1.save(niiResampledVolumeCT, filenameCT)

    if export_cubic_voxel == 1:
        filenameCT1 = 'volumeCT_' + OutSurfSet1 + '_{:03d}.nii'.format(casePatient)
        nib.nifti1.save(niiResampledVolumeCT1, filenameCT1)

    os.chdir(mainCodeDirectory)




