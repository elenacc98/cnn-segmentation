
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
    # Resolution
    resolution = [1, 1, 1]
    # Dimension
    Dimension = [192, 192, 192]

    # Sets
    InDCMmSet = 'knee'
    InNiiSet = 'crop_knee'
    OutSurfSet = 'reshape_knee_' + str(Dimension[0]) + '_' + str(Dimension[1]) + '_' + str(Dimension[2])

    os.chdir(mainInputDataDirectoryLoc)

    fp = open('case.txt', 'r+')
    casePatient = fp.read()
    casePatient = int(casePatient)
    # casePatient = 3
    fp.close()
    print('Patient no. {:04d}'.format(casePatient))

    # Read excel file to get patients' codes
    xlsName = os.path.join(mainInputDataDirectoryLoc, 'Case_statistics.xlsx')
    # name = pandas.ExcelFile(xlsName)
    name = xlrd.open_workbook(xlsName)
    sheet = name.sheet_by_index(0)
    rows = sheet.nrows
    study = [sheet.cell_value(i, 0) for i in range(1, rows)]
    patientCode = study[casePatient - 1]

    ## Read volume nii
    patientDirectory = 'Patient{:04d}'.format(casePatient)
    mainInputPatientDirectoryLoc = mainInputDataDirectoryLoc + '/preprocessedData/' + patientDirectory + '/'
    mainInputPatientDirectoryNAS = mainInputDataDirectoryNAS + '/OriginalData/' + patientCode
    mainInputDicomDirectory = mainInputPatientDirectoryNAS + '/dicom/'
    if os.path.isdir(mainInputDicomDirectory + '/ct/'):
        mainInputDicomDirectory = mainInputDicomDirectory + '/ct/' + InDCMmSetdicom + '/'
    else:
        mainInputDicomDirectory = mainInputDicomDirectory + InDCMmSetdicom + '/'

    os.chdir(mainInputPatientDirectoryLoc)

    # Read NII
    niiFilename = 'volumeCT_' + InNiiSet + '_{:04d}.nii'.format(casePatient)
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
    VolumeResampled = resampleVolumeScan(VolumeCT, Dimension)
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

    os.chdir(mainInputPatientDirectoryLoc)

    # Save nii
    filenameCT = 'volumeCT_' + OutSurfSet + '_{:04d}_py.nii'.format(casePatient)
    nib.nifti1.save(niiResampledVolumeCT, filenameCT)

    os.chdir(mainCodeDirectory)




