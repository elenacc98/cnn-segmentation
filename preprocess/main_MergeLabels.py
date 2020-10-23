
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

def Merge_Labels():
    setDirVariables()

    # Options
    b_display = 0
    # Output size

    outSize = [128, 128, 128]
    templateSize = str(outSize[0]) + '_' + str(outSize[1]) + '_' + str(outSize[2])

    # Sets
    InDCMmSetdicom = 'knee'
    anatomy1 = 'femur'
    InSurfSet1 = 'dist' + anatomy1 + '_' + templateSize
    anatomy2 = 'tibia'
    InSurfSet2 = 'prox' + anatomy2 + '_' + templateSize
    anatomy3 = 'patella'
    InSurfSet3 = anatomy3 + '_' + templateSize
    anatomy4 = 'fibula'
    InSurfSet4 = anatomy4 + '_' + templateSize

    templateOut = 'reshape_knee_' + templateSize

    os.chdir(mainInputDataDirectoryLoc)

    fp = open('case.txt', 'r+')
    casePatient = fp.read()
    casePatient = int(casePatient)
    fp.close()
    print('Patient no. {:04d}'.format(casePatient))

    # Read excel file to get patients' codes
    xlsName = os.path.join(mainInputDataDirectoryLoc, '/Case Statistics.xlsx')
    # name = pandas.ExcelFile(xlsName)
    name = xlrd.open_workbook(xlsName)
    sheet = name.sheet_by_index(0)
    rows = sheet.nrows
    study = [sheet.cell_value(i, 0) for i in range(1, rows)]
    patientCode = study[casePatient - 1]

    mainPatientDirectory = 'Patient{:03d}'.format(casePatient)
    mainInputPatientDirectoryLoc = mainInputDataDirectoryLoc + '/preprocessedData/' + mainPatientDirectory + '/'
    mainInputPatientDirectoryNAS = mainInputDataDirectoryNAS + '/OriginalData/' + patientCode
    mainInputDicomDirectory = mainInputPatientDirectoryNAS + '/' + InDCMmSetdicom + '/'

    os.chdir(mainInputPatientDirectoryLoc)

    # Read label volume1
    pathName1 = 'volumeLabel_' + InSurfSet1 + '_{:04d}.nii'.format(casePatient)
    # pathName1 = mainPatientDirectory + pathName1
    VolumeLabel1 = loadNiiVolume(pathName1, mainInputDicomDirectory)

    # Read label volume2
    pathName2 = 'volumeLabel_' + InSurfSet2 + '_{:04d}.nii'.format(casePatient)
    # pathName2 = mainPatientDirectory + pathName2
    VolumeLabel2 = loadNiiVolume(pathName2, mainInputDicomDirectory)

    # Read label volume3
    pathName3 = 'volumeLabel_' + InSurfSet3 + '_{:04d}.nii'.format(casePatient)
    # pathName3 = mainPatientDirectory + pathName1
    if os.path.exists(pathName3):
        VolumeLabel3 = loadNiiVolume(pathName3, mainInputDicomDirectory)

    # Read label volume4
    pathName4 = 'volumeLabel_' + InSurfSet4 + '_{:04d}.nii'.format(casePatient)
    # pathName4 = mainPatientDirectory + pathName1
    if os.path.exists(pathName4):
        VolumeLabel4 = loadNiiVolume(pathName4, mainInputDicomDirectory)

    ## MERGE WANTED LABELS
    # Increase index of the label
    indexLabel = np.where(VolumeLabel2.volumeData == 1)
    VolumeLabel2.volumeData[indexLabel] = VolumeLabel2.volumeData[indexLabel] + 1

    VolumeLabelJoin = VolumeLabel1
    VolumeLabelJoin.volumeData = VolumeLabelJoin.volumeData + VolumeLabel2.volumeData

    # Make nii Label file
    volumeData = VolumeLabelJoin.volumeData.astype(np.short)
    voxelSize = VolumeLabelJoin.voxelSize
    volumeOffset = VolumeLabelJoin.volumeOffset

    affine = np.eye(4)
    niiVolumeLabelJoin = nib.Nifti1Image(volumeData, affine)
    niiVolumeLabelJoin.header.set_slope_inter(VolumeLabelJoin.rescaleSlope, VolumeLabelJoin.rescaleIntercept)
    niiVolumeLabelJoin.header.set_qform(affine, 1)
    niiVolumeLabelJoin.header.set_zooms(voxelSize)
    niiVolumeLabelJoin.header['qoffset_x'] = volumeOffset[0]
    niiVolumeLabelJoin.header['qoffset_y'] = volumeOffset[1]
    niiVolumeLabelJoin.header['qoffset_z'] = volumeOffset[2]

    # Save joined labels
    filenameOut = 'volumeLabel_' + templateOut + '_{:04d}.nii'.format(casePatient)
    pathName = mainInputPatientDirectoryLoc + filenameOut
    nib.nifti1.save(niiVolumeLabelJoin, pathName)

    os.chdir(mainCodeDirectory)
def Merge_Labels():
    setDirVariables()

    # Options
    b_display = 0
    # Output size

    outSize = [128, 128, 128]
    templateSize = str(outSize[0]) + '_' + str(outSize[1]) + '_' + str(outSize[2])

    # Sets
    InDCMmSetdicom = 'knee'
    anatomy1 = 'femur'
    InSurfSet1 = 'dist' + anatomy1 + '_' + templateSize
    anatomy2 = 'tibia'
    InSurfSet2 = 'prox' + anatomy2 + '_' + templateSize
    anatomy3 = 'patella'
    InSurfSet3 = anatomy3 + '_' + templateSize
    anatomy4 = 'fibula'
    InSurfSet4 = anatomy4 + '_' + templateSize

    templateOut = 'reshape_knee_' + templateSize

    os.chdir(mainInputDataDirectoryLoc)

    fp = open('case.txt', 'r+')
    casePatient = fp.read()
    casePatient = int(casePatient)
    fp.close()
    print('Patient no. {:04d}'.format(casePatient))

    # Read excel file to get patients' codes
    xlsName = os.path.join(mainInputDataDirectoryLoc, '/Case Statistics.xlsx')
    # name = pandas.ExcelFile(xlsName)
    name = xlrd.open_workbook(xlsName)
    sheet = name.sheet_by_index(0)
    rows = sheet.nrows
    study = [sheet.cell_value(i, 0) for i in range(1, rows)]
    patientCode = study[casePatient - 1]

    mainPatientDirectory = 'Patient{:03d}'.format(casePatient)
    mainInputPatientDirectoryLoc = mainInputDataDirectoryLoc + '/preprocessedData/' + mainPatientDirectory + '/'
    mainInputPatientDirectoryNAS = mainInputDataDirectoryNAS + '/OriginalData/' + patientCode
    mainInputDicomDirectory = mainInputPatientDirectoryNAS + '/' + InDCMmSetdicom + '/'

    os.chdir(mainInputPatientDirectoryLoc)

    # Read label volume1
    pathName1 = 'volumeLabel_' + InSurfSet1 + '_{:04d}.nii'.format(casePatient)
    # pathName1 = mainPatientDirectory + pathName1
    VolumeLabel1 = loadNiiVolume(pathName1, mainInputDicomDirectory)

    # Read label volume2
    pathName2 = 'volumeLabel_' + InSurfSet2 + '_{:04d}.nii'.format(casePatient)
    # pathName2 = mainPatientDirectory + pathName2
    VolumeLabel2 = loadNiiVolume(pathName2, mainInputDicomDirectory)

    # Read label volume3
    pathName3 = 'volumeLabel_' + InSurfSet3 + '_{:04d}.nii'.format(casePatient)
    # pathName3 = mainPatientDirectory + pathName1
    if os.path.exists(pathName3):
        VolumeLabel3 = loadNiiVolume(pathName3, mainInputDicomDirectory)

    # Read label volume4
    pathName4 = 'volumeLabel_' + InSurfSet4 + '_{:04d}.nii'.format(casePatient)
    # pathName4 = mainPatientDirectory + pathName1
    if os.path.exists(pathName4):
        VolumeLabel4 = loadNiiVolume(pathName4, mainInputDicomDirectory)

    ## MERGE WANTED LABELS
    # Increase index of the label
    indexLabel = np.where(VolumeLabel2.volumeData == 1)
    VolumeLabel2.volumeData[indexLabel] = VolumeLabel2.volumeData[indexLabel] + 1

    VolumeLabelJoin = VolumeLabel1
    VolumeLabelJoin.volumeData = VolumeLabelJoin.volumeData + VolumeLabel2.volumeData

    # Make nii Label file
    volumeData = VolumeLabelJoin.volumeData.astype(np.short)
    voxelSize = VolumeLabelJoin.voxelSize
    volumeOffset = VolumeLabelJoin.volumeOffset

    affine = np.eye(4)
    niiVolumeLabelJoin = nib.Nifti1Image(volumeData, affine)
    niiVolumeLabelJoin.header.set_slope_inter(VolumeLabelJoin.rescaleSlope, VolumeLabelJoin.rescaleIntercept)
    niiVolumeLabelJoin.header.set_qform(affine, 1)
    niiVolumeLabelJoin.header.set_zooms(voxelSize)
    niiVolumeLabelJoin.header['qoffset_x'] = volumeOffset[0]
    niiVolumeLabelJoin.header['qoffset_y'] = volumeOffset[1]
    niiVolumeLabelJoin.header['qoffset_z'] = volumeOffset[2]

    # Save joined labels
    filenameOut = 'volumeLabel_' + templateOut + '_{:04d}.nii'.format(casePatient)
    pathName = mainInputPatientDirectoryLoc + filenameOut
    nib.nifti1.save(niiVolumeLabelJoin, pathName)

    os.chdir(mainCodeDirectory)