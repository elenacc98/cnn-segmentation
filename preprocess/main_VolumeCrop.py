
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

def Volume_Crop():
    """
    Crop slices in x,y,z direction to minimize the amount of background voxels, to optimize memory.
    It saves at "filenameCT" .nii cropped files
    Returns:
    """
    # Set directories
    setDirVariables()

    # Options
    b_display = 0

    # Sets
    InDCMmSet = 'knee'
    InStlSet = 'stl'
    InSurfSet1 = 'ct_femur_'
    InSurfSet2 = 'ct_tibia_'
    InSurfSet3 = 'ct_fibula_'
    InSurfSet4 = 'ct_patella_'
    OutSurfSet = 'crop_knee'

    os.chdir(mainInputDataDirectoryNAS)

    fp = open('case.txt', 'r+')
    casePatient = fp.read()
    casePatient = int(casePatient)
    fp.close()
    print('Patient no. {:04d}'.format(casePatient))

    # Read excel file to get patients' codes
    xlsName = os.path.join(mainInputDataDirectoryNAS, 'Case_statistics.xlsx')
    #name = pandas.ExcelFile(xlsName)
    name = xlrd.open_workbook(xlsName)
    sheet = name.sheet_by_index(0)
    rows = sheet.nrows
    study = [sheet.cell_value(i, 0) for i in range(1, rows)]
    patientCode = study[casePatient - 1]


    ## Patient folder
    mainInputPatientDirectoryNAS = mainInputDataDirectoryNAS + '/OriginalData/' + patientCode

    # Load DICOM files
    mainInputDicomDirectory = mainInputPatientDirectoryNAS + '/dicom/'
    if os.path.isdir(mainInputDicomDirectory + '/ct/'):
        mainInputDicomDirectory = mainInputDicomDirectory + '/ct/' + InDCMmSetdicom + '/'
    else:
        mainInputDicomDirectory = mainInputDicomDirectory + InDCMmSetdicom + '/'
    VolumeCT = loadDicomVolume(mainInputDicomDirectory)
    VolumeCT.volumeData = np.flip(VolumeCT.volumeData, axis=2)  # flip on z axis to restore natural positions of bones

    # Read stl and display
    filename1 = InSurfSet1 + study[casePatient - 1] + '.stl'
    filename2 = InSurfSet2 + study[casePatient - 1] + '.stl'
    filename3 = InSurfSet3 + study[casePatient - 1] + '.stl'
    filename4 = InSurfSet4 + study[casePatient - 1] + '.stl'

    mainInputStlDirectory = mainInputPatientDirectoryNAS + '/' + InStlSet + '/'
    os.chdir(mainInputStlDirectory)

    meshes = []
    # Femur
    if os.path.isfile(filename1):
        meshes.append(mesh.Mesh.from_file(filename1))
    # Tibia
    if os.path.isfile(filename2):
        meshes.append(mesh.Mesh.from_file(filename2))
    # Fibula
    if os.path.isfile(filename3):
        meshes.append(mesh.Mesh.from_file(filename3))
    # Patella
    if os.path.isfile(filename4):
        meshes.append(mesh.Mesh.from_file(filename4))

    if len(meshes) == 2:
        new_mesh = mesh.Mesh(np.concatenate([meshes[0].data, meshes[1].data]))
    elif len(meshes) == 3:
        new_mesh = mesh.Mesh(np.concatenate([meshes[0].data, meshes[1].data]))
        new_mesh = mesh.Mesh(np.concatenate([new_mesh.data, meshes[2].data]))
    elif len(meshes) == 4:
        new_mesh = mesh.Mesh(np.concatenate([meshes[0].data, meshes[1].data]))
        new_mesh = mesh.Mesh(np.concatenate([new_mesh.data, meshes[2].data]))
        new_mesh = mesh.Mesh(np.concatenate([new_mesh.data, meshes[3].data]))
    elif len(meshes) == 0:
        print('*** Stl file not found ***')
        return
    else:
        new_mesh = meshes[0]

    CropVolumeCT = VolumeCT
    deltaNumberofSlices = 2

    # CROP THE VOLUME IN ALL THREE DIRECTIONS Z, Y, X

    indexCoord = 0  # Sagittal slicing direction (X direction) (columns)
    indexVolume = 1
    minValue = new_mesh.min_[indexCoord]
    maxValue = new_mesh.max_[indexCoord]
    CropVolumeCT = cropVolume(CropVolumeCT, deltaNumberofSlices, minValue, maxValue, indexCoord, indexVolume)

    indexCoord = 1  # frontal slicing direction (Y direction) (rows)
    indexVolume = 0
    minValue = new_mesh.min_[indexCoord]
    maxValue = new_mesh.max_[indexCoord]
    CropVolumeCT = cropVolume(CropVolumeCT, deltaNumberofSlices, minValue, maxValue, indexCoord, indexVolume)

    indexCoord = 2  # axial slicing direction (Z direction) (slices)
    indexVolume = indexCoord
    minValue = new_mesh.min_[indexCoord]
    maxValue = new_mesh.max_[indexCoord]
    CropVolumeCT = cropVolume(CropVolumeCT, deltaNumberofSlices, minValue, maxValue, indexCoord, indexVolume)

    volumeData = np.transpose(CropVolumeCT.volumeData, [1, 0, 2])
    volumeOffset = CropVolumeCT.volumeOffset
    affine = np.eye(4)
    niiCropVolumeCT = nib.Nifti1Image(volumeData, affine)
    niiCropVolumeCT.header.set_slope_inter(CropVolumeCT.rescaleSlope, CropVolumeCT.rescaleIntercept)
    #niiCropVolumeCT.header['scl_slope'] = int(CropVolumeCT.rescaleSlope)
    #niiCropVolumeCT.header['scl_inter'] = int(CropVolumeCT.rescaleIntercept)
    niiCropVolumeCT.header.set_qform(affine, 1)
    niiCropVolumeCT.header.set_zooms(CropVolumeCT.voxelSize)
    niiCropVolumeCT.header['qoffset_x'] = volumeOffset[0]
    niiCropVolumeCT.header['qoffset_y'] = volumeOffset[1]
    niiCropVolumeCT.header['qoffset_z'] = volumeOffset[2]
    # *** OFFSET SETTING MISSING ***

    # Save file to nii format
    patientFolder = 'Patient{:03d}'.format(casePatient)
    outputPatientDirectory = mainOutputDataDirectoryLoc + '/' + patientFolder + '/'
    mainOutputPatientDirectory = outputPatientDirectory
    os.chdir(mainOutputPatientDirectory)
    filenameCT = 'volumeCT_' + OutSurfSet + '_{:04d}_py.nii'.format(casePatient)
    nib.nifti1.save(niiCropVolumeCT, filenameCT)


