
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
    # Set directories
    setDirVariables()

    # Options
    b_display = 0

    # Sets
    InDCMmSet = 'knee'
    InSurfSet1 = 'ct_femur_'
    InSurfSet2 = 'ct_tibia_'
    InSurfSet3 = 'ct_fibula_'
    InSurfSet4 = 'ct_patella_'
    OutSurfSet = 'crop_knee'

    os.chdir(mainInputDataDirectory)

    fp = open('case.txt', 'r+')
    casePatient = fp.read()
    casePatient = int(casePatient)
    #casePatient = 5
    fp.close()
    print('Patient no. {}'.format(casePatient))

    # Read excel file to get patients' cpdes
    xlsName = os.path.join(mainInputDataDirectory, 'Preop Data.xlsx')
    name = pandas.ExcelFile(xlsName)
    name = xlrd.open_workbook(xlsName)
    sheet = name.sheet_by_index(0)
    rows = sheet.nrows
    study = [sheet.cell_value(i, 0) for i in range(1, rows)]

    # Get back into the code directory
    os.chdir(mainCodeDirectory)

    # % if (casePatient>100)
    # %    mainInputDataDirectory = NASDirectory;
    # % end

    ## Patient folder
    volumeFormat = 'dcm'
    mainPatientDirectory = 'Patient{:03d}'.format(casePatient)
    mainPatientDirectory = mainInputDataDirectory + '/' + mainPatientDirectory + '/'

    # Modifiche Prof. Cerveri per nuova organizzazione dati
    # patientCode = str(study[casePatient-1])
    # mainInputPatientDirectory = mainInputDataDirectory + '/OriginalData/' + patientCode

    # Load DICOM files
    mainInputDicomDirectory = mainPatientDirectory + '/' + InDCMmSet + '/'
    VolumeCT = loadDicomVolume(mainInputDicomDirectory)
    VolumeCT.volumeData = np.flip(VolumeCT.volumeData, axis=2)  # flip on z axis to restore natural positions of bones

    if VolumeCT.volumeData.size == 0:
        print('No dicom file for patient #{:03d}_{}'.format(casePatient, patientCode))
        return

    # Read stl and display
    filename1 = InSurfSet1 + study[casePatient - 1] + '.stl'
    filename2 = InSurfSet2 + study[casePatient - 1] + '.stl'
    filename3 = InSurfSet3 + study[casePatient - 1] + '.stl'
    filename4 = InSurfSet4 + study[casePatient - 1] + '.stl'

    # Get into the zip directory
    os.chdir(mainPatientDirectory)
    # Femur
    my_mesh1 = mesh.Mesh.from_file(filename1)
    # Tibia
    my_mesh2 = mesh.Mesh.from_file(filename2)
    # Add Prof. Cerveri's edit for Patella and Fibula
    new_mesh = mesh.Mesh(np.concatenate([my_mesh1.data, my_mesh2.data]))

    CropVolumeCT = VolumeCT
    deltaNumberofSlices = 5

    # CROP THE VOLUME IN ALL THREE DIRECTIONS Z, Y, X
    indexCoord = 2  # axial slicing direction
    indexVolume = indexCoord
    minValue = new_mesh.min_[indexCoord]
    maxValue = new_mesh.max_[indexCoord]
    CropVolumeCT = cropVolume(CropVolumeCT, deltaNumberofSlices, minValue, maxValue, indexCoord, indexVolume)

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


    volumeData = np.transpose(CropVolumeCT.volumeData, [1, 0, 2])
    volumeOffset = CropVolumeCT.volumeOffset
    affine = np.eye(4)
    niiCropVolumeCT = nib.Nifti1Image(volumeData, affine)
    niiCropVolumeCT.header.set_slope_inter(CropVolumeCT.rescaleSlope, CropVolumeCT.rescaleIntercept)
    niiCropVolumeCT.header.set_qform(affine, 1)
    niiCropVolumeCT.header.set_zooms(CropVolumeCT.voxelSize)
    niiCropVolumeCT.header['qoffset_x'] = volumeOffset[0]
    niiCropVolumeCT.header['qoffset_y'] = volumeOffset[1]
    niiCropVolumeCT.header['qoffset_z'] = volumeOffset[2]
    # *** OFFSET SETTING MISSING ***

    # Save file to nii format
    patientFolder = 'Patient{:03d}'.format(casePatient)
    outputPatientDirectory = mainOutputDataDirectory + '/' + patientFolder + '/'
    mainOutputPatientDirectory = outputPatientDirectory
    os.chdir(mainOutputPatientDirectory)
    filenameCT = 'volumeCT_' + OutSurfSet + '_{:03d}.nii'.format(casePatient)
    nib.nifti1.save(niiCropVolumeCT, filenameCT)

    os.chdir(mainCodeDirectory)
