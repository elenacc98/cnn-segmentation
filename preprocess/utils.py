"""
Utils functions
"""

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


class VolumeStruct:
    """Volumetric data to be stored in nii format"""

    def __init__(self, data, voxelSize, volumeDim, volumeOffset,
                 bitStored, rescaleIntercept, rescaleSlope, sliceLocation):
        self.volumeData = data
        self.voxelSize = voxelSize
        self.volumeDim = volumeDim
        self.volumeOffset = volumeOffset
        self.bitStored = bitStored
        self.rescaleIntercept = rescaleIntercept
        self.rescaleSlope = rescaleSlope
        self.sliceLocation = sliceLocation


class IndexTracker(object):
    """To plot and scroll slices of 3D volumetric data"""

    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()



def setDirVariables():
    rootDirectoryNAS = '/Volumes/Dati/Users/Data-out/'  # NAS
    rootDirectoryLoc = '/Users/AlbertoFaglia/Desktop/POLIMI Biomedica/TESI/Dati_prova/'
    mainZipDirectory = 'E:\\Users\\Pietro\\Documents\\Work\\Company\\_Medacta\\ModelliStatistici\\Data_Aprile 2016\\'
    mainInputDataDirectoryNAS = os.path.join(rootDirectoryNAS, '/MEDACTA_2/')
    mainInputDataDirectoryLoc = os.path.join(rootDirectoryLoc, 'Data/MEDACTA_2/')
    NASDirectory = 'Z:\\Users\\Data-out\\MEDACTA\\'
    mainOutputDataDirectoryLoc = os.path.join(rootDirectoryLoc, '/Data/MEDACTA_2/preprocessedData)
    mainCodeDirectory = os.path.join(rootDirectoryLoc, 'Matlab/')


def loadDicomVolume(VolumeDir):
    listFilesDCM = []
    for dirName, subDirList, fileList in os.walk(VolumeDir):
        for filename in fileList:
            listFilesDCM.append(os.path.join(dirName, filename))
    if len(listFIlesDCM) == 0:
        print('No dicom files for patient')
        return
    listFilesDCM.sort()  # If 'reverse=True' we don't need to adjust volumeOffset[2] and to flip slices in z
    # direction (in Volume_Crop) but slice location is then different from the Matlab one

    RefDs = pydicom.read_file(listFilesDCM[0])
    RefDs1 = pydicom.read_file(listFilesDCM[1])

    # Load dimensions
    VolumeDims = (int(RefDs.Rows), int(RefDs.Columns), len(listFilesDCM))

    # Load spacing values
    loc0 = RefDs.SliceLocation
    loc1 = RefDs1.SliceLocation
    PixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), abs(round(float(loc1 - loc0), 4)))
    bitStored = RefDs.BitsAllocated
    rescaleIntercept = RefDs.RescaleIntercept
    rescaleSlope = RefDs.RescaleSlope

    # Compute offset
    offset2 = RefDs.ImagePositionPatient[2]
    offset2 = offset2 - PixelSpacing[2] * (VolumeDims[2]-1)
    volumeOffset = (float(RefDs.ImagePositionPatient[0]), float(RefDs.ImagePositionPatient[1]), offset2)

    # The array is sized based on 'ConstPixelDims'
    ArrayDicom = np.zeros(VolumeDims, dtype=RefDs.pixel_array.dtype)
    # Loop through all the DICOM files
    for filenameDCM in listFilesDCM:
        # Read the file
        ds = pydicom.read_file(filenameDCM)
        # Store the raw image data
        ArrayDicom[:, :, listFilesDCM.index(filenameDCM)] = ds.pixel_array

    VolumeCT = VolumeStruct(ArrayDicom, PixelSpacing, VolumeDims, volumeOffset,
                            bitStored, rescaleIntercept, rescaleSlope, loc0)

    return VolumeCT


def loadNiiVolume(volumeDir, dicomDir):

    """ Get nii Volumes and all metadata. Dicom directory also needed"""

    nii = nib.load(volumeDir)
    volumeData = nii.get_fdata()
    voxelSize = nii.header.get_zooms()
    volumeDim = volumeData.shape
    volumeOffset = (round(float(nii.header['qoffset_x']), 3),
                    round(float(nii.header['qoffset_y']), 3),
                    round(float(nii.header['qoffset_z']), 3))
    # rescaleSlope, rescaleIntercept = nii.header.get_slope_inter() # HEADER RETURNS None VALUES FOR BOTH !!
    rescaleSlope, rescaleIntercept, sliceLocation, bitStored = getMetadata(dicomDir)

    VolumeCT = VolumeStruct(volumeData, voxelSize, volumeDim, volumeOffset,
                            bitStored, rescaleIntercept, rescaleSlope, sliceLocation)
    return VolumeCT



def loadNiiVolume1(volumeDir):

    """ Function to load nii Volume without storing intercept and slope and sliceLocation,
 for which dicom files are needed """

    nii = nib.load(volumeDir)
    volumeData = nii.get_fdata()
    voxelSize = nii.header.get_zooms()
    volumeDim = volumeData.shape
    volumeOffset = (round(float(nii.header['qoffset_x']), 3),
                    round(float(nii.header['qoffset_y']), 3),
                    round(float(nii.header['qoffset_z']), 3))
    bitStored = nii.header['bitpix']
    rescaleSlope = None
    rescaleIntercept = None
    sliceLocation = None
    # rescaleSlope, rescaleIntercept = nii.header.get_slope_inter() # HEADER RETURNS None VALUES FOR BOTH !!
    # rescaleSlope, rescaleIntercept, sliceLocation, bitStored = getMetadata(dicomDir)

    VolumeCT = VolumeStruct(volumeData, voxelSize, volumeDim, volumeOffset,
                            bitStored, rescaleIntercept, rescaleSlope, sliceLocation)
    return VolumeCT


def getMetadata(dicomDir):

    """ Get slope, intercept and slice location from dicom files.
    They are not accessible through nii FIles in loadNiiVolume"""

    listFilesDCM = []
    for dirName, subDirList, fileList in os.walk(dicomDir):
        for filename in fileList:
            listFilesDCM.append(os.path.join(dirName, filename))
    listFilesDCM.sort()

    RefDs = pydicom.read_file(listFilesDCM[0])
    loc0 = RefDs.SliceLocation

    rescaleIntercept = RefDs.RescaleIntercept
    rescaleSlope = RefDs.RescaleSlope
    bitStored = RefDs.BitsAllocated

    return rescaleSlope, rescaleIntercept, loc0, bitStored


def normVolumeScan(VolumeIn):
    VolumeOut = VolumeIn

    if 'bitStored' in dir(VolumeOut):
        if VolumeOut.bitStored == 12:
            minV = VolumeOut.volumeData.min()
            VolumeOut.volumeData = VolumeOut.volumeData - minV
            VolumeOut.volumeData = (VolumeOut.volumeData) / 4096
        elif VolumeOut.bitStored == 16:
            minV = VolumeOut.volumeData.min()
            indexZ = [VolumeOut.volumeData <= -1024]
            VolumeOut.volumeData[indexZ] = -1024
            VolumeOut.volumeData = VolumeOut.volumeData + 1024
            maxV = VolumeOut.volumeData.max()
            if maxV > 4096:
                maxV = 4096
            VolumeOut.volumeData = (VolumeOut.volumeData) / maxV

        else:
            print('Impossible to recognize datatype')

        return VolumeOut


def resampleVolumeScan(VolumeIn, resample):
    VolumeOut = VolumeIn
    sizeVolIn = [voxSize * (volDim - 1) for voxSize, volDim in zip(VolumeIn.voxelSize, VolumeIn.volumeDim)]
    sizeVoxelOut = [round(float(sizeVol / scale), 3) for sizeVol, scale in zip(sizeVolIn, resample)]

    resampledVolume = resize(VolumeOut.volumeData, resample, anti_aliasing=True)

    VolumeOut.volumeData = resampledVolume
    VolumeOut.voxelSize = (sizeVoxelOut[0], sizeVoxelOut[1], sizeVoxelOut[2])
    VolumeOut.volumeDim = resample
    return VolumeOut


def mesh2vol(mesh, heights, volumeOffset, voxelSize, volumeDim):
    dirZ = [0, 0, 1]
    contours = []
    for ip in range(volumeDim[2]):

        p = [float(volumeOffset[0]), float(volumeOffset[1]), float(volumeOffset[2])]
        p[2] = p[2] + ip * voxelSize[2]
        lines, face_index = trimesh.intersections.mesh_plane(mesh, dirZ, p, return_faces=True)

        if lines.shape[0] != 0:
            unique1 = np.unique(lines[:, 0, :], axis=0)
            unique2 = np.unique(lines[:, 1, :], axis=0)
            contour_points = np.concatenate((unique1, unique2))
            contour_points = np.unique(contour_points, axis=0)
            contours.append(contour_points)
            # lines[ip] = contour_points
        else:
            contours.append(np.array([]))
            # lines[ip] = np.array([])

    return contours


def dice_coeff(pyLabel, outSet):
    value = 0
    counter = 0
    for ip in range(pyLabel.shape[2]):
        namefile = 'Label_' + outSet + '_{:03d}'.format(ip + 1)
        bimg_mat = loadmat(namefile)
        matLabel = bimg_mat['variable']
        matLabel = matLabel / 255

        intersection = np.logical_and(matLabel, pyLabel[:, :, ip])
        if int(matLabel.sum() + pyLabel[:, :, ip].sum()) != 0:
            temp = (2. * intersection.sum()) / (matLabel.sum() + pyLabel[:, :, ip].sum())
            value = value + temp
            counter = counter + 1

    dice = value / counter
    return dice


def cropVolume(Volume, deltaNumberofSlices, minValue, maxValue, indexCoord, indexVolume):

    CropVolume = Volume

    if indexCoord == 0:  # indexVolume = 1

        numberOfSlicetoRemoveDown = int(np.ceil((minValue - Volume.volumeOffset[indexCoord]) / Volume.voxelSize[indexVolume]))
        numberOfSlicetoRemoveUp = int(np.ceil((maxValue - Volume.volumeOffset[indexCoord]) / Volume.voxelSize[indexVolume]))
        indexToRemove1 = np.arange(0, (numberOfSlicetoRemoveDown - deltaNumberofSlices), dtype=int)
        indexToRemove2 = np.arange((numberOfSlicetoRemoveUp + deltaNumberofSlices), Volume.volumeDim[indexVolume],
                                   dtype=int)
        indexToRemove = np.concatenate((indexToRemove1, indexToRemove2))
        if len(indexToRemove) < Volume.volumeDim[indexVolume]:
            VolumeDataCropped = np.delete(CropVolume.volumeData, [indexToRemove], axis=indexVolume)

            CropVolume.volumeData = VolumeDataCropped
            CropVolume.volumeDim = VolumeDataCropped.shape
            CropVolume.volumeOffset = (CropVolume.volumeOffset[indexCoord] +
                                         (numberOfSlicetoRemoveDown - deltaNumberofSlices) * Volume.voxelSize[indexVolume],
                                         CropVolume.volumeOffset[1],
                                         CropVolume.volumeOffset[2])

    if indexCoord == 1:  # indexVolume = 0

        numberOfSlicetoRemoveDown = int(np.ceil((minValue - Volume.volumeOffset[indexCoord]) / Volume.voxelSize[indexVolume]))
        numberOfSlicetoRemoveUp = int(np.ceil((maxValue - Volume.volumeOffset[indexCoord]) / Volume.voxelSize[indexVolume]))
        indexToRemove1 = np.arange(0, (numberOfSlicetoRemoveDown - deltaNumberofSlices), dtype=int)
        indexToRemove2 = np.arange((numberOfSlicetoRemoveUp + deltaNumberofSlices), Volume.volumeDim[indexVolume],
                                   dtype=int)
        indexToRemove = np.concatenate((indexToRemove1, indexToRemove2))
        if len(indexToRemove) < Volume.volumeDim[indexVolume]:
            VolumeDataCropped = np.delete(CropVolume.volumeData, [indexToRemove], axis=indexVolume)

            CropVolume.volumeData = VolumeDataCropped
            CropVolume.volumeDim = VolumeDataCropped.shape
            CropVolume.volumeOffset = (CropVolume.volumeOffset[0],
                                         CropVolume.volumeOffset[indexCoord] +
                                         (numberOfSlicetoRemoveDown - deltaNumberofSlices) * Volume.voxelSize[indexVolume],
                                         CropVolume.volumeOffset[2])

    if indexCoord == 2: # indexVolume = 2

        numberOfSlicetoRemoveDown = int(np.ceil((minValue - Volume.volumeOffset[indexCoord]) / Volume.voxelSize[indexVolume]))
        numberOfSlicetoRemoveUp = int(np.ceil((maxValue - Volume.volumeOffset[indexCoord]) / Volume.voxelSize[indexVolume]))
        indexToRemove1 = np.arange(0, (numberOfSlicetoRemoveDown - deltaNumberofSlices), dtype=int)
        indexToRemove2 = np.arange((numberOfSlicetoRemoveUp + deltaNumberofSlices), Volume.volumeDim[indexVolume],
                                   dtype=int)
        indexToRemove = np.concatenate((indexToRemove1, indexToRemove2))
        if len(indexToRemove) < Volume.volumeDim[indexVolume]:
            VolumeDataCropped = np.delete(CropVolume.volumeData, [indexToRemove], axis=indexVolume)

            CropVolume.volumeData = VolumeDataCropped
            CropVolume.volumeDim = VolumeDataCropped.shape
            CropVolume.volumeOffset = (CropVolume.volumeOffset[0],
                                         CropVolume.volumeOffset[1],
                                         CropVolume.volumeOffset[indexCoord] +
                                         (numberOfSlicetoRemoveDown - deltaNumberofSlices) * Volume.voxelSize[indexVolume])
    return CropVolume

def compareLabels():

    outSize = [128, 128, 128]
    templateSize = str(outSize[0]) + '_' + str(outSize[1]) + '_' + str(outSize[2])

    # Sets
    InSurfSet = 'volumeLabel_'
    InDCMmSetdicom = 'knee'

    templateOut = InSurfSet + 'reshape_knee_' + templateSize
    sum_dice = 0
    counter_tot = 0

    for casePatient in range(1,6):
        mainInputPy = os.path.join(rootDirectory, 'Data/MEDACTA/Patient{:03d}/{:s}_{:03d}.nii'.format(casePatient, templateOut, casePatient))
        mainInputMat = os.path.join(rootDirectory, 'Data/MAT/Patient{:03d}/{:s}_{:03d}.nii'.format(casePatient, templateOut, casePatient))
        dicomDir = os.path.join(rootDirectory, 'Data/MEDACTA/Patient{:03d}/knee'.format(casePatient))

        LabelPy = loadNiiVolume(mainInputPy, dicomDir)
        LabelMat = loadNiiVolume(mainInputMat, dicomDir)

        Py = LabelPy.volumeData
        Mat = LabelMat.volumeData
        indexpy = np.where(Py > 1)
        indexmat = np.where(Mat > 1)
        Py[indexpy] = 1
        Mat[indexmat] = 1

        value = 0
        counter = 0
        for ip in range(Py.shape[2]):
            intersection = np.logical_and(Py[:, :, ip], Mat[:, :, ip])
            if int(Mat[:, :, ip].sum() + Py[:, :, ip].sum()) != 0:
                temp = (2. * intersection.sum()) / (Mat[:, :, ip].sum() + Py[:, :, ip].sum())
                value = value + temp
                counter = counter + 1

        dice = value / counter
        print('Dice coeff. for Patient no {:03d}: {:.8f}'.format(casePatient, dice))
        sum_dice = sum_dice + dice
        counter_tot = counter_tot + 1

    dice_mean = sum_dice / counter_tot
    print('Mean dice coeff. for patients from 1 to {:03d} is : {:.8f}'.format(counter_tot, dice_mean))
