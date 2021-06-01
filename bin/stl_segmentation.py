"""
This is a Python script to be used when performing STL 
segmentation in 3D starting from the femur and tibia
true labels and predicted labels.
"""

from loguru import logger
import math
import numpy as np
from pathlib import Path
import pyvista as pv
import re
from scipy.spatial import KDTree
from sys import platform
from sympy.geometry.point import Point3D
from sympy.geometry.line import Line
from sympy.geometry.plane import Plane
from stl import mesh

if 'darwin' in platform:
    DATA_PATH = Path('/Users') / 'Dado' / 'Desktop' / 'Polimi' / 'BoneData'
else:
    DATA_PATH = Path('E:\\') / 'Dev' / 'BoneSegmentation' / 'Data'

# Flag to enable or disbale display of meshes
DISPLAY = True

# STL File femur name format
femur_stl_name = 'ct_femur_predicted_{:04d}_m{:02d}.stl'
# STL File tibia name format
tibia_stl_name = 'ct_tibia_predicted_{:04d}_m{:02d}.stl'

# Regular expression to match patient folder
patient_regex = 'Patient\d\d\d\d'
patient_pattern = re.compile(patient_regex)

model = '00'


def compute_center_of_mass(data):
    """
    Compute the center of mass of the given mesh.

    Args:
        - the mesh for which the center of mass must be computed
    """
    assert (np.array(data).shape[1] == 3)

    cm = np.array([np.mean(data[:, 0]), np.mean(
        data[:, 1]), np.mean(data[:, 2])])
    return cm


def bounding_box_3d(points):
    """
    Compute bounding box of points. This function returns the bounding box
    of a set of points. 

    Args:
        - points: an array with shape (n,3) containing points coordinates.
    Returns:
        - box: tuple with shape (1,6) containing (xmin, xmax, ymin, ymax, zmin, zmax) 
    """

    assert (np.array(points).shape[1] == 3)
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0])
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1])
    zmin = np.min(points[:, 2])
    zmax = np.max(points[:, 2])
    return np.array([xmin, xmax, ymin, ymax, zmin, zmax])


def distance_point_to_plane(plane_cm, plane_normal, point):
    plane_n = plane_normal.reshape(3, 1)
    point_r = point.reshape(3, 1)
    plane_cm_r = plane_cm.reshape(3, 1)
    d = - np.dot(np.transpose(plane_n), (plane_cm_r - point_r))
    return d[0][0]


for patient in DATA_PATH.iterdir():
    # For each patient predicted
    if ((re.match(patient_pattern, patient.name) == None)):
        # Check that we have a valid folder name
        continue
    # Get femur and tibia stl file names
    patient_number = int(patient.name[-4:])
    femur_stl_patient = patient / \
        femur_stl_name.format(patient_number, int(model))
    tibia_stl_patient = patient / \
        tibia_stl_name.format(patient_number, int(model))
    if (not femur_stl_patient.exists()):
        logger.debug('Femur stl file not found')
        continue
    if (not tibia_stl_patient.exists()):
        logger.debug('Tibia stl file not found')
        continue

    logger.info(f'Reading femur stl from {femur_stl_patient.name}')
    logger.info(f'Reading tibia stl from {tibia_stl_patient.name}')

    if (DISPLAY):
        p = pv.Plotter()

    # Read femur and tibia STL files
    femur = pv.read(femur_stl_patient)
    tibia = pv.read(tibia_stl_patient)

    if (DISPLAY):
        logger.info('Plotting femur and tibia meshes')
        p.add_mesh(femur, color='blue', show_edges=False)
        p.add_mesh(tibia, color='red', show_edges=False)
        p.title = 'Left Knee'

    # Compute center of mass
    femur_cm = compute_center_of_mass(femur.points)
    # Compute bounding box of femur
    box_bb = bounding_box_3d(femur.points)

    # Shaft axis
    shaft_axis = np.array([0, 0, 1])
    if (DISPLAY):
        # Show shaft line
        lines = np.hstack(([femur_cm[0], femur_cm[1], femur_cm[2]-100],
                           [femur_cm[0], femur_cm[1], femur_cm[2]+100]))
        p.add_lines(lines)

    # Othogonal planes
    axial_plane = pv.Plane(
        femur_cm, direction=shaft_axis, i_size=150, j_size=150)
    sagitall_plane = pv.Plane(femur_cm, direction=np.array(
        [1, 0, 0]), i_size=150, j_size=150)
    if (DISPLAY):
        p.add_mesh(axial_plane, color='white',
                   opacity=0.2, label='Axial Plane')
        p.add_mesh(sagitall_plane, color='white',
                   opacity=0.2, label='Sagittal Plane')

    # Initial AnteriorPosterior direction
    initial_antero_posterior_axis = np.array([0, 1, 0])

    # Initial frontal plane
    initial_antero_posterior_plane = pv.Plane(femur_cm,
                                              direction=initial_antero_posterior_axis,
                                              i_size=150, j_size=150)
    if (DISPLAY):
        p.add_mesh(initial_antero_posterior_plane,
                   color='black', opacity=0.2, label='AP Plane')

    # Find anterior and posterior points
    max_distance = -1000000
    for femural_point in femur.points:
        # Compute distance from plane
        d = distance_point_to_plane(
            femur_cm, initial_antero_posterior_axis, femural_point)
        if d > max_distance:
            max_distance = d
            posterior_point = femural_point

    max_distance = 1000000
    for femural_point in femur.points:
        # Compute distance from plane
        d = distance_point_to_plane(
            femur_cm, initial_antero_posterior_axis, femural_point)
        if d < max_distance:
            max_distance = d
            anterior_point = femural_point

    print(posterior_point)
    print(anterior_point)

    if (DISPLAY):
        p.add_point_labels(posterior_point, [
                           "Post Point"], point_color='white', point_size=20)
        p.add_point_labels(
            anterior_point, ["Ant Point"], point_color='red', point_size=20)
        p.add_legend()
        p.add_axes()
        p.show()

    raise

    # Check if point is lateral or medial
    # posterior_point_distance =
    posterior_point_distance = sagittal_plane.distance(
        Point3D(posterior_point, evaluate=False))

    if (posterior_point_distance < 0):
        # Find sagittale plane
        initial_sagitall_axis = np.array([1, 0, 0])
        initial_sagitall_plane = Plane(
            femur_cm, normal_vector=initial_sagitall_axis)
        max_distance = -100000
        for idx, point in enumerate(femur_3D_points):
            d = initial_sagitall_plane.distance(point)
            if (d > max_distance):
                max_distance = d
                index_point_max_sagitall_distance = idx
    else:
        initial_sagitall_axis = np.array([1, 0, 0])
        initial_sagitall_plane = Plane(
            femur_cm, normal_vector=initial_sagitall_axis)
        max_distance = 100000
        for idx, point in enumerate(femur_3D_points):
            d = initial_sagitall_plane.distance(point)
            if (d < max_distance):
                max_distance = d
                index_point_max_sagitall_distance = idx

    # Get the point on sagitall axis
    sagitall_point = femur.points[index_point_max_sagitall_distance]
    print(sagitall_point)
    if (DISPLAY):
        p.add_point_labels(
            sagitall_point, ["Sagitall Point"], point_color='magenta', point_size=20)
        p.show()

    # Get direction of frontal axis
    if (posterior_point_distance > 0):
        frontal_direction = sagitall_point - posterior_point
    else:
        frontal_direction = postPoint - sagitall_point
    print(frontal_direction)
    frontal_direction = frontal_direction/np.linalg.norm(frontal_direction)

    print(frontal_direction)
    """
    %% Othogonalization
    iAntPostAxis        = cross(shaftAxis,ifrontalDirection);
    iAntPostAxis        = iAntPostAxis/norm(iAntPostAxis);            
    ifrontal_plane      = createPlane(postPoint,iAntPostAxis);

    maxDist = +100000;
    for i = 1 : size(SurfaceDataPoints,1)
    point = SurfaceDataPoints(i,:);
    d = distancePointPlane(point, ifrontal_plane);
        if (d<maxDist)
        maxDist = d;
        indexPointMaxDistance = i;
        end
    end
    sagPoint = SurfaceDataPoints(indexPointMaxDistance,:);
    """
