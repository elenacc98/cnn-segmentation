"""
The mesh submodule provides functions to be used to process STL files and perform
three-dimensional segmentation of the STL files.
"""

from stl import mesh

def get_surface_center_of_mass(data):
  """
  Compute the center of mass of the given surface.
  
  Args:
    - data: the three-dimensional data 
  """
  assert np.array(data).shape[1] == 3
  cm = np.array([np.mean(data[:,0]), np.mean(data[:,1]), np.mean(data[:,2])])
  return cm
