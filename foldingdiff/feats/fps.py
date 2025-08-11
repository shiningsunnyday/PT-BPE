import numpy as np
from pyzernike import ZernikeDescriptor

def voxelize(coords, grid_size=64, padding=2.0):
    """
    Convert 3D points into a binary occupancy grid.
    - coords: (n_points,3) numpy array
    - grid_size: number of voxels per axis
    - padding: extra Ångstroms around the min/max before voxelizing
    """
    # Compute bounding box
    mins = coords.min(axis=0) - padding
    maxs = coords.max(axis=0) + padding
    # Voxel spacing
    spacing = (maxs - mins) / (grid_size - 1)
    # Initialize grid
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    # Map each atom into voxel coords
    ijk = ((coords - mins) / spacing).astype(int)
    # Clip to valid range
    ijk = np.clip(ijk, 0, grid_size - 1)
    # Set occupancy
    for (i, j, k) in ijk:
        grid[i, j, k] = 1.0
    return grid


def compute_3d_zernike(grid, order=8):
    """
    Compute the 3D Zernike descriptor up to a given order.
    """
    # Fit descriptor
    zd = ZernikeDescriptor.fit(data=grid, order=order)
    # Extract the invariant coefficients
    coeffs = zd.get_coefficients()
    return coeffs


