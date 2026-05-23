"""OpenFOAM data processing: read mesh + velocity, interpolate onto regular grid."""

import os
import numpy as np
from numba import njit, prange
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from fluidfoam import readmesh, readfield


@njit(cache=True, parallel=True)
def _dumbbell_mask(Y, Z, L, R, H):
    """Boolean mask of the dumbbell cross-section (two circles + handle rect).

    ``Y`` and ``Z`` are the 2-D meshgrid coordinate arrays. Returns a ``bool``
    array of the same shape equal to ``left_circle | right_circle | handle``.
    """
    ny, nz = Y.shape
    half_L = L * 0.5
    half_H = H * 0.5
    R2 = R * R
    mask = np.empty((ny, nz), dtype=np.bool_)
    for a in prange(ny):
        for b in range(nz):
            yv = Y[a, b]
            zv = Z[a, b]
            yl = yv + half_L
            yr = yv - half_L
            left_circle = yl * yl + zv * zv <= R2
            right_circle = yr * yr + zv * zv <= R2
            rect_region = (abs(yv) <= half_L) and (abs(zv) <= half_H)
            handle = rect_region and (not left_circle) and (not right_circle)
            mask[a, b] = left_circle or right_circle or handle
    return mask


def build_openFoam_interp(case_dir, cache_file=None, force_reload=False,
                          Ny=1000, Nz=1000, Nx=100,
                          slope=-0.0075, L=4.0, R=1.0):
    """Build velocity interpolators from an OpenFOAM dumbbell-channel case.

    Parameters
    ----------
    case_dir : str or path-like
        Path to the OpenFOAM case directory.
    cache_file : str or path-like, optional
        Path for caching raw foam data as ``.npz``. If None, no caching is used.
    force_reload : bool
        If True, re-read from the case directory even if a cache exists.
    Ny, Nz, Nx : int
        Interpolation grid resolution.
    slope : float
        Taper slope for the handle height function.
    L : float
        Distance between circle centers.
    R : float
        Radius of each circle.

    Returns
    -------
    ux_interp, uy_interp, uz_interp : RegularGridInterpolator
        Velocity component interpolators on a regular (x, y, z) grid.
    """
    z_func = lambda x: np.where(x < 0, 1, slope * x + 1)

    z_min, z_max = -R - 0.1, R + 0.1
    y_min, y_max = -L / 2 - R - 0.1, L / 2 + R + 0.1
    x_min, x_max = 0, 120

    x_interp = np.linspace(x_min, x_max, Nx)
    z_interp = np.linspace(z_min, z_max, Nz)
    y_interp = np.linspace(y_min, y_max, Ny)

    if cache_file and os.path.exists(cache_file) and not force_reload:
        data = np.load(cache_file)
        x_foam = data['x']
        y_foam = data['y']
        z_foam = data['z']
        U_x_foam = data['Ux']
        U_y_foam = data['Uy']
        U_z_foam = data['Uz']
    else:
        time_dirs = sorted(
            [d for d in os.listdir(case_dir) if d.replace(".", "").isdigit()],
            key=float)
        latest_time = time_dirs[-1]
        x_foam, y_foam, z_foam = readmesh(case_dir, 'constant')
        U_foam = readfield(case_dir, latest_time, 'U')
        U_x_foam = U_foam[0][:-1]
        U_y_foam = U_foam[1][:-1]
        U_z_foam = U_foam[2][:-1]
        if cache_file:
            np.savez_compressed(
                cache_file,
                x=x_foam, y=y_foam, z=z_foam,
                Ux=U_x_foam, Uy=U_y_foam, Uz=U_z_foam)

    Y, Z = np.meshgrid(y_interp, z_interp, indexing='xy')
    slice_tol = 1
    U_x_3D = np.zeros((Nx, Ny, Nz))
    U_y_3D = np.zeros((Nx, Ny, Nz))
    U_z_3D = np.zeros((Nx, Ny, Nz))

    H = 1
    mask = _dumbbell_mask(Y, Z, float(L), float(R), float(H))

    for i, x_val in enumerate(x_interp):
        x_slice_loc = x_val + 10
        slice_mask = np.abs(x_foam - x_slice_loc) < slice_tol
        H = 2 * z_func(x_val)
        y_slice = y_foam[slice_mask]
        z_slice = z_foam[slice_mask]

        mask = _dumbbell_mask(Y, Z, float(L), float(R), float(H))

        y_can = np.abs(y_slice)
        z_can = np.abs(z_slice)

        y_void_all = Y[~mask]
        z_void_all = Z[~mask]

        y_full = np.concatenate([y_can, y_can, -y_can, -y_can, y_void_all])
        z_full = np.concatenate([z_can, -z_can, z_can, -z_can, z_void_all])

        U_x_slice = U_x_foam[slice_mask]
        U_y_slice = U_y_foam[slice_mask]
        U_z_slice = U_z_foam[slice_mask]
        U_x_full = np.concatenate([U_x_slice, U_x_slice, U_x_slice, U_x_slice, np.zeros_like(y_void_all)])
        U_y_full = np.concatenate([U_y_slice, U_y_slice, -U_y_slice, -U_y_slice, np.zeros_like(y_void_all)])
        U_z_full = np.concatenate([U_z_slice, -U_z_slice, U_z_slice, -U_z_slice, np.zeros_like(y_void_all)])

        Y_masked = Y[mask]
        Z_masked = Z[mask]

        coords = np.column_stack((y_full, z_full))
        _, unique_indices = np.unique(np.round(coords, decimals=6), axis=0, return_index=True)
        coords_clean = coords[unique_indices]

        U_x_clean = U_x_full[unique_indices]
        U_y_clean = U_y_full[unique_indices]
        U_z_clean = U_z_full[unique_indices]

        U_x_masked = griddata(coords_clean, U_x_clean, (Y_masked, Z_masked), method='linear', fill_value=0.0)
        U_y_masked = griddata(coords_clean, U_y_clean, (Y_masked, Z_masked), method='linear', fill_value=0.0)
        U_z_masked = griddata(coords_clean, U_z_clean, (Y_masked, Z_masked), method='linear', fill_value=0.0)

        U_x_grid = np.zeros_like(Y)
        U_x_grid[mask] = U_x_masked
        U_x_grid = gaussian_filter(U_x_grid, sigma=2.0)
        U_x_grid[~mask] = 0.0

        U_y_grid = np.zeros_like(Y)
        U_y_grid[mask] = U_y_masked
        U_y_grid = gaussian_filter(U_y_grid, sigma=2.0)
        U_y_grid[~mask] = 0.0

        U_z_grid = np.zeros_like(Y)
        U_z_grid[mask] = U_z_masked
        U_z_grid = gaussian_filter(U_z_grid, sigma=2.0)
        U_z_grid[~mask] = 0.0

        U_x_3D[i, :, :] = U_x_grid.T
        U_y_3D[i, :, :] = U_y_grid.T
        U_z_3D[i, :, :] = U_z_grid.T

    ux_interp = RegularGridInterpolator(
        (x_interp, y_interp, z_interp), U_x_3D,
        method='linear', bounds_error=False, fill_value=0)
    uy_interp = RegularGridInterpolator(
        (x_interp, y_interp, z_interp), U_y_3D,
        method='linear', bounds_error=False, fill_value=0)
    uz_interp = RegularGridInterpolator(
        (x_interp, y_interp, z_interp), U_z_3D,
        method='linear', bounds_error=False, fill_value=0)

    return ux_interp, uy_interp, uz_interp
