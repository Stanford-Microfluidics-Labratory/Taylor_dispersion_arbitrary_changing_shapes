"""Finite difference Poisson solver for the stream function Psi in dumbbell channels.

Provides three interfaces:
- ``solve_psi_at_slice``: solve at a single x-location.
- ``compute_dumbbell_psi_field``: solve at multiple x-locations, return detailed results.
- ``compute_dumbbell_psi_interpolator``: solve and return a RegularGridInterpolator.
"""

import numpy as np
from numba import njit, prange
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator


@njit(cache=True, parallel=True)
def _build_dumbbell_mask(Y, Z, H, L, R):
    """Return boolean mask for interior of a dumbbell cross-section."""
    n0, n1 = Y.shape
    mask = np.empty((n0, n1), dtype=np.bool_)
    half_L = L / 2
    R2 = R * R
    half_H = H / 2
    for i in prange(n0):
        for j in range(n1):
            yv = Y[i, j]
            zv = Z[i, j]
            z2 = zv * zv
            left_circle = (yv + half_L) ** 2 + z2 <= R2
            right_circle = (yv - half_L) ** 2 + z2 <= R2
            rect_region = (abs(yv) <= half_L) and (abs(zv) <= half_H)
            handle = rect_region and (not left_circle) and (not right_circle)
            mask[i, j] = left_circle or right_circle or handle
    return mask


@njit(cache=True)
def _assemble_poisson(flat_mask, Y_flat, Z_flat, Ny, N, dy, dz,
                      flat_source, D, gamma_over_area, H, L, R, n_x):
    """Assemble the COO triplets and RHS for the Poisson system.

    Returns ``(rows, cols, data, b)`` ready to hand to
    ``scipy.sparse.coo_matrix``.
    """
    dy2 = dy * dy
    dz2 = dz * dz
    half_L = L / 2
    R2 = R * R
    half_H = H / 2

    b = np.zeros(N)

    n_comp = 0
    for k in range(N):
        if flat_mask[k]:
            n_comp += 1
    n_identity = N - n_comp

    max_entries = n_identity + 5 * n_comp
    rows = np.empty(max_entries, dtype=np.int64)
    cols = np.empty(max_entries, dtype=np.int64)
    data = np.empty(max_entries, dtype=np.float64)
    e = 0

    shifts = (-1, 1, -Ny, Ny)
    dist_sqs = (dy2, dy2, dz2, dz2)
    dist_lins = (-dy, dy, -dz, dz)

    for k in range(N):
        if not flat_mask[k]:
            rows[e] = k
            cols[e] = k
            data[e] = 1.0
            e += 1
            continue

        b[k] = flat_source[k] / D - gamma_over_area

        diag = 0.0
        for d in range(4):
            shift = shifts[d]
            dist_sq = dist_sqs[d]
            dist_lin = dist_lins[d]
            nk = k + shift

            inside = (0 <= nk < N) and flat_mask[nk]
            if inside:
                rows[e] = k
                cols[e] = nk
                data[e] = 1.0 / dist_sq
                e += 1
                diag -= 1.0 / dist_sq
            else:
                if shift == -Ny or shift == Ny:
                    gy = Y_flat[k]
                    gz = Z_flat[k]
                    z2 = gz * gz
                    left_circle = (gy + half_L) ** 2 + z2 <= R2
                    right_circle = (gy - half_L) ** 2 + z2 <= R2
                    rect_region = (abs(gy) <= half_L) and (abs(gz) <= half_H)
                    is_handle_boundary = (rect_region and (not left_circle)
                                          and (not right_circle))
                else:
                    is_handle_boundary = False

                if is_handle_boundary:
                    sign = 1.0 if shift > 0 else -1.0
                    flux = -n_x * sign
                else:
                    flux = 0.0
                b[k] -= flux / dist_lin

        rows[e] = k
        cols[e] = k
        data[e] = diag
        e += 1

    return rows[:e], cols[:e], data[:e], b


def solve_psi_at_slice(x_val, ux_interp, uy_interp, uz_interp, D,
                       z_func=None, L=4.0, R=1.0, Ny=1000, Nz=1000,
                       scale_factor=10):
    """Solve the Poisson equation for the stream function at a single x-location.

    Parameters
    ----------
    x_val : float
        Streamwise position.
    ux_interp, uy_interp, uz_interp : callable
        Velocity field interpolators (e.g. RegularGridInterpolator).
    D : float
        Diffusion coefficient.
    z_func : callable, optional
        Half-height function z(x). Defaults to the standard dumbbell taper.
    L : float
        Distance between circle centers.
    R : float
        Radius of the circles.
    Ny, Nz : int
        Grid resolution.
    scale_factor : float
        Scaling between simulation coords and interpolator coords. The velocity
        profile is divided by scale_factor because the OpenFoam channel is
        scale_factor times shorter than the actual channel.

    Returns
    -------
    dict
        Dictionary with keys: ``psi`` (masked array), ``u_prime``, ``mask``,
        ``up_Psi`` (scalar), ``H``, ``area``, ``y``, ``z``,
        ``avg_ux``, ``avg_uy``, ``avg_uz``,
        ``avg_dpsi_dy``, ``avg_dpsi_dz``,
        ``avg_up_dpsi_dy``, ``avg_up_dpsi_dz``, ``Gamma``.
    """
    if z_func is None:
        z_func = lambda x: np.where(x < 1200, np.where(x < 0, 1, -0.00075 * x + 1), 0.1)

    H = 2 * z_func(x_val)

    z_min, z_max = -R - 0.01, R + 0.01
    y_min, y_max = -L / 2 - R - 0.01, L / 2 + R + 0.01

    z = np.linspace(z_min, z_max, Nz)
    y = np.linspace(y_min, y_max, Ny)
    Y, Z = np.meshgrid(y, z, indexing='xy')
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    mask = _build_dumbbell_mask(Y, Z, H, L, R)
    num_points = np.sum(mask)
    area_val = num_points * dz * dy

    coords = (np.full_like(Y, np.minimum(x_val / scale_factor, 120)), Y, Z)
    ux_vals = ux_interp(coords)
    avg_ux = np.sum(ux_vals) / num_points
    u_prime = ux_vals - avg_ux
    u_prime[~mask] = 0

    N = Ny * Nz
    flat_indices = np.arange(N)
    flat_mask = mask.ravel()

    Y_flat = np.ascontiguousarray(Y.ravel())
    Z_flat = np.ascontiguousarray(Z.ravel())
    is_computational = flat_mask

    flat_source = np.ascontiguousarray(u_prime.ravel())

    n_x = 0.9 / 1200
    gamma_area = lambda x: 4 * (L / 2 - np.sqrt(R ** 2 - (z_func(x)) ** 2)) * n_x

    gamma_over_area = gamma_area(x_val) / area_val
    Gamma_val = gamma_over_area

    rows, cols, data, b = _assemble_poisson(
        np.ascontiguousarray(flat_mask), Y_flat, Z_flat, Ny, N, dy, dz,
        flat_source, D, gamma_over_area, H, L, R, n_x)

    A = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    center_dist = Y_flat[is_computational] ** 2 + Z_flat[is_computational] ** 2
    ref_idx_local = np.argmin(center_dist)
    ref_idx_global = flat_indices[is_computational][ref_idx_local]
    csr_rows_start = A.indptr[ref_idx_global]
    csr_rows_end = A.indptr[ref_idx_global + 1]
    A.data[csr_rows_start:csr_rows_end] = 0.0
    A[ref_idx_global, ref_idx_global] = 1.0
    b[ref_idx_global] = 0.0
    psi = spsolve(A, b)
    psi_grid = psi.reshape((Nz, Ny))

    psi_plot = np.ma.masked_where(~mask, psi_grid)
    psi_plot -= np.sum(psi_plot) / num_points

    dpsi_dz, dpsi_dy = np.gradient(psi_plot, dz, dy)
    dpsi_dz = np.ma.masked_where(~mask, dpsi_dz)
    dpsi_dy = np.ma.masked_where(~mask, dpsi_dy)

    uy_vals = uy_interp(coords) / scale_factor
    avg_uy = np.sum(uy_vals) / num_points
    uy_prime = uy_vals - avg_uy
    uy_prime[~mask] = 0

    uz_vals = uz_interp(coords) / scale_factor
    avg_uz = np.sum(uz_vals) / num_points
    uz_prime = uz_vals - avg_uz
    uz_prime[~mask] = 0

    return {
        'psi': psi_plot,
        'u_prime': u_prime,
        'mask': mask.astype(np.bool_),
        'up_Psi': np.sum(u_prime * psi_plot) / num_points,
        'H': H,
        'area': area_val,
        'num_points': num_points,
        'y': y,
        'z': z,
        'avg_ux': avg_ux,
        'avg_uy': avg_uy,
        'avg_uz': avg_uz,
        'avg_dpsi_dy': np.sum(dpsi_dy) / num_points,
        'avg_dpsi_dz': np.sum(dpsi_dz) / num_points,
        'avg_up_dpsi_dy': np.sum(dpsi_dy * uy_prime) / num_points,
        'avg_up_dpsi_dz': np.sum(dpsi_dz * uz_prime) / num_points,
        'Gamma': Gamma_val,
    }


def compute_dumbbell_psi_field(x_vals, ux_interp, uy_interp, uz_interp, D, **kwargs):
    """Solve the stream function at multiple x-locations and return detailed results.

    Parameters
    ----------
    x_vals : array_like
        Streamwise positions.
    ux_interp, uy_interp, uz_interp : callable
        Velocity field interpolators.
    D : float
        Diffusion coefficient.
    **kwargs
        Additional keyword arguments passed to ``solve_psi_at_slice``.

    Returns
    -------
    dict
        Aggregated results with arrays for each quantity, plus lists of per-slice
        ``psi``, ``u_prime``, and ``mask`` arrays.
    """
    x_vals = np.asarray(x_vals)
    n = len(x_vals)

    up_Psi_list = np.zeros(n)
    H_list = np.zeros(n)
    Gamma_list = np.zeros(n)
    avg_ux_list = np.zeros(n)
    avg_uy_list = np.zeros(n)
    avg_uz_list = np.zeros(n)
    avg_dpsi_dy_arr = np.zeros(n)
    avg_dpsi_dz_arr = np.zeros(n)
    avg_up_dpsi_dy_arr = np.zeros(n)
    avg_up_dpsi_dz_arr = np.zeros(n)
    area_list = np.zeros(n)
    mask_list = []
    Psi_list = []
    up_list = []

    for i, x_val in enumerate(x_vals):
        res = solve_psi_at_slice(x_val, ux_interp, uy_interp, uz_interp, D, **kwargs)
        up_Psi_list[i] = res['up_Psi']
        H_list[i] = res['H']
        Gamma_list[i] = res['Gamma']
        avg_ux_list[i] = res['avg_ux']
        avg_uy_list[i] = res['avg_uy']
        avg_uz_list[i] = res['avg_uz']
        avg_dpsi_dy_arr[i] = res['avg_dpsi_dy']
        avg_dpsi_dz_arr[i] = res['avg_dpsi_dz']
        avg_up_dpsi_dy_arr[i] = res['avg_up_dpsi_dy']
        avg_up_dpsi_dz_arr[i] = res['avg_up_dpsi_dz']
        area_list[i] = res['num_points']

        psi_clean = res['psi'].filled(0.0).astype(np.float32)
        u_prime = res['u_prime']
        if np.ma.is_masked(u_prime):
            u_clean = u_prime.filled(0.0).astype(np.float32)
        else:
            u_clean = u_prime.astype(np.float32)
        mask_list.append(res['mask'])
        Psi_list.append(psi_clean)
        up_list.append(u_clean)

    up_arr = np.array(up_list)
    Psi_arr = np.array(Psi_list)
    mask_arr = np.array(mask_list)

    dx = x_vals[1] - x_vals[0]
    dpsi_dx = np.gradient(Psi_arr, dx, axis=0)
    avg_dpsi_dx = np.sum(dpsi_dx, axis=(1, 2)) / area_list
    avg_up_dpsi_dx = np.sum(dpsi_dx * up_arr, axis=(1, 2)) / area_list

    d2psi_dx2 = np.gradient(dpsi_dx, dx, axis=0)
    avg_d2psi_dx2 = np.sum(d2psi_dx2, axis=(1, 2)) / area_list

    dy = res['y'][1] - res['y'][0]
    dz = res['z'][1] - res['z'][0]
    area_list_physical = area_list * dy * dz

    return {
        'H': H_list,
        'up_Psi': up_Psi_list,
        'Gamma': Gamma_list,
        'avg_ux': avg_ux_list,
        'avg_uy': avg_uy_list,
        'avg_uz': avg_uz_list,
        'avg_dpsi_dx': avg_dpsi_dx,
        'avg_dpsi_dy': avg_dpsi_dy_arr,
        'avg_dpsi_dz': avg_dpsi_dz_arr,
        'avg_up_dpsi_dx': avg_up_dpsi_dx,
        'avg_up_dpsi_dy': avg_up_dpsi_dy_arr,
        'avg_up_dpsi_dz': avg_up_dpsi_dz_arr,
        'avg_d2psi_dx2': avg_d2psi_dx2,
        'x_vals': x_vals,
        'area_vals': area_list_physical,
    }


def compute_dumbbell_psi_interpolator(x_vals, ux_interp, uy_interp, uz_interp, D, **kwargs):
    """Solve the stream function and return a RegularGridInterpolator.

    Parameters
    ----------
    x_vals : array_like
        Streamwise positions.
    ux_interp, uy_interp, uz_interp : callable
        Velocity field interpolators.
    D : float
        Diffusion coefficient.
    **kwargs
        Additional keyword arguments passed to ``solve_psi_at_slice``.

    Returns
    -------
    RegularGridInterpolator
        3-D interpolator for Psi(x, y, z).
    """
    x_vals = np.asarray(x_vals)
    Psi_list = []

    for x_val in x_vals:
        res = solve_psi_at_slice(x_val, ux_interp, uy_interp, uz_interp, D, **kwargs)
        Psi_list.append(res['psi'].filled(0.0).T)

    y = res['y']
    z = res['z']
    psi_interp = RegularGridInterpolator(
        (x_vals, y, z),
        np.array(Psi_list),
        method='linear',
        bounds_error=False,
        fill_value=0.0,
    )
    return psi_interp
