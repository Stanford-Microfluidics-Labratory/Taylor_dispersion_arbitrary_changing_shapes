"""Semi-Lagrangian PDE solver for the 1-D effective advection-diffusion equation.

Uses a backward-characteristic (semi-Lagrangian) advection step with RK4
trajectory integration, combined with Crank-Nicolson diffusion via a
sparse direct solve.
"""

import numpy as np
from numba import njit
from scipy.sparse import diags, eye
from scipy.sparse.linalg import factorized


def _build_cubic_lagrange_weights(x_vals, query_points):
    """Return (base_idx, w_m1, w_0, w_1, w_2) for 4-point cubic Lagrange
    interpolation of values sampled on the uniform grid ``x_vals`` at
    ``query_points``. The interpolated value at query j is
    ``w_m1[j]*f[base_idx[j]-1] + w_0[j]*f[base_idx[j]] + w_1[j]*f[base_idx[j]+1]
      + w_2[j]*f[base_idx[j]+2]``.
    Query points outside the interpolatable interior are clamped to the nearest
    valid stencil base index.
    """
    nx = x_vals.shape[0]
    dx = x_vals[1] - x_vals[0]
    i = np.floor((query_points - x_vals[0]) / dx).astype(np.int64)
    np.clip(i, 1, nx - 3, out=i)
    s = (query_points - x_vals[i]) / dx
    w_m1 = -s * (s - 1.0) * (s - 2.0) / 6.0
    w_0 = (s + 1.0) * (s - 1.0) * (s - 2.0) / 2.0
    w_1 = -(s + 1.0) * s * (s - 2.0) / 2.0
    w_2 = (s + 1.0) * s * (s - 1.0) / 6.0
    return i, w_m1, w_0, w_1, w_2


@njit(cache=True, fastmath=True)
def _laplacian_and_rhs(c, rhs, base_idx, w_m1, w_0, w_1, w_2, dt2D,
                      num_below, inv_dx2):
    """Fused kernel: compute 4th-order Laplacian of ``c`` and assemble
    ``rhs[num_below:] = S @ c + dt2D * (S @ L_dot_c)`` in a single set of
    passes over the grid.

    ``S`` is the cubic-Lagrange interp encoded by (base_idx, w_*); ``dt2D[j]``
    holds ``(dt/2) * D_depart[j]``; ``inv_dx2`` is ``1/dx**2``.
    ``rhs[:num_below]`` is zeroed.
    """
    n = c.shape[0]
    a0 = -inv_dx2 / 12.0
    a1 = 4.0 * inv_dx2 / 3.0
    a2 = -5.0 * inv_dx2 / 2.0

    L_dot_c = np.empty(n, dtype=c.dtype)
    for i in range(n):
        s = a2 * c[i]
        if i - 2 >= 0:
            s += a0 * c[i - 2]
        if i - 1 >= 0:
            s += a1 * c[i - 1]
        if i + 1 < n:
            s += a1 * c[i + 1]
        if i + 2 < n:
            s += a0 * c[i + 2]
        L_dot_c[i] = s

    for j in range(num_below):
        rhs[j] = 0.0

    nq = base_idx.shape[0]
    for j in range(nq):
        k = base_idx[j]
        wm = w_m1[j]; w0 = w_0[j]; w1v = w_1[j]; w2v = w_2[j]
        c_m1 = c[k - 1]; c_0 = c[k]; c_1 = c[k + 1]; c_2 = c[k + 2]
        l_m1 = L_dot_c[k - 1]; l_0 = L_dot_c[k]; l_1 = L_dot_c[k + 1]; l_2 = L_dot_c[k + 2]
        Sc = wm * c_m1 + w0 * c_0 + w1v * c_1 + w2v * c_2
        Sg = wm * l_m1 + w0 * l_0 + w1v * l_1 + w2v * l_2
        rhs[num_below + j] = Sc + dt2D[j] * Sg


@njit(cache=True)
def _moments(x_vals, c, area_vals):
    """Mass, area-weighted mean and variance of the concentration field."""
    n = c.shape[0]
    total = 0.0
    mom1 = 0.0
    for i in range(n):
        nc = c[i] * area_vals[i]
        total += nc
        mom1 += x_vals[i] * nc
    mean_x = mom1 / total
    var = 0.0
    for i in range(n):
        nc = c[i] * area_vals[i]
        d = x_vals[i] - mean_x
        var += d * d * nc
    var /= total
    return total, mean_x, var


def solve_concentration_pde(U_eff_interp, D_eff_interp, Area_interp, dt, nt,
                            x_min=-200, x_max=2000, nx=50000,
                            sigx2_0=10.0, num_sub=5000,
                            store_timesteps=None, period=None):
    """Solve the 1-D effective advection-diffusion equation.

    Solves:
        dc/dt + d(U_eff * c)/dx = d/dx(D_eff * dc/dx)

    using semi-Lagrangian advection (backward characteristics with RK4)
    and Crank-Nicolson diffusion (4th-order Laplacian stencil).

    Parameters
    ----------
    U_eff_interp : callable
        Effective velocity U_eff(x).
    D_eff_interp : callable
        Effective diffusivity D_eff(x).
    Area_interp : callable
        Cross-sectional area A(x), used for weighting moments.
    dt : float
        Time step.
    nt : int
        Number of time steps.
    x_min, x_max : float
        Spatial domain bounds.
    nx : int
        Number of spatial grid points.
    sigx2_0 : float
        Initial Gaussian variance.
    num_sub : int
        Number of sub-steps for backward characteristic RK4 integration.
    store_timesteps : list of int, optional
        Time step indices at which to store the concentration field.
        If None, nothing is stored.
    period : float, optional
        Channel period. If given, the solver runs in periodic mode:
        ``U_eff``/``D_eff``/area lookups are wrapped with ``% period`` (so a
        single-period interpolant stays valid no matter how far the tracer
        advects, instead of extrapolating a runaway polynomial), and a
        co-moving frame keeps the tracer on-grid -- whenever its leading edge
        nears the right boundary the field is rolled back by a whole number of
        periods. Because the shift is an integer number of periods and the
        coefficients are periodic, the CN operator, departure stencil and
        diffusivity all stay valid (no re-factorization). Moments are reported
        in the absolute (lab-frame) coordinate. If None (default) the grid is
        fixed and no wrapping/shifting occurs.

    Returns
    -------
    dict with keys ``x_locations``, ``predicted_mass``, ``predicted_mean``,
    ``predicted_variance``, ``con_field``, ``con_diff``, ``con_x_locations``.
    ``con_x_locations`` holds the absolute (lab-frame) grid coordinates at each
    stored timestep (they differ from ``x_locations`` once shifting kicks in).
    """
    x_vals = np.linspace(x_min, x_max, nx)
    dx = x_vals[1] - x_vals[0]

    if store_timesteps is None:
        store_timesteps = []

    # In periodic mode, wrap every interpolant lookup into one period so a
    # single-period U_eff/D_eff/area interpolant stays valid arbitrarily far
    # downstream. In non-periodic mode this is the identity, so behavior (and
    # the dumbbell case, which clamps instead) is unchanged.
    wrap = (lambda x: np.mod(x, period)) if period is not None else (lambda x: x)

    c = np.exp(-0.5 * (x_vals / np.sqrt(sigx2_0)) ** 2)

    D_vals = D_eff_interp(wrap(x_vals))
    area_vals = Area_interp(wrap(x_vals))

    # Absolute (lab-frame) coordinate of each grid point. With co-moving
    # shifting, the field content rolls back by whole periods while x_stats
    # records the true downstream position, so moments stay in the lab frame.
    x_stats = x_vals.copy()

    sub_dt = dt / num_sub
    depart_points = x_vals.copy()

    for _ in range(num_sub):
        k1 = -U_eff_interp(wrap(depart_points))
        k2 = -U_eff_interp(wrap(depart_points + 0.5 * sub_dt * k1))
        k3 = -U_eff_interp(wrap(depart_points + 0.5 * sub_dt * k2))
        k4 = -U_eff_interp(wrap(depart_points + sub_dt * k3))
        depart_points += sub_dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    coeffs = np.array([-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12])
    offsets = np.arange(-2, 3)
    diagonals = [coeffs[k] * np.ones(nx - abs(offsets[k])) for k in range(5)]
    L = diags(diagonals, offsets) / dx ** 2
    L.tocsc()

    lhs = (eye(nx) - dt * diags([D_vals], offsets=[0]) * L / 2).tocsc()

    loc_neg = depart_points <= x_min
    num_below = np.sum(loc_neg)
    depart_points_valid = depart_points[~loc_neg]
    D_depart = D_eff_interp(wrap(depart_points_valid))

    rhs = np.zeros(nx)
    solver = factorized(lhs)

    base_idx, w_m1, w_0, w_1, w_2 = _build_cubic_lagrange_weights(
        x_vals, depart_points_valid)
    dt2D = 0.5 * dt * D_depart
    inv_dx2 = 1.0 / dx ** 2

    predicted_mass = np.zeros(nt)
    predicted_x_bar = np.zeros(nt)
    predicted_var = np.zeros(nt)

    stored_concentration = []
    stored_diff = []
    stored_x = []

    if period is not None:
        near_end = int(np.floor(0.93 * nx))
        nx_in_period = period / dx

    for n in range(nt):
        if n in store_timesteps:
            stored_concentration.append(c.copy())
            stored_diff.append(np.gradient(c, dx, edge_order=2))
            stored_x.append(x_stats.copy())

        _laplacian_and_rhs(c, rhs, base_idx, w_m1, w_0, w_1, w_2,
                           dt2D, num_below, inv_dx2)
        c = solver(rhs)

        total, mean_x, var_x = _moments(x_stats, c, area_vals)
        predicted_x_bar[n] = mean_x
        predicted_var[n] = var_x
        predicted_mass[n] = total

        # Co-moving shift: once the leading edge reaches the right of the grid,
        # slide the field back by a whole number of periods. The grid, CN
        # operator and departure stencil are untouched; only the field content
        # rolls and x_stats advances, so the tracer never runs off the domain.
        if period is not None and np.sum(c[near_end:]) > 1e-4 * total:
            cum = np.cumsum(c)
            j = int(np.searchsorted(cum, 5e-5 * total))
            num_periods = np.floor(j / nx_in_period) - 1
            if num_periods >= 1:
                shift_back = int(round(num_periods * nx_in_period))
                tail = c[j:].copy()
                c[j - shift_back:nx - shift_back] = tail
                c[:j - shift_back] = 0.0
                c[nx - shift_back:] = 0.0
                x_stats += shift_back * dx
                area_vals = Area_interp(wrap(x_stats))

    return {
        'x_locations': x_vals,
        'predicted_mass': predicted_mass,
        'predicted_mean': predicted_x_bar,
        'predicted_variance': predicted_var,
        'con_field': stored_concentration,
        'con_diff': stored_diff,
        'con_x_locations': stored_x,
    }
