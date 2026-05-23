"""ODE solver for the moment equations of Taylor dispersion.

Solves the coupled system for the mean position and variance:
    d(x_bar)/dt = u_eff - D * Gamma
    d(sigma^2)/dt = 2*D + 2*u_eff*sigma^2*Gamma - 2*<u'Psi>

using a 4th-order Runge-Kutta scheme. The interpolants are sampled onto fine
uniform 1-D tables and the RK4 time-stepping loop runs as a single ``@njit``
kernel with linear-table interpolation.
"""

import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def _interp1d(x0, dx, table, x):
    """Linear interpolation on a uniform 1-D table, with edge extrapolation."""
    n = table.shape[0]
    fx = (x - x0) / dx
    i = int(np.floor(fx))
    if i < 0:
        i = 0
    elif i > n - 2:
        i = n - 2
    t = fx - i
    return table[i] * (1.0 - t) + table[i + 1] * t


@njit(cache=True)
def _rk4_moment(predicted_x_bar, predicted_var, dt_sub, D,
                tbl_x0, tbl_dx, g_table, u_table, psi_table):
    """RK4 time-stepping kernel for :func:`solve_moment_ode`.

    ``predicted_x_bar``/``predicted_var`` are length ``n_total`` with index 0
    pre-filled with the initial conditions; the rest are written in place.
    """
    n_total = predicted_x_bar.shape[0]
    for i in range(1, n_total):
        x_curr = predicted_x_bar[i - 1]
        var_curr = predicted_var[i - 1]

        # k1
        g = _interp1d(tbl_x0, tbl_dx, g_table, x_curr)
        u = _interp1d(tbl_x0, tbl_dx, u_table, x_curr)
        psi_u = _interp1d(tbl_x0, tbl_dx, psi_table, x_curr)
        k1 = u - D * g
        q1 = 2 * D + 2 * u * var_curr * g - 2 * psi_u

        # k2
        xx = x_curr + 0.5 * dt_sub * k1
        vv = var_curr + 0.5 * dt_sub * q1
        g = _interp1d(tbl_x0, tbl_dx, g_table, xx)
        u = _interp1d(tbl_x0, tbl_dx, u_table, xx)
        psi_u = _interp1d(tbl_x0, tbl_dx, psi_table, xx)
        k2 = u - D * g
        q2 = 2 * D + 2 * u * vv * g - 2 * psi_u

        # k3
        xx = x_curr + 0.5 * dt_sub * k2
        vv = var_curr + 0.5 * dt_sub * q2
        g = _interp1d(tbl_x0, tbl_dx, g_table, xx)
        u = _interp1d(tbl_x0, tbl_dx, u_table, xx)
        psi_u = _interp1d(tbl_x0, tbl_dx, psi_table, xx)
        k3 = u - D * g
        q3 = 2 * D + 2 * u * vv * g - 2 * psi_u

        # k4
        xx = x_curr + dt_sub * k3
        vv = var_curr + dt_sub * q3
        g = _interp1d(tbl_x0, tbl_dx, g_table, xx)
        u = _interp1d(tbl_x0, tbl_dx, u_table, xx)
        psi_u = _interp1d(tbl_x0, tbl_dx, psi_table, xx)
        k4 = u - D * g
        q4 = 2 * D + 2 * u * vv * g - 2 * psi_u

        predicted_x_bar[i] = x_curr + (dt_sub / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        predicted_var[i] = var_curr + (dt_sub / 6) * (q1 + 2 * q2 + 2 * q3 + q4)


@njit(cache=True)
def _rk4_moment_circular(predicted_x_bar, predicted_var, dt_sub, D,
                         tbl_x0, tbl_dx, a_table, ap_table):
    """RK4 time-stepping kernel for :func:`solve_moment_ode_circular`.

    Computes the analytical coefficients inline from tabulated ``a(x)`` and
    ``a'(x)``.
    """
    n_total = predicted_x_bar.shape[0]
    for i in range(1, n_total):
        x_curr = predicted_x_bar[i - 1]
        var_curr = predicted_var[i - 1]

        # k1
        a = _interp1d(tbl_x0, tbl_dx, a_table, x_curr)
        b = _interp1d(tbl_x0, tbl_dx, ap_table, x_curr)
        g = -2 * b / a
        u = 1 / a ** 2
        avg_psi_u = -a ** 2 * u ** 2 / (48 * D) - a * b * u / 12
        k1 = u - D * g
        q1 = 2 * D + 2 * u * var_curr * g - 2 * avg_psi_u

        # k2
        xx = x_curr + 0.5 * dt_sub * k1
        vv = var_curr + 0.5 * dt_sub * q1
        a = _interp1d(tbl_x0, tbl_dx, a_table, xx)
        b = _interp1d(tbl_x0, tbl_dx, ap_table, xx)
        g = -2 * b / a
        u = 1 / a ** 2
        avg_psi_u = -a ** 2 * u ** 2 / (48 * D) - a * b * u / 12
        k2 = u - D * g
        q2 = 2 * D + 2 * u * vv * g - 2 * avg_psi_u

        # k3
        xx = x_curr + 0.5 * dt_sub * k2
        vv = var_curr + 0.5 * dt_sub * q2
        a = _interp1d(tbl_x0, tbl_dx, a_table, xx)
        b = _interp1d(tbl_x0, tbl_dx, ap_table, xx)
        g = -2 * b / a
        u = 1 / a ** 2
        avg_psi_u = -a ** 2 * u ** 2 / (48 * D) - a * b * u / 12
        k3 = u - D * g
        q3 = 2 * D + 2 * u * vv * g - 2 * avg_psi_u

        # k4
        xx = x_curr + dt_sub * k3
        vv = var_curr + dt_sub * q3
        a = _interp1d(tbl_x0, tbl_dx, a_table, xx)
        b = _interp1d(tbl_x0, tbl_dx, ap_table, xx)
        g = -2 * b / a
        u = 1 / a ** 2
        avg_psi_u = -a ** 2 * u ** 2 / (48 * D) - a * b * u / 12
        k4 = u - D * g
        q4 = 2 * D + 2 * u * vv * g - 2 * avg_psi_u

        predicted_x_bar[i] = x_curr + (dt_sub / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        predicted_var[i] = var_curr + (dt_sub / 6) * (q1 + 2 * q2 + 2 * q3 + q4)


def _build_x_table(dt, nt, sub_step):
    """Uniform x-sampling grid covering the reachable mean-position range."""
    n_total = nt * sub_step
    dt_sub = dt / sub_step
    x_reach = n_total * dt_sub * 2.0 + 1000.0
    x_lo, x_hi = -x_reach, x_reach
    spacing = 1.0
    n_tbl = int((x_hi - x_lo) / spacing) + 1
    x_grid = np.linspace(x_lo, x_hi, n_tbl)
    return x_grid


def solve_moment_ode(dt, nt, D, Gamma_interp, u_interp, Psi_interp,
                     sigx2_0=10.0, sub_step=1):
    """Solve the moment ODEs for mean position and variance using RK4.

    Parameters
    ----------
    dt : float
        Base time step between output samples.
    nt : int
        Number of output time steps.
    D : float
        Diffusion coefficient.
    Gamma_interp : callable
        Interpolant for Gamma(x) = -A'(x)/A(x).
    u_interp : callable
        Interpolant for the effective mean velocity u(x).
    Psi_interp : callable
        Interpolant for <u'*Psi>(x).
    sigx2_0 : float
        Initial variance.
    sub_step : int
        Number of sub-steps per output step for higher accuracy.

    Returns
    -------
    predicted_x_bar : ndarray of shape (nt * sub_step,)
        Mean position at each sub-step.
    predicted_var : ndarray of shape (nt * sub_step,)
        Variance at each sub-step.
    """
    n_total = nt * sub_step
    dt_sub = dt / sub_step

    predicted_x_bar = np.zeros(n_total)
    predicted_var = np.zeros(n_total)
    predicted_var[0] = sigx2_0

    x_grid = _build_x_table(dt, nt, sub_step)
    tbl_x0 = x_grid[0]
    tbl_dx = x_grid[1] - x_grid[0]
    g_table = np.ascontiguousarray(Gamma_interp(x_grid), dtype=np.float64)
    u_table = np.ascontiguousarray(u_interp(x_grid), dtype=np.float64)
    psi_table = np.ascontiguousarray(Psi_interp(x_grid), dtype=np.float64)

    _rk4_moment(predicted_x_bar, predicted_var, dt_sub, D,
                tbl_x0, tbl_dx, g_table, u_table, psi_table)

    return predicted_x_bar, predicted_var


def solve_moment_ode_circular(dt, nt, D, func_x, sigx2_0=10.0, sub_step=1):
    """Solve the moment ODEs for a circular channel using analytical coefficients.

    For a circular channel with radius a(x), the effective coefficients are
    computed analytically: u = 1/a^2, Gamma = -2*a'/a,
    <u'Psi> = -a^2*u^2/(48*D) - a*a'*u/12.

    Parameters
    ----------
    dt : float
        Base time step.
    nt : int
        Number of output time steps.
    D : float
        Diffusion coefficient.
    func_x : callable
        Radius function a(x).  Must also support a derivative via
        finite differences (or pass a function with known derivative).
    sigx2_0 : float
        Initial variance.
    sub_step : int
        Number of sub-steps per output step.

    Returns
    -------
    predicted_x_bar : ndarray of shape (nt * sub_step,)
    predicted_var : ndarray of shape (nt * sub_step,)
    """
    n_total = nt * sub_step
    dt_sub = dt / sub_step

    predicted_x_bar = np.zeros(n_total)
    predicted_var = np.zeros(n_total)
    predicted_var[0] = sigx2_0

    x_grid = _build_x_table(dt, nt, sub_step)
    tbl_x0 = x_grid[0]
    tbl_dx = x_grid[1] - x_grid[0]
    h = 1e-6
    a_table = np.ascontiguousarray(func_x(x_grid), dtype=np.float64)
    ap_table = np.ascontiguousarray(
        (func_x(x_grid + h) - func_x(x_grid - h)) / (2 * h), dtype=np.float64)

    _rk4_moment_circular(predicted_x_bar, predicted_var, dt_sub, D,
                         tbl_x0, tbl_dx, a_table, ap_table)

    return predicted_x_bar, predicted_var
