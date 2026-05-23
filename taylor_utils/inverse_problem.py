"""Inverse problem solver: find channel shape f(x) that produces a target variance.

Given a desired variance profile sigma^2(x), solve the implicit ODE for df/dx
using Newton iteration at each step, then integrate with RK4.
"""

import numpy as np
import numdifftools as nd
from numba import njit
from scipy.interpolate import CubicSpline

from .rect_psi import Psi_uPrime_avg

_PSI_NY = 2000          # y-grid points
_PSI_M_P0 = 500         # series terms for dp0/dx
_PSI_M = 20             # series terms for U/Y/C_corr
_PSI_NMAX = 20          # n runs 1..19


@njit(cache=True)
def _psi_uprime_scalar(x, fx, df_dx, D, up_mean):
    """Compute ``<u'*Psi>`` for a single streamwise location."""
    pi = np.pi

    s = 0.0
    for m in range(_PSI_M_P0):
        km = (m + 0.5) * pi
        s += 6.0 * km ** (-5) * np.tanh(km * fx)
    p0 = 3.0 / (4.0 * (s - fx))

    km_arr = np.empty(_PSI_M)
    km2_arr = np.empty(_PSI_M)
    cosh_kmfx = np.empty(_PSI_M)
    tanh_kmfx = np.empty(_PSI_M)
    for m in range(_PSI_M):
        km = (m + 0.5) * pi
        km_arr[m] = km
        km2_arr[m] = km * km
        cosh_kmfx[m] = np.cosh(km * fx)
        tanh_kmfx[m] = np.tanh(km * fx)

    nmax = _PSI_NMAX
    npi_arr = np.empty(nmax)
    npi2_arr = np.empty(nmax)
    sign_arr = np.empty(nmax)
    sinh_npifx = np.empty(nmax)
    for n in range(1, nmax):
        npi_arr[n] = n * pi
        npi2_arr[n] = (n * pi) ** 2
        sign_arr[n] = 1.0 if (n % 2 == 0) else -1.0
        sinh_npifx[n] = np.sinh(n * pi * fx)

    inv_denom = np.empty((nmax, _PSI_M))
    inv_denom2 = np.empty((nmax, _PSI_M))
    denom_over_npi2 = np.empty((nmax, _PSI_M))
    for n in range(1, nmax):
        for m in range(_PSI_M):
            denom = km2_arr[m] - npi2_arr[n]
            inv_denom[n, m] = 1.0 / denom
            inv_denom2[n, m] = 1.0 / (denom * denom)
            denom_over_npi2[n, m] = denom / npi2_arr[n]

    ny = _PSI_NY
    y0 = -fx
    h = (2.0 * fx) / (ny - 1)

    u0 = np.empty(ny)
    y2u0 = np.empty(ny)
    y0u0 = np.empty(ny)
    yu = np.empty(ny)

    ratio = np.empty(_PSI_M)
    g_arr = np.empty(_PSI_M)
    A_arr = np.empty(_PSI_M)
    B_arr = np.empty(_PSI_M)
    Cc_arr = np.empty(_PSI_M)
    for j in range(ny):
        y = y0 + j * h
        yy = y * y

        U0v = -1.0
        Y0v = -yy / 2.0
        for m in range(_PSI_M):
            km = km_arr[m]
            km2 = km2_arr[m]
            r = np.cosh(km * y) / cosh_kmfx[m]
            ratio[m] = r
            base = 8.0 * fx * p0 / (km2 * km2)
            U0v += base * (-1.0 + r)
            Y0v += base * (-yy / 2.0 + r / km2)
            common = 8.0 * fx * p0 / km2
            A_arr[m] = common
            B_arr[m] = common * r
            g_arr[m] = common * (-1.0 + r)
            Cc_arr[m] = common * km * tanh_kmfx[m]

        yu_sum = 0.0
        for n in range(1, nmax):
            npi = npi_arr[n]
            sign2 = 2.0 * sign_arr[n]
            cfac = np.cosh(npi * y) / (npi * sinh_npifx[n])
            Unv = 0.0
            Ynv = 0.0
            for m in range(_PSI_M):
                Unv += g_arr[m] * inv_denom[n, m]
                Ynv += (A_arr[m] * denom_over_npi2[n, m]
                        + B_arr[m]
                        - Cc_arr[m] * cfac) * inv_denom2[n, m]
            yu_sum += sign2 * sign2 * (Unv * Ynv)

        u0[j] = U0v
        y2u0[j] = yy * U0v
        y0u0[j] = Y0v * U0v
        yu[j] = yu_sum

    U0 = _simpson_uniform(u0, h)
    ytimesU0 = _simpson_uniform(y2u0, h)
    Y0timesU0 = _simpson_uniform(y0u0, h)
    YU_sum = _simpson_uniform(yu, h)

    Gamma_val = -df_dx / fx

    csum = 0.0
    for m in range(_PSI_M):
        km = km_arr[m]
        csum += (up_mean * 4.0 / (km ** 4 * D)) * p0 * (
            -fx ** 3 / 3.0 + 2.0 * tanh_kmfx[m] / (km ** 3))
    C_corr = (Gamma_val * fx * fx) / 6.0 - csum + up_mean * fx * fx / 6.0 / D

    return (-1.0 * ytimesU0 * Gamma_val + 2.0 * up_mean * Y0timesU0 / D
            + up_mean * YU_sum / D + 2.0 * U0 * C_corr) * up_mean / 4.0 / fx


@njit(cache=True, inline="always")
def _simpson_uniform(f, h):
    """Composite Simpson's rule on a uniform grid.

    For an even number of samples uses basic composite Simpson over the first
    ``N-3`` points plus the Cartwright last-interval correction.
    """
    n = f.shape[0]
    if n % 2 == 1:
        result = 0.0
        for i in range(0, n - 2, 2):
            result += f[i] + 4.0 * f[i + 1] + f[i + 2]
        return result * h / 3.0

    result = 0.0
    for i in range(0, n - 3, 2):
        result += f[i] + 4.0 * f[i + 1] + f[i + 2]
    result *= h / 3.0

    alpha = 5.0 * h / 12.0
    beta = 2.0 * h / 3.0
    eta = h / 12.0
    result += alpha * f[n - 1] + beta * f[n - 2] - eta * f[n - 3]
    return result


@njit(cache=True)
def _dfdx_newton(df_dx, fx, x, up_mean, D, sigma2, dsig2):
    """Newton iteration for ``df/dx``.

    All quantities are scalars. ``sigma2`` and ``dsig2`` are passed in
    pre-computed.
    """
    u_mean = up_mean
    denominator = 2.0 * u_mean * sigma2 + D * dsig2
    e = 1e-6
    while True:
        psi_u = _psi_uprime_scalar(x, fx, df_dx, D, up_mean)
        residual = df_dx + fx * (
            (u_mean * dsig2 + 2.0 * psi_u - 2.0 * D) / denominator)

        psi_plus = _psi_uprime_scalar(x, fx, df_dx + e, D, up_mean)
        psi_minus = _psi_uprime_scalar(x, fx, df_dx - e, D, up_mean)
        residual_dx = 1.0 + fx * 2.0 * (psi_plus - psi_minus) / 2.0 / e \
            / denominator

        temp_df_dx = df_dx - residual / residual_dx
        if abs(temp_df_dx - df_dx) < 1e-8:
            return temp_df_dx
        df_dx = temp_df_dx


def sigma2_const(x):
    """Constant target variance."""
    return 300 + 0 * x


def sigma2_sin_drift(x):
    """Sinusoidal target variance with linear drift."""
    return 300 + 50 * np.sin(2 * np.pi * x / 600) + 0.3 * x


def sigma2_sin_nodrift(x):
    """Sinusoidal target variance without drift."""
    return 300 + 50 * np.sin(2 * np.pi * x / 600)


def dfdx_rhs(x, df_dx, fx_temp, up_init, f0, sigma2_func, D, dx):
    """Compute df/dx via implicit Newton iteration.

    Parameters
    ----------
    x : float
        Current streamwise position.
    df_dx : float
        Current guess for df/dx.
    fx_temp : float or array
        Current half-width value.
    up_init : float
        Initial mean velocity.
    f0 : float
        Half-width at x=0 (for mass conservation).
    sigma2_func : callable
        Target variance function.
    D : float
        Diffusion coefficient.
    dx : float
        Step size.

    Returns
    -------
    float
        Converged df/dx value.
    """
    fx = float(np.asarray(fx_temp).ravel()[0]) if np.ndim(fx_temp) else float(fx_temp)

    dsigma2_dx = nd.Derivative(sigma2_func, n=1, step=1e-6)
    up_mean = up_init * f0 / fx
    x_f64 = float(x)
    sigma2 = float(sigma2_func(x_f64))
    dsig2 = float(dsigma2_dx(x_f64))

    return float(_dfdx_newton(float(df_dx), fx, x_f64, float(up_mean),
                              float(D), sigma2, dsig2))


def solve_inverse_problem(sigma2_func, D=0.1, up_init=1.0, f0=1.0,
                          x_forward=(0, 900, 450), x_backward=(-150, 0, 75)):
    """Solve the inverse problem to find the channel shape for a target variance.

    Integrates forward from x=0 and backward from x=0 using RK4, then
    combines results into a single CubicSpline.

    Parameters
    ----------
    sigma2_func : callable
        Target variance function sigma^2(x).
    D : float
        Diffusion coefficient.
    up_init : float
        Initial mean velocity at x=0.
    f0 : float
        Half-width at x=0.
    x_forward : tuple (start, end, n_points)
        Forward integration domain.
    x_backward : tuple (start, end, n_points)
        Backward integration domain.

    Returns
    -------
    CubicSpline
        Interpolant for the channel half-width f(x).
    """
    x_fwd = np.linspace(*x_forward, dtype=np.longdouble)
    fx_fwd = np.zeros_like(x_fwd, dtype=np.longdouble)
    fx_fwd[0] = f0

    k4 = 0
    dx = x_fwd[1] - x_fwd[0]
    for i in range(len(x_fwd) - 1):
        xc = x_fwd[i]
        fc = fx_fwd[i]
        k1 = dfdx_rhs(xc, k4, fc, up_init, f0, sigma2_func, D, dx)
        k2 = dfdx_rhs(xc + dx / 2, k1, fc + dx / 2 * k1, up_init, f0, sigma2_func, D, dx)
        k3 = dfdx_rhs(xc + dx / 2, k2, fc + dx / 2 * k2, up_init, f0, sigma2_func, D, dx)
        k4 = dfdx_rhs(xc + dx, k3, fc + dx * k3, up_init, f0, sigma2_func, D, dx)
        fx_fwd[i + 1] = fc + dx * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    x_bwd = np.linspace(*x_backward, dtype=np.longdouble)
    fx_bwd = np.zeros_like(x_bwd, dtype=np.longdouble)
    fx_bwd[-1] = f0

    k4 = 0
    dx_bwd = -x_bwd[1] + x_bwd[0]
    for i in range(1, len(x_bwd)):
        xc = x_bwd[-i]
        fc = fx_bwd[-i]
        k1 = dfdx_rhs(xc, k4, fc, up_init, f0, sigma2_func, D, dx_bwd)
        k2 = dfdx_rhs(xc + dx_bwd / 2, k1, fc + dx_bwd / 2 * k1, up_init, f0, sigma2_func, D, dx_bwd)
        k3 = dfdx_rhs(xc + dx_bwd / 2, k2, fc + dx_bwd / 2 * k2, up_init, f0, sigma2_func, D, dx_bwd)
        k4 = dfdx_rhs(xc + dx_bwd, k3, fc + dx_bwd * k3, up_init, f0, sigma2_func, D, dx_bwd)
        fx_bwd[-i - 1] = fc + dx_bwd * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    x_full = np.concatenate((x_bwd[:-1], x_fwd))
    fx_full = np.concatenate((fx_bwd[:-1], fx_fwd))
    return CubicSpline(x_full, fx_full)
