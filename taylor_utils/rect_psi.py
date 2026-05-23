"""Analytical computation of <u'*Psi> for rectangular channels.

Provides a vectorized version and a scalar loop-based version.
"""

import numpy as np
import numdifftools as nd
from numba import njit, prange
from scipy.integrate import fixed_quad, simpson


def area(x, fx_func):
    return fx_func(x) * 4


def Gamma(x_array, fx_func):
    x_array = np.asarray(x_array)
    derive_func = nd.Derivative(area, n=1, step=1e-6)
    deriv_vals = derive_func(x_array, fx_func)
    area_vals = area(x_array, fx_func)
    return -deriv_vals / area_vals


def k_m(m):
    return (m + 0.5) * np.pi


def upsilon_func(fx, phi, M=500):
    m = np.arange(M)
    km = k_m(m)
    am = np.sqrt(km ** 2 + phi ** 2)
    return np.tanh(phi) / phi + np.sum(2 * phi ** 2 / km ** 2 / am ** 3 / fx * np.tanh(am * fx))


def dp0_dx(fx, M=500):
    m = np.arange(M)
    km = k_m(m)
    return 3 / (4 * (np.sum(6 * km ** (-5) * np.tanh(km * fx), axis=0) - fx))


def _dp0_dx_vec(fx, M=500):
    m = np.arange(M)
    km = k_m(m)[:, None]
    return (3 / (4 * (np.nansum(6 * km ** (-5) * np.tanh(km * fx), axis=0) - fx))).reshape(-1)


def _C_corr_vec(Gamma_val, fx, p0, up_mean, D, M=20):
    m = np.arange(M)
    km = k_m(m)[:, None]
    return ((Gamma_val[None, :] * fx[None, :] ** 2) / 6
            - np.nansum((up_mean[None, :] * 4 / (km ** 4 * D)) * p0[None, :] * (
                -fx[None, :] ** 3 / 3 + 2 * np.tanh(km * fx[None, :]) / (km ** 3)), axis=0)
            + (up_mean[None, :]) * fx[None, :] ** 2 / 6 / D).reshape(-1)


def _U_0_vec(y, fx, p0, up_mean, M=20):
    m = np.arange(M)[:, None, None]
    km = k_m(m)
    u = up_mean
    return np.nansum(8 * fx * up_mean * p0 / km ** 4 / u * (
        -1 + np.cosh(km * y) / np.cosh(km * fx)), axis=0) - 1


def _Y_0_vec(y, fx, p0, up_mean, M=20):
    m = np.arange(M)[:, None, None]
    km = k_m(m)
    u = up_mean
    return np.nansum(8 * fx * up_mean * p0 / km ** 4 / u * (
        -y ** 2 / 2 + np.cosh(km * y) / (km ** 2 * np.cosh(km * fx))), axis=0) - y.reshape(1, -1) ** 2 / 2


def _U_n_vec(y, fx, p0, up_mean, n, M=20):
    m = np.arange(M)[:, None, None, None]
    km = k_m(m)
    u = up_mean
    return np.nansum(2 * (-1) ** n * (8 * fx * up_mean * p0 / km ** 2 / u * (
        -1 + np.cosh(km * y) / np.cosh(km * fx))) / (km ** 2 - (n * np.pi) ** 2), axis=0)


def _Y_n_vec(y, fx, p0, up_mean, n, M=20):
    m = np.arange(M)[:, None, None, None]
    km = k_m(m)
    u = up_mean
    return np.nansum(2 * (-1) ** n * (8 * fx * up_mean * p0 / km ** 2 / u * (
        (km ** 2 - (n * np.pi) ** 2) / ((n * np.pi) ** 2) + np.cosh(km * y) / np.cosh(km * fx)
        - km * np.tanh(km * fx) * np.cosh(n * np.pi * y) / n / np.pi / np.sinh(n * np.pi * fx)))
        / ((km ** 2 - (n * np.pi) ** 2) ** 2), axis=0)


@njit(cache=True, inline="always")
def _km_njit(m):
    return (m + 0.5) * np.pi


@njit(cache=True, inline="always")
def _C_corr_njit(Gamma_val, fxj, p0j, up_meanj, D, M=20):
    s = 0.0
    for m in range(M):
        km = _km_njit(m)
        s += (up_meanj * 4.0 / (km ** 4 * D)) * p0j * (
            -fxj ** 3 / 3.0 + 2.0 * np.tanh(km * fxj) / (km ** 3))
    return (Gamma_val * fxj ** 2) / 6.0 - s + up_meanj * fxj ** 2 / 6.0 / D


@njit(cache=True, parallel=True, fastmath=True)
def _psi_uprime_kernel(x_array, fx, df_dx, p0, up_mean, y, D, M_series, n_max):
    """Numeric core of :func:`Psi_uPrime_avg`.

    For each streamwise position, performs Simpson integration over the ``y``
    grid, summing the truncated cosh-series term by term. Runs the x-points
    with ``prange``.
    """
    nx = x_array.shape[0]
    ny = y.shape[0]
    out = np.empty(nx, dtype=np.float64)

    for j in prange(nx):
        fxj = fx[j]
        p0j = p0[j]
        uj = up_mean[j]
        Gamma_val = -df_dx[j] / fxj

        f_U0 = np.empty(ny, dtype=np.float64)
        f_yU0 = np.empty(ny, dtype=np.float64)
        f_Y0U0 = np.empty(ny, dtype=np.float64)
        f_YnUn = np.empty(ny, dtype=np.float64)

        for iy in range(ny):
            yy = y[iy]
            inside = np.abs(yy) <= fxj

            U0 = 0.0
            Y0 = 0.0
            if inside:
                for m in range(M_series):
                    km = _km_njit(m)
                    base = 8.0 * fxj * uj * p0j / km ** 4 / uj
                    ch_ratio = np.cosh(km * yy) / np.cosh(km * fxj)
                    U0 += base * (-1.0 + ch_ratio)
                    Y0 += base * (-yy ** 2 / 2.0 + np.cosh(km * yy)
                                  / (km ** 2 * np.cosh(km * fxj)))
                U0 -= 1.0
                Y0 -= yy ** 2 / 2.0

            f_U0[iy] = U0 if inside else 0.0
            f_yU0[iy] = (yy ** 2 * U0) if inside else 0.0
            f_Y0U0[iy] = (Y0 * U0) if inside else 0.0

        U0_int = _simpson_uniform(f_U0, y)
        ytimesU0 = _simpson_uniform(f_yU0, y)
        Y0timesU0 = _simpson_uniform(f_Y0U0, y)

        YU_sum = 0.0
        for n in range(1, n_max):
            npi = n * np.pi
            for iy in range(ny):
                yy = y[iy]
                inside = np.abs(yy) <= fxj
                if not inside:
                    f_YnUn[iy] = 0.0
                    continue
                Un = 0.0
                Yn = 0.0
                for m in range(M_series):
                    km = _km_njit(m)
                    denom = km ** 2 - npi ** 2
                    common = 8.0 * fxj * uj * p0j / km ** 2 / uj
                    ch_ratio = np.cosh(km * yy) / np.cosh(km * fxj)
                    Un += 2.0 * (-1.0) ** n * (common * (-1.0 + ch_ratio)) / denom
                    Yn += 2.0 * (-1.0) ** n * (common * (
                        denom / (npi ** 2) + ch_ratio
                        - km * np.tanh(km * fxj) * np.cosh(npi * yy)
                        / npi / np.sinh(npi * fxj))) / (denom ** 2)
                f_YnUn[iy] = Yn * Un
            YU_sum += _simpson_uniform(f_YnUn, y)

        C_corr = _C_corr_njit(Gamma_val, fxj, p0j, uj, D)
        out[j] = (-1.0 * ytimesU0 * Gamma_val + 2.0 * uj * Y0timesU0 / D
                  + uj * YU_sum / D + 2.0 * U0_int * C_corr) * uj / 4.0 / fxj
    return out


@njit(cache=True, inline="always")
def _basic_simpson_uniform(f, start, stop, dx):
    """Composite Simpson's rule for uniform spacing.

    Sums ``(f[s0] + 4 f[s1] + f[s2]) * dx / 3`` over ``s0`` in
    ``range(start, stop, 2)`` (``s1 = s0 + 1``, ``s2 = s0 + 2``).
    """
    result = 0.0
    for s0 in range(start, stop, 2):
        result += f[s0] + 4.0 * f[s0 + 1] + f[s0 + 2]
    return result * dx / 3.0


@njit(cache=True, inline="always")
def _simpson_uniform(f, x):
    """Composite Simpson's rule on a uniform grid.

    For an even number of sample points includes the Cartwright correction
    for the final interval.
    """
    n = f.shape[0]
    dx = x[1] - x[0]
    if n == 1:
        return 0.0
    if n == 2:
        return 0.5 * dx * (f[0] + f[1])

    if n % 2 == 1:
        return _basic_simpson_uniform(f, 0, n - 2, dx)
    else:
        result = _basic_simpson_uniform(f, 0, n - 3, dx)
        alpha = 5.0 * dx / 12.0
        beta = 2.0 * dx / 3.0
        eta = dx / 12.0
        result += alpha * f[n - 1] + beta * f[n - 2] - eta * f[n - 3]
        return result


def Psi_uPrime_avg(x_array, fx, df_dx, D, up_mean=None, up_init=None):
    """Compute <u'*Psi> for a rectangular channel (vectorized).

    Parameters
    ----------
    x_array : array_like
        Streamwise positions.
    fx : array_like
        Half-width values at each x position.
    df_dx : array_like
        Derivative of half-width at each x position.
    D : float
        Diffusion coefficient.
    up_mean : array_like, optional
        Mean velocity at each x position. If None, computed from ``up_init``.
    up_init : float, optional
        Initial mean velocity (used with conservation of mass to get up_mean).

    Returns
    -------
    numpy.ndarray
        Values of <u'*Psi> at each x position.
    """
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    x_array = np.atleast_1d(x_array)
    fx = np.atleast_1d(fx)
    df_dx = np.atleast_1d(df_dx)
    if up_mean is None:
        up_mean = up_init / fx
    up_mean = np.atleast_1d(up_mean)

    x_array = np.ascontiguousarray(x_array, dtype=np.float64)
    fx = np.ascontiguousarray(fx, dtype=np.float64)
    df_dx = np.ascontiguousarray(df_dx, dtype=np.float64)
    up_mean = np.ascontiguousarray(up_mean, dtype=np.float64)

    p0 = _dp0_dx_vec(fx.reshape(1, -1)).astype(np.float64)
    y = np.linspace(np.min(-fx), np.max(fx), 2000).astype(np.float64)

    x_array_vals = _psi_uprime_kernel(x_array, fx, df_dx, p0, up_mean, y,
                                      float(D), 20, 20)
    return x_array_vals


def _C_corr_scalar(D, fx, p0, up_mean, gamma_vals, M=100):
    m = np.arange(M)
    km = k_m(m)
    return ((gamma_vals * fx ** 2) / 6
            - np.sum((up_mean * 4 / (km ** 4 * D)) * p0 * (-fx ** 3 / 3 + 2 * np.tanh(km * fx) / (km ** 3)))
            + (up_mean) * fx ** 2 / 6 / D)


def _U_0_scalar(y, fx, p0, up_mean, M=100):
    m = np.arange(M)[:, None]
    y = np.atleast_1d(y)[None, :]
    km = k_m(m)
    u = up_mean
    return np.sum(8 * fx * up_mean * p0 / km ** 4 / u * (-1 + np.cosh(km * y) / np.cosh(km * fx)), axis=0) - 1


def _Y_0_scalar(y, fx, p0, up_mean, M=100):
    m = np.arange(M)[:, None]
    y = np.atleast_1d(y)[None, :]
    km = k_m(m)
    u = up_mean
    return np.sum(8 * fx * up_mean * p0 / km ** 4 / u * (
        -y ** 2 / 2 + np.cosh(km * y) / (km ** 2 * np.cosh(km * fx))), axis=0) - y ** 2 / 2


def _U_n_scalar(y, fx, p0, up_mean, n, M=100):
    m = np.arange(M)[:, None]
    y = np.atleast_1d(y)[None, :]
    km = k_m(m)
    u = up_mean
    return np.sum(2 * (-1) ** n * (8 * fx * up_mean * p0 / km ** 2 / u * (
        -1 + np.cosh(km * y) / np.cosh(km * fx))) / (km ** 2 - (n * np.pi) ** 2), axis=0)


def _Y_n_scalar(y, fx, p0, up_mean, n, M=100):
    m = np.arange(M)[:, None]
    y = np.atleast_1d(y)[None, :]
    km = k_m(m)
    u = up_mean
    return np.sum(2 * (-1) ** n * (8 * fx * up_mean * p0 / km ** 2 / u * (
        (km ** 2 - (n * np.pi) ** 2) / ((n * np.pi) ** 2) + np.cosh(km * y) / np.cosh(km * fx)
        - km * np.tanh(km * fx) * np.cosh(n * np.pi * y) / n / np.pi / np.sinh(n * np.pi * fx)))
        / ((km ** 2 - (n * np.pi) ** 2) ** 2), axis=0)


def Psi_uPrime_avg_scalar(x_array, up_init, fx_func, D):
    """Compute <u'*Psi> for a rectangular channel using a scalar loop.

    Parameters
    ----------
    x_array : array_like
        Streamwise positions.
    up_init : float
        Initial mean velocity.
    fx_func : callable
        Half-width function f(x).
    D : float
        Diffusion coefficient.

    Returns
    -------
    numpy.ndarray
        Values of <u'*Psi> at each x position.
    """
    x_array = np.atleast_1d(x_array)
    x_array_vals = np.zeros(len(x_array))
    for i in range(len(x_array)):
        x = x_array[i]
        fx = fx_func(x)
        quad = 200
        up_mean = up_init * fx_func(0) / fx
        u_mean = up_mean
        p0 = dp0_dx(fx)
        gamma_vals = Gamma(x, fx_func)

        U0, _ = fixed_quad(lambda y: _U_0_scalar(y, fx, p0, up_mean), -fx, fx, n=quad)
        ytimesU0, _ = fixed_quad(lambda y: y ** 2 * _U_0_scalar(y, fx, p0, up_mean), -fx, fx, n=quad)
        Y0timesU0, _ = fixed_quad(
            lambda y: _Y_0_scalar(y, fx, p0, up_mean) * _U_0_scalar(y, fx, p0, up_mean), -fx, fx, n=quad)

        YU_sum = 0.0
        for n in range(1, 25):
            val, _ = fixed_quad(
                lambda y: _Y_n_scalar(y, fx, p0, up_mean, n) * _U_n_scalar(y, fx, p0, up_mean, n),
                -fx, fx, n=quad)
            if np.abs(val) < 10 ** (-6):
                break
            YU_sum += val
        x_array_vals[i] = (-1 * ytimesU0 * gamma_vals + 2 * u_mean * Y0timesU0[0] / D + u_mean * YU_sum / D
                           + 2 * U0 * _C_corr_scalar(D, fx, p0, up_mean, gamma_vals)) * u_mean / 4 / fx

    return x_array_vals
