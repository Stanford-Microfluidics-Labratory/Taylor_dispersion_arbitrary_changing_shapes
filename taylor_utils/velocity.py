import numpy as np
from numba import njit, prange
from scipy.integrate import quad_vec
from scipy.interpolate import RegularGridInterpolator


def k_m(m):
    return (m + 0.5) * np.pi


def dp0_dx(fx, N=50):
    n = np.arange(N)
    kn = k_m(n)[:, None]
    fx_temp = fx.copy().reshape(1, -1)
    return (3 / (4 * (np.sum(6 * kn ** (-5) * np.tanh(kn * fx_temp), axis=0) - fx_temp))).reshape(1, -1)


def Un(fx, y, p0, n):
    kn = k_m(n)
    return 2 * (-1) ** n * p0 * (np.cosh(kn * y) / np.cosh(kn * fx) - 1) / kn ** 3


def dUn_dx(fx, dfdx, y, n, h=0.001):
    fx_temp = fx.copy().reshape(1, -1)
    p0_plush = dp0_dx(fx_temp + h)
    p0_minush = dp0_dx(fx_temp - h)
    U_plush = Un(fx + h, y, p0_plush, n)
    U_minush = Un(fx - h, y, p0_minush, n)
    return dfdx * (U_plush - U_minush) / (2 * h)


def Tm(fx, z, m):
    return (fx / m / np.pi * np.tanh(m * np.pi / fx) + fx ** 2 / m ** 2 / np.pi ** 2) * np.sinh(
        m * np.pi * z / fx) - fx / m / np.pi * z * np.cosh(m * np.pi * z / fx)


def anbar(fx, dfdx, n):
    kn = k_m(n)
    return -dUn_dx(fx, dfdx, fx.copy(), n) / (
        fx * kn * np.cosh(kn * fx) - np.cosh(kn * fx) * np.tanh(kn * fx) - fx * kn * np.sinh(kn * fx) * np.tanh(
            kn * fx))


def Anm(fx, n, m):
    kn = k_m(n)
    return 2 * (-1) ** (m + 1) * m * np.pi * kn / fx / (
        fx * kn * np.cosh(kn * fx) - np.cosh(kn * fx) * np.tanh(kn * fx) - fx * kn * np.sinh(
            kn * fx) * np.tanh(kn * fx)) * \
        quad_vec(lambda z, fx: Tm(fx, z, m) * np.sin(kn * z), 0, 1, args=(fx,))[0]


def bmbar(fx, dfdx, m, N=30):
    n = np.arange(N)[:, None, None, None]
    kn = k_m(n)
    bm_vals = np.zeros((len(m), fx.size))
    for i, mval in enumerate(m.flatten()):
        int_val = quad_vec(
            lambda y, fx: (y <= fx) * dUn_dx(fx, dfdx, y, n) * np.cos(mval * np.pi * y / fx),
            0, np.max(fx), args=(fx,))[0]
        bm = 2 / (mval * np.pi * Tm(fx, 1, mval)) * np.sum(int_val * (-1) ** (n + 1) / kn, axis=0)
        bm_vals[i, :] = bm.flatten()
    return bm_vals


def Bmn(fx, m, n):
    kn = k_m(n)
    return 2 * (-1) ** (n + 1) / kn / m / np.pi / Tm(fx, 1, m) * quad_vec(
        lambda y, fx: (y <= fx) * (
            fx * kn * np.cosh(kn * y) - np.cosh(kn * y) * np.tanh(kn * fx) - y * kn * np.sinh(
                kn * y) * np.tanh(kn * fx)) * np.cos(m * np.pi * y / fx),
        0, np.max(fx), args=(fx,))[0]


def compute_an_bm(fx, dfdx, num_linear=20):
    num_x = fx.size
    n = np.arange(num_linear)[:, None, None, None]
    m = np.arange(1, num_linear + 1)[:, None, None, None]
    abars = anbar(fx, dfdx, n).reshape(num_linear, num_x).T
    bbars = bmbar(fx, dfdx, m).reshape(num_linear, num_x).T
    A = np.zeros((num_x, num_linear, num_linear))
    B = np.zeros((num_x, num_linear, num_linear))
    for i in range(num_linear):
        for j in range(num_linear):
            A[:, i, j] = Anm(fx, i, j + 1).flatten()
            B[:, j, i] = Bmn(fx, j + 1, i).flatten()
    I = np.eye(num_linear)[None, :, :]
    ans = np.linalg.solve(I - A @ B, (abars + (A @ bbars[..., None]).squeeze(-1))[..., None]).squeeze(-1)
    bms = bbars + (B @ ans[..., None]).squeeze(-1)
    return ans.T, bms.T


@njit(cache=True, parallel=True, fastmath=True)
def _u_kernel(fxv, yv, zv, p0v, N):
    """Evaluation of ``u_func``.

    ``fxv``/``p0v`` are ``(num_x,)``, ``yv`` is ``(num_yz, num_x)``,
    ``zv`` is ``(num_yz, num_x)``. Result is ``(num_yz, num_yz, num_x)``.
    """
    num_yz = yv.shape[0]
    num_x = fxv.shape[0]
    out = np.zeros((num_yz, num_yz, num_x))
    for ix in prange(num_x):
        fxx = fxv[ix]
        p0x = p0v[ix]
        for nn in range(N):
            kn = (nn + 0.5) * np.pi
            sgn = 1.0 if (nn % 2 == 0) else -1.0
            cosh_kfx = np.cosh(kn * fxx)
            coef = 2.0 * sgn * p0x / kn ** 3
            for iy in range(num_yz):
                un = coef * (np.cosh(kn * yv[iy, ix]) / cosh_kfx - 1.0)
                for iz in range(num_yz):
                    out[iy, iz, ix] += un * np.cos(kn * zv[iz, ix])
    return out


def u_func(fx, y, z, p0, N=30):
    nx = fx.shape[-1]
    fxv = np.ascontiguousarray(np.reshape(fx, (nx,)), dtype=np.float64)
    p0v = np.ascontiguousarray(np.reshape(p0, (nx,)), dtype=np.float64)
    yv = np.ascontiguousarray(np.reshape(y, (-1, nx)), dtype=np.float64)
    zv = np.ascontiguousarray(np.reshape(z, (-1, nx)), dtype=np.float64)
    return _u_kernel(fxv, yv, zv, p0v, N)


@njit(cache=True, parallel=True, fastmath=True)
def _v_kernel(fxv, yv, zv, ans, bms, num_linear):
    """Evaluation of ``v_func``.

    ``ans``/``bms`` are ``(num_linear, num_x)``; ``fxv`` is ``(num_x,)``;
    ``yv``/``zv`` are ``(num_yz, num_x)``. Result is
    ``(num_yz, num_yz, num_x)``.
    """
    num_yz = yv.shape[0]
    num_x = fxv.shape[0]
    out = np.zeros((num_yz, num_yz, num_x))
    for ix in prange(num_x):
        fxx = fxv[ix]
        for mm in range(num_linear):
            km = (mm + 0.5) * np.pi
            tanh_kfx = np.tanh(km * fxx)
            a = ans[mm, ix]
            for iy in range(num_yz):
                yy = yv[iy, ix]
                term = a * (fxx * np.sinh(km * yy) - yy * np.cosh(km * yy) * tanh_kfx)
                for iz in range(num_yz):
                    out[iy, iz, ix] += term * np.cos(km * zv[iz, ix])
        for idx in range(num_linear):
            nn = idx + 1
            np_over_fx = nn * np.pi / fxx
            tanh_npfx = np.tanh(np_over_fx)
            b = bms[idx, ix]
            for iy in range(num_yz):
                sin_y = np.sin(np_over_fx * yv[iy, ix])
                term = b * sin_y
                for iz in range(num_yz):
                    zz = zv[iz, ix]
                    out[iy, iz, ix] += term * (
                        tanh_npfx * np.cosh(np_over_fx * zz) - zz * np.sinh(np_over_fx * zz))
    return out


def v_func(fx, y, z, ans, bms, num_linear=20):
    nx = fx.shape[-1]
    fxv = np.ascontiguousarray(np.reshape(fx, (nx,)), dtype=np.float64)
    yv = np.ascontiguousarray(np.reshape(y, (-1, nx)), dtype=np.float64)
    zv = np.ascontiguousarray(np.reshape(z, (-1, nx)), dtype=np.float64)
    ansv = np.ascontiguousarray(np.reshape(ans, (num_linear, nx)), dtype=np.float64)
    bmsv = np.ascontiguousarray(np.reshape(bms, (num_linear, nx)), dtype=np.float64)
    return _v_kernel(fxv, yv, zv, ansv, bmsv, num_linear)


def w_func(fx, dfdx, y, z, p0, ans, bms, num_linear=20):
    m = np.arange(num_linear)[:, None, None, None]
    n = np.arange(1, num_linear + 1)[:, None, None, None]
    km = k_m(m)
    return -np.sum(np.sin(km * z) / km * (
        dUn_dx(fx, dfdx, y, m) + ans * (
            fx * km * np.cosh(km * y) - np.cosh(km * y) * np.tanh(km * fx) - y * km * np.sinh(
                km * y) * np.tanh(km * fx))), axis=0) \
        - np.sum(n * np.pi * bms * Tm(fx, z, n) * np.cos(n * np.pi * y / fx) / fx, axis=0)


def create_velo_full(fx_func, num_x_grid=100, num_y_z=25, num_linear=20, origin=-200, x_max=900):
    """Create velocity field interpolators for a rectangular channel.

    Parameters
    ----------
    fx_func : callable
        Half-width function f(x) with a .derivative() method (e.g. CubicSpline).
    num_x_grid : int
        Number of grid points in the x direction.
    num_y_z : int
        Number of grid points in y and z directions.
    num_linear : int
        Number of linear modes.
    origin : float
        Starting x value.
    x_max : float
        Ending x value.

    Returns
    -------
    u_interp, v_interp, w_interp : RegularGridInterpolator
        Velocity component interpolators.
    """
    x_velo = np.linspace(origin, x_max, num_x_grid)
    fx = np.atleast_1d(fx_func(x_velo))[None, :]
    dfdx_func = fx_func.derivative()
    dfdx = dfdx_func(x_velo)[None, None, None, :]
    p0 = dp0_dx(fx)
    y_velo = np.linspace(-1, 1, num_y_z)[:, None] * fx
    z_velo = np.linspace(-1, 1, num_y_z)[:, None] * np.ones(num_x_grid)
    y_velo = y_velo[None, :, None]
    fx = fx[None, None, :]
    p0 = p0[None, None, :]
    z_velo = z_velo[None, None, :]
    ans, bms = compute_an_bm(fx.ravel(), dfdx.ravel(), num_linear)

    ans = ans[:, None, None, :]
    bms = bms[:, None, None, :]
    u_vals = (u_func(fx, y_velo, z_velo, p0) * 4 * fx_func(0)).transpose(2, 0, 1)
    v_vals = (v_func(fx, y_velo, z_velo, ans, bms, num_linear) * 4 * fx_func(0)).transpose(2, 0, 1)
    w_vals = (w_func(fx, dfdx, y_velo, z_velo, p0, ans, bms, num_linear) * 4 * fx_func(0)).transpose(2, 0, 1)

    y_grid = np.linspace(-1, 1, num_y_z)
    temp_grid = np.linspace(-1, 1, num_y_z)

    u_interp = RegularGridInterpolator((x_velo, y_grid, temp_grid), u_vals,
                                       bounds_error=False, fill_value=None)
    v_interp = RegularGridInterpolator((x_velo, y_grid, temp_grid), v_vals,
                                       bounds_error=False, fill_value=None)
    w_interp = RegularGridInterpolator((x_velo, y_grid, temp_grid), w_vals,
                                       bounds_error=False, fill_value=None)

    return u_interp, v_interp, w_interp
