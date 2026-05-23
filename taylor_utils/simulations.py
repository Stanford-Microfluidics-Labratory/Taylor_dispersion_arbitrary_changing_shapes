"""Numba-accelerated Brownian dynamics simulations.

Three channel geometries are supported: :func:`simulation_rectangular`,
:func:`simulation_circular`, and :func:`simulation_dumbell`. Each stepper is
written as a single ``@njit(parallel=True)`` kernel that:

* parallelizes over particles with ``prange``, each particle running its full
  time evolution on one thread;
* does its own interpolation: trilinear on the regular velocity grids, linear
  on fine 1-D tables for the geometry callables (``f(x)`` etc.);
* finds wall crossings by bisection on the chord between a strictly-inside
  reference point and the candidate sub-step endpoint, then performs specular
  reflection of the remaining displacement (see ``_WALL_EPS_NUDGE`` /
  ``_BISECT_ITERS`` and the inline state machine in each runner).

Each particle is seeded with an independent random stream.
"""

import numpy as np
from numba import njit, prange

from .moments import (weighted_mean, weighted_variance, weighted_skewness,
                      weighted_kurtosis)


# --------------------------------------------------------------------------
# Shared scalar kernels
# --------------------------------------------------------------------------

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


@njit(cache=True, inline="always")
def _trilin(x0, dx, y0, dy, z0, dz, vals, x, y, z):
    """Trilinear interpolation on a regular 3-D grid, with linear extrapolation.

    Out of bounds, the nearest cell is reused and the local fraction is allowed
    to leave ``[0, 1]`` so the result extrapolates linearly.
    """
    nx, ny, nz = vals.shape

    fx = (x - x0) / dx
    ix = int(np.floor(fx))
    if ix < 0:
        ix = 0
    elif ix > nx - 2:
        ix = nx - 2
    tx = fx - ix

    fy = (y - y0) / dy
    iy = int(np.floor(fy))
    if iy < 0:
        iy = 0
    elif iy > ny - 2:
        iy = ny - 2
    ty = fy - iy

    fz = (z - z0) / dz
    iz = int(np.floor(fz))
    if iz < 0:
        iz = 0
    elif iz > nz - 2:
        iz = nz - 2
    tz = fz - iz

    c00 = vals[ix, iy, iz] * (1.0 - tx) + vals[ix + 1, iy, iz] * tx
    c01 = vals[ix, iy, iz + 1] * (1.0 - tx) + vals[ix + 1, iy, iz + 1] * tx
    c10 = vals[ix, iy + 1, iz] * (1.0 - tx) + vals[ix + 1, iy + 1, iz] * tx
    c11 = vals[ix, iy + 1, iz + 1] * (1.0 - tx) + vals[ix + 1, iy + 1, iz + 1] * tx
    c0 = c00 * (1.0 - ty) + c10 * ty
    c1 = c01 * (1.0 - ty) + c11 * ty
    return c0 * (1.0 - tz) + c1 * tz


@njit(cache=True, inline="always")
def _trilin_fill0(x0, dx, x_hi, y0, dy, y_hi, z0, dz, z_hi, vals, x, y, z):
    """Trilinear interpolation that returns 0 outside the grid."""
    if x < x0 or x > x_hi or y < y0 or y > y_hi or z < z0 or z > z_hi:
        return 0.0

    nx, ny, nz = vals.shape

    fx = (x - x0) / dx
    ix = int(np.floor(fx))
    if ix > nx - 2:
        ix = nx - 2
    tx = fx - ix

    fy = (y - y0) / dy
    iy = int(np.floor(fy))
    if iy > ny - 2:
        iy = ny - 2
    ty = fy - iy

    fz = (z - z0) / dz
    iz = int(np.floor(fz))
    if iz > nz - 2:
        iz = nz - 2
    tz = fz - iz

    c00 = vals[ix, iy, iz] * (1.0 - tx) + vals[ix + 1, iy, iz] * tx
    c01 = vals[ix, iy, iz + 1] * (1.0 - tx) + vals[ix + 1, iy, iz + 1] * tx
    c10 = vals[ix, iy + 1, iz] * (1.0 - tx) + vals[ix + 1, iy + 1, iz] * tx
    c11 = vals[ix, iy + 1, iz + 1] * (1.0 - tx) + vals[ix + 1, iy + 1, iz + 1] * tx
    c0 = c00 * (1.0 - ty) + c10 * ty
    c1 = c01 * (1.0 - ty) + c11 * ty
    return c0 * (1.0 - tz) + c1 * tz


# Wall-reflection numerical parameters.
#
# Each runner reflects off the wall as a state machine: an "inside" reference
# point (xi, ...) is kept strictly inside the channel, and a "candidate" end
# point (xe, ...) is the proposed sub-step landing site. Each iteration bisects
# the chord from inside to candidate for the wall crossing, reflects the
# remaining displacement, and nudges the inside point along the inward normal
# so the next iteration starts strictly inside.
_WALL_EPS_NUDGE = 1e-12   # inward offset after reflection
_BISECT_ITERS = 40        # bisection iterations


# --------------------------------------------------------------------------
# Rectangular channel
# --------------------------------------------------------------------------

@njit(cache=True, parallel=True)
def _run_rect(x_out, y_out, z_out,
              vx0, vdx, vy0, vdy, vz0, vdz, u_vals, v_vals, w_vals,
              fx_x0, fx_dx, fx_table, fxp_table,
              dt, sig_s, sampling_ratio, max_iter, seeds):
    """Per-particle Brownian dynamics kernel; columns are written in place.

    ``x_out``/``y_out``/``z_out`` are ``(Npts, Nt)`` with column 0 pre-filled
    with the initial positions. The loop over particles is a ``prange``.
    """
    Npts = x_out.shape[0]
    Nt = x_out.shape[1]

    for p in prange(Npts):
        np.random.seed(seeds[p])
        x = x_out[p, 0]
        y = y_out[p, 0]
        z = z_out[p, 0]

        for i in range(1, Nt):
            for _ in range(sampling_ratio):
                xp = x
                yp = y
                zp = z

                fxv = _interp1d(fx_x0, fx_dx, fx_table, xp)
                yn_norm = yp / fxv
                uu = _trilin(vx0, vdx, vy0, vdy, vz0, vdz, u_vals, xp, yn_norm, zp)
                vv = _trilin(vx0, vdx, vy0, vdy, vz0, vdz, v_vals, xp, yn_norm, zp)
                ww = _trilin(vx0, vdx, vy0, vdy, vz0, vdz, w_vals, xp, yn_norm, zp)

                xn = xp + uu * dt + sig_s * np.random.standard_normal()
                yn = yp + vv * dt + sig_s * np.random.standard_normal()
                zn = zp + ww * dt + sig_s * np.random.standard_normal()

                # Specular reflection off the channel walls. State machine:
                # (xi, yi, zi) is strictly inside, (xe, ye, ze) is the
                # candidate end. Each iteration handles the currently violated
                # wall (z first, then y) and updates both.
                xi = xp
                yi = yp
                zi = zp
                xe = xn
                ye = yn
                ze = zn
                reverted = False
                for _it in range(max_iter):
                    sign_z = 0.0
                    if ze > 1.0:
                        sign_z = 1.0
                    elif ze < -1.0:
                        sign_z = -1.0

                    a_e = _interp1d(fx_x0, fx_dx, fx_table, xe)
                    sign_y = 0.0
                    if ye > a_e:
                        sign_y = 1.0
                    elif ye < -a_e:
                        sign_y = -1.0

                    if sign_z == 0.0 and sign_y == 0.0:
                        break

                    if sign_z != 0.0:
                        # z-wall is flat at z = sign_z; chord crosses linearly.
                        s = (sign_z - zi) / (ze - zi)
                        ex = xi + s * (xe - xi)
                        ey = yi + s * (ye - yi)
                        ze = 2.0 * sign_z - ze
                        xi = ex
                        yi = ey
                        zi = sign_z * (1.0 - _WALL_EPS_NUDGE)
                        continue

                    # Bisect on the tapered y-wall +/- a(x).
                    s_lo = 0.0
                    s_hi = 1.0
                    for _b in range(_BISECT_ITERS):
                        s_mid = 0.5 * (s_lo + s_hi)
                        x_mid = xi + s_mid * (xe - xi)
                        y_mid = yi + s_mid * (ye - yi)
                        a_mid = _interp1d(fx_x0, fx_dx, fx_table, x_mid)
                        if sign_y * y_mid <= a_mid:
                            s_lo = s_mid
                        else:
                            s_hi = s_mid
                    s = s_lo
                    ex = xi + s * (xe - xi)
                    ez = zi + s * (ze - zi)
                    a_ex = _interp1d(fx_x0, fx_dx, fx_table, ex)
                    ey_wall = sign_y * a_ex
                    fp = _interp1d(fx_x0, fx_dx, fxp_table, ex)
                    norm = np.sqrt(1.0 + fp * fp)
                    nx_out = -sign_y * fp / norm
                    ny_out = sign_y / norm
                    d_x = xe - ex
                    d_y = ye - ey_wall
                    d_z = ze - ez
                    dot = d_x * nx_out + d_y * ny_out
                    xe = ex + d_x - 2.0 * dot * nx_out
                    ye = ey_wall + d_y - 2.0 * dot * ny_out
                    ze = ez + d_z
                    xi = ex - _WALL_EPS_NUDGE * nx_out
                    yi = ey_wall - _WALL_EPS_NUDGE * ny_out
                    zi = ez
                else:
                    reverted = True

                if reverted:
                    xe = xp
                    ye = yp
                    ze = zp

                x = xe
                y = ye
                z = ze

            x_out[p, i] = x
            y_out[p, i] = y
            z_out[p, i] = z


def simulation_rectangular(fx_func, Pe0, u, v, w, Nt0=500, seed=0,
                           sigx2_0=10, upper_bound=None, sampling_ratio=3,
                           fx_prime=None, max_iter=10):
    """Run a Brownian dynamics simulation in a rectangular channel.

    Parameters
    ----------
    fx_func : callable
        Half-width function f(x).
    Pe0 : float
        Peclet number.
    u, v, w : RegularGridInterpolator
        Velocity interpolators (axial, transverse-y, transverse-z) on a common
        regular grid, as produced by :func:`taylor_utils.velocity.create_velo_full`.
    Nt0, seed, sigx2_0, upper_bound, sampling_ratio : various
        Simulation parameters.

    Returns
    -------
    dict
        Simulation results including particle positions, moments, and parameters.
    """
    z0 = 1
    U0 = 1
    A = 4000
    B = A / Pe0
    D = B / A
    dt = 1 / B
    Npts = 5000
    Nt = int(Nt0 * A / sampling_ratio + 1)
    sig_s = np.sqrt(2 * D * dt)
    np.random.seed(seed)

    x = np.zeros((Npts, Nt))
    y = np.zeros((Npts, Nt))
    z = np.zeros((Npts, Nt))
    x[:, 0] = np.random.randn(Npts) * np.sqrt(sigx2_0)
    y[:, 0] = np.random.uniform(-1, 1, size=Npts) * fx_func(x[:, 0])
    z[:, 0] = np.random.uniform(-1, 1, size=Npts)

    gx, gy, gz = u.grid
    vx0, vdx = gx[0], gx[1] - gx[0]
    vy0, vdy = gy[0], gy[1] - gy[0]
    vz0, vdz = gz[0], gz[1] - gz[0]
    u_vals = np.ascontiguousarray(u.values, dtype=np.float64)
    v_vals = np.ascontiguousarray(v.values, dtype=np.float64)
    w_vals = np.ascontiguousarray(w.values, dtype=np.float64)

    x_reach = Nt * dt * sampling_ratio * U0 * 8.0 + 1000.0
    fx_lo, fx_hi = -1000.0, max(x_reach, 2000.0)
    n_fx = int((fx_hi - fx_lo) / 0.05) + 1
    fx_grid = np.linspace(fx_lo, fx_hi, n_fx)
    fx_table = np.asarray(fx_func(fx_grid), dtype=np.float64)
    if fx_prime is None:
        if hasattr(fx_func, "derivative"):
            fxp_table = np.asarray(fx_func.derivative()(fx_grid), dtype=np.float64)
        else:
            h = 1e-5
            fxp_table = (np.asarray(fx_func(fx_grid + h))
                         - np.asarray(fx_func(fx_grid - h))) / (2.0 * h)
    else:
        fxp_table = np.asarray(fx_prime(fx_grid), dtype=np.float64)
    fx_x0, fx_dx = fx_grid[0], fx_grid[1] - fx_grid[0]

    seeds = np.random.SeedSequence(seed).generate_state(Npts).astype(np.int64)

    _run_rect(x, y, z,
              vx0, vdx, vy0, vdy, vz0, vdz, u_vals, v_vals, w_vals,
              fx_x0, fx_dx, fx_table, fxp_table,
              dt, sig_s, int(sampling_ratio), int(max_iter), seeds)

    T = np.arange(Nt) * dt * sampling_ratio
    weighted_x = weighted_mean(x)
    weighted_var = weighted_variance(x, weighted_x)

    return {'x': x, 'y': y, 'z': z, 'T': T,
            'weighted_x': weighted_x,
            'weighted_var': weighted_var,
            'z0': z0, 'Pe0': Pe0, 'D': D, 'U0': U0, 'dt': dt,
            'Npts': Npts, 'Nt': Nt}


# --------------------------------------------------------------------------
# Circular channel
# --------------------------------------------------------------------------

@njit(cache=True, parallel=True)
def _run_circular(x_out, r_out, theta_out,
                  fx_x0, fx_dx, fx_table, fxp_table, beta_table,
                  dt, sig_s, sampling_ratio, max_iter, period, seeds):
    """Per-particle kernel for the axisymmetric circular channel.

    State is ``(x, r, theta)``. Axial and radial velocities are analytic in
    ``a(x) = fx_table`` and ``beta(x) = beta_table``; the wall is ``r = a(x)``.
    """
    Npts = x_out.shape[0]
    Nt = x_out.shape[1]

    for p in prange(Npts):
        np.random.seed(seeds[p])
        x = x_out[p, 0]
        r = r_out[p, 0]
        theta = theta_out[p, 0]

        for i in range(1, Nt):
            for _ in range(sampling_ratio):
                xp = x
                rp = r
                thetap = theta

                xmod = xp % period
                a = _interp1d(fx_x0, fx_dx, fx_table, xmod)
                beta = _interp1d(fx_x0, fx_dx, beta_table, xmod)
                inv = 1.0 / a
                ux = 2.0 * inv * inv * (1.0 - rp * rp * inv * inv)
                ur = 2.0 * inv * inv * beta * (rp * inv - rp * rp * rp * inv * inv * inv)

                xn = xp + ux * dt + sig_s * np.random.standard_normal()
                r_temp = rp + ur * dt
                x2 = r_temp * np.cos(thetap) + sig_s * np.random.standard_normal()
                x3 = r_temp * np.sin(thetap) + sig_s * np.random.standard_normal()
                theta = np.arctan2(x3, x2)
                rn = np.sqrt(x2 * x2 + x3 * x3)

                # Reflection off the wall r = a(x). rn is non-negative by the
                # Cartesian construction above, so r=0 is not a boundary.
                xi = xp
                ri = rp
                xe = xn
                re = rn
                reverted = False
                for _it in range(max_iter):
                    a_e = _interp1d(fx_x0, fx_dx, fx_table, xe % period)
                    if re <= a_e:
                        break

                    # Bisect for the first wall crossing on the chord
                    # (xi, ri) -> (xe, re).
                    s_lo = 0.0
                    s_hi = 1.0
                    for _b in range(_BISECT_ITERS):
                        s_mid = 0.5 * (s_lo + s_hi)
                        x_mid = xi + s_mid * (xe - xi)
                        r_mid = ri + s_mid * (re - ri)
                        a_mid = _interp1d(fx_x0, fx_dx, fx_table, x_mid % period)
                        if r_mid <= a_mid:
                            s_lo = s_mid
                        else:
                            s_hi = s_mid

                    s = s_lo
                    ex = xi + s * (xe - xi)
                    ex_mod = ex % period
                    er = _interp1d(fx_x0, fx_dx, fx_table, ex_mod)
                    slope = _interp1d(fx_x0, fx_dx, fxp_table, ex_mod)
                    norm = np.sqrt(1.0 + slope * slope)
                    nx_out = -slope / norm
                    nr_out = 1.0 / norm

                    # Reflect remaining displacement across the outward normal.
                    d_x = xe - ex
                    d_r = re - er
                    dot = d_x * nx_out + d_r * nr_out
                    xe = ex + d_x - 2.0 * dot * nx_out
                    re = er + d_r - 2.0 * dot * nr_out

                    # If reflection sent the candidate past the axis, fold it
                    # back onto r >= 0 and rotate theta by pi.
                    if re < 0.0:
                        re = -re
                        theta = theta + np.pi

                    xi = ex - _WALL_EPS_NUDGE * nx_out
                    ri = er - _WALL_EPS_NUDGE * nr_out
                else:
                    reverted = True

                if reverted:
                    xe = xp
                    re = rp
                    theta = thetap

                x = xe
                r = re

            x_out[p, i] = x
            r_out[p, i] = r
            theta_out[p, i] = theta


def simulation_circular(Pe0, func_x, func_x_prime, Nt0=500, seed=0, sigx2_0=300,
                        upper_bound=None, sampling_ratio=100, A=1500):
    """Run a Brownian dynamics simulation in a circular channel.

    Parameters
    ----------
    Pe0 : float
        Peclet number.
    func_x : callable
        Radius function a(x).
    func_x_prime : callable
        Derivative of radius function a'(x).

    Returns
    -------
    dict
        Simulation results including particle positions, moments, and parameters.
    """
    U0 = 1
    a0 = 1
    B = A / Pe0
    D = B / A
    dt = 1 / B
    Npts = 5000
    Nt = int(Nt0 * A / sampling_ratio + 1)
    sig_s = np.sqrt(2 * D * dt)
    period = 2 * np.pi * 400

    np.random.seed(seed)

    r = np.zeros((Npts, Nt))
    theta = np.zeros((Npts, Nt))
    x = np.zeros((Npts, Nt))
    x[:, 0] = np.random.randn(Npts) * np.sqrt(sigx2_0)
    theta[:, 0] = np.random.rand(Npts) * 2 * np.pi - np.pi
    r[:, 0] = np.sqrt(np.random.rand(Npts) * func_x(x[:, 0]) ** 2)

    if upper_bound:
        x_range_est = upper_bound
    else:
        x_range_est = Nt * dt * U0 * 8

    # The wall loop in _run_circular looks up the table at ``xe % period``,
    # so the table must cover at least [0, period).
    x_hi = max(x_range_est, period + 1.0)
    x_range_for_ur = np.linspace(-500, x_hi, 100000 + 1)
    dx = (x_hi + 500) / 100000
    a_x = func_x(x_range_for_ur)
    beta_for_ur = np.gradient(a_x, dx)
    fx_table = np.asarray(a_x, dtype=np.float64)
    fxp_table = np.asarray(func_x_prime(x_range_for_ur), dtype=np.float64)
    beta_table = np.asarray(beta_for_ur, dtype=np.float64)

    seeds = np.random.SeedSequence(seed).generate_state(Npts).astype(np.int64)

    _run_circular(x, r, theta,
                  -500.0, dx, fx_table, fxp_table, beta_table,
                  dt, sig_s, int(sampling_ratio), 10, float(period), seeds)

    T = np.arange(Nt) * dt * sampling_ratio
    weighted_x = weighted_mean(x)
    weighted_var = weighted_variance(x, weighted_x)
    weighted_skew = weighted_skewness(x, weighted_x, weighted_var)
    weighted_kurt = weighted_kurtosis(x, weighted_x, weighted_var)

    result = {'x': x, 'r': r, 'theta': theta, 'T': T,
              'weighted_x': weighted_x,
              'weighted_var': weighted_var,
              'weighted_skewness': weighted_skew,
              'weighted_kurtosis': weighted_kurt,
              'dx': dx, 'a_x': a_x,
              'a0': a0, 'Pe0': Pe0, 'D': D, 'U0': U0, 'dt': dt, 'Npts': Npts, 'Nt': Nt}
    return result


# --------------------------------------------------------------------------
# Dumbbell channel
# --------------------------------------------------------------------------

@njit(cache=True, inline="always")
def _z_func_dumbbell(x, slope):
    """Handle half-height function H(x)/2 used by the dumbbell geometry."""
    if x < 0.0:
        return 1.0
    elif x < 1200.0:
        return slope * x + 1.0
    else:
        return 0.1


@njit(cache=True, inline="always")
def _z_prime_dumbbell(x, slope):
    """Derivative of :func:`_z_func_dumbbell`."""
    if 0.0 <= x < 1200.0:
        return slope
    return 0.0


@njit(cache=True, inline="always")
def _in_dumbbell(H, y, z, L, R):
    """True when (y, z) lies inside the dumbbell cross-section at handle height H."""
    half_L = 0.5 * L
    R2 = R * R
    left_circle = (y + half_L) ** 2 + z * z <= R2
    right_circle = (y - half_L) ** 2 + z * z <= R2
    handle = (abs(y) <= half_L) and (abs(z) <= 0.5 * H) and (not left_circle) \
        and (not right_circle)
    return left_circle or right_circle or handle


@njit(cache=True, parallel=True)
def _run_dumbbell(x_out, y_out, z_out,
                  ux0, udx, ux_hi, uy0, udy, uy_hi, uz0, udz, uz_hi,
                  ux_vals, uy_vals, uz_vals,
                  dt, sig_s, scale_factor, sampling_ratio, max_iter,
                  L, R, slope, seeds):
    """Per-particle kernel for the dumbbell channel.

    The cross-section is two circles (centres ``y = +/-L/2``, radius ``R``)
    joined by a handle of half-height ``z_func(x)``. Velocities come from
    OpenFOAM-derived regular-grid interpolators.
    """
    Npts = x_out.shape[0]
    Nt = x_out.shape[1]
    half_L = 0.5 * L
    R2 = R * R

    for p in prange(Npts):
        np.random.seed(seeds[p])
        x = x_out[p, 0]
        y = y_out[p, 0]
        z = z_out[p, 0]

        for i in range(1, Nt):
            for _ in range(sampling_ratio):
                xp = x
                yp = y
                zp = z

                cx = xp / scale_factor
                if cx < 0.0:
                    cx = 0.0
                elif cx > 120.0:
                    cx = 120.0
                ab = 1.0 if (cx != 120.0 and cx > 0.0) else 0.0

                uu = _trilin_fill0(ux0, udx, ux_hi, uy0, udy, uy_hi,
                                   uz0, udz, uz_hi, ux_vals, cx, yp, zp)
                vv = _trilin_fill0(ux0, udx, ux_hi, uy0, udy, uy_hi,
                                   uz0, udz, uz_hi, uy_vals, cx, yp, zp)
                ww = _trilin_fill0(ux0, udx, ux_hi, uy0, udy, uy_hi,
                                   uz0, udz, uz_hi, uz_vals, cx, yp, zp)

                xn = xp + uu * dt + sig_s * np.random.standard_normal()
                yn = yp + vv * dt * ab / scale_factor + sig_s * np.random.standard_normal()
                zn = zp + ww * dt * ab / scale_factor + sig_s * np.random.standard_normal()

                # Reflection off the dumbbell walls. (xi, yi, zi) tracks the
                # strictly-inside reference; the classification of which wall
                # to reflect off uses this reference.
                xi = xp
                yi = yp
                zi = zp
                xe = xn
                ye = yn
                ze = zn
                reverted = False
                for _it in range(max_iter):
                    H_e = 2.0 * _z_func_dumbbell(xe, slope)
                    if _in_dumbbell(H_e, ye, ze, L, R):
                        break

                    # Classify (xi, yi, zi), the current strictly-inside ref.
                    H_i = 2.0 * _z_func_dumbbell(xi, slope)
                    left_circle = (yi < 0.0) and ((yi + half_L) ** 2 + zi * zi <= R2)
                    right_circle = (yi > 0.0) and ((yi - half_L) ** 2 + zi * zi <= R2)
                    out_handle = (abs(yi) <= half_L) and (abs(zi) <= 0.5 * H_i) \
                        and (not left_circle) and (not right_circle)

                    dxs = xe - xi
                    dys = ye - yi
                    dzs = ze - zi

                    if left_circle or right_circle:
                        cy = -half_L if left_circle else half_L
                        # Bisect for (y(s) - cy)^2 + z(s)^2 = R^2.
                        s_lo = 0.0
                        s_hi = 1.0
                        for _b in range(_BISECT_ITERS):
                            s_mid = 0.5 * (s_lo + s_hi)
                            y_mid = yi + s_mid * dys
                            z_mid = zi + s_mid * dzs
                            g_mid = (y_mid - cy) ** 2 + z_mid * z_mid - R2
                            if g_mid <= 0.0:
                                s_lo = s_mid
                            else:
                                s_hi = s_mid
                        s = s_lo
                        ex = xi + s * dxs
                        ey = yi + s * dys
                        ez = zi + s * dzs
                        # Outward normal: radial from cap center (cy, 0) in y-z.
                        nx_ = 0.0
                        ny_ = ey - cy
                        nz_ = ez
                    elif out_handle:
                        sign_z = 1.0 if dzs > 0.0 else -1.0
                        # Bisect for sign_z * z(s) - z_func(x(s)) = 0.
                        s_lo = 0.0
                        s_hi = 1.0
                        for _b in range(_BISECT_ITERS):
                            s_mid = 0.5 * (s_lo + s_hi)
                            x_mid = xi + s_mid * dxs
                            z_mid = zi + s_mid * dzs
                            zfm = _z_func_dumbbell(x_mid, slope)
                            if sign_z * z_mid <= zfm:
                                s_lo = s_mid
                            else:
                                s_hi = s_mid
                        s = s_lo
                        ex = xi + s * dxs
                        ey = yi + s * dys
                        ez = zi + s * dzs
                        # Outward normal: grad(sign_z*z - z_func(x))
                        # = (-sign_z * z', 0, sign_z).
                        nx_ = -sign_z * _z_prime_dumbbell(ex, slope)
                        ny_ = 0.0
                        nz_ = sign_z
                    else:
                        # Inside reference falls in no region. Revert to
                        # sub-step start.
                        reverted = True
                        break

                    norm = np.sqrt(nx_ * nx_ + ny_ * ny_ + nz_ * nz_)
                    if norm <= 0.0:
                        reverted = True
                        break
                    nx_ /= norm
                    ny_ /= norm
                    nz_ /= norm

                    # Reflect remaining displacement across the outward normal.
                    d_x = xe - ex
                    d_y = ye - ey
                    d_z = ze - ez
                    dot = d_x * nx_ + d_y * ny_ + d_z * nz_
                    xe = ex + d_x - 2.0 * dot * nx_
                    ye = ey + d_y - 2.0 * dot * ny_
                    ze = ez + d_z - 2.0 * dot * nz_

                    xi = ex - _WALL_EPS_NUDGE * nx_
                    yi = ey - _WALL_EPS_NUDGE * ny_
                    zi = ez - _WALL_EPS_NUDGE * nz_
                else:
                    reverted = True

                if reverted:
                    xe = xp
                    ye = yp
                    ze = zp

                x = xe
                y = ye
                z = ze

            x_out[p, i] = x
            y_out[p, i] = y
            z_out[p, i] = z


def simulation_dumbell(ux, uy, uz, Pe0, L0, scale_factor=10, Nt0=500, seed=0,
                       sigx2_0=300, upper_bound=None, sampling_ratio=100):
    """Run a Brownian dynamics simulation in a dumbbell-shaped channel.

    Parameters
    ----------
    ux, uy, uz : RegularGridInterpolator
        Velocity component interpolators on a common regular (x, y, z) grid,
        as produced by :func:`taylor_utils.openfoam.build_openFoam_interp`.
    Pe0 : float
        Peclet number.
    L0 : float
        Characteristic length scale.
    scale_factor : float
        Scaling between simulation coords and interpolator coords. The OpenFOAM
        channel is ``scale_factor`` times shorter than the actual channel.

    Returns
    -------
    dict
        Simulation results including particle positions, moments, and parameters.
    """
    U0 = 1
    A = 15000
    B = A / Pe0
    dt = 1 / B
    D = L0 * B / A
    Npts = 5000
    Nt = int(Nt0 * A / sampling_ratio + 1)
    sig_s = np.sqrt(2 * D * dt)
    np.random.seed(seed)
    max_iter = 10

    slope = -0.00075
    z_func = lambda x: np.where(x < 1200, np.where(x < 0, 1, slope * x + 1), 0.1)
    L, R = 4.0, 1.0

    # Initial positions, area-weighted across the two lobes and handle.
    y = np.zeros((Npts, Nt))
    z = np.zeros((Npts, Nt))
    x = np.zeros((Npts, Nt))

    area_circle = np.pi * R ** 2
    area_handle = 2 * L - np.pi * R ** 2
    total_area = 2 * area_circle + area_handle
    prob_circle = area_circle / total_area

    rand_region = np.random.rand(Npts)
    y0 = np.zeros(Npts)
    z0 = np.zeros(Npts)

    left_mask = rand_region < prob_circle
    right_mask = (rand_region >= prob_circle) & (rand_region < 2 * prob_circle)
    handle_mask = rand_region >= 2 * prob_circle

    def sample_circle(mask, center_y):
        num = np.sum(mask)
        rr = R * np.sqrt(np.random.rand(num))
        th = 2 * np.pi * np.random.rand(num)
        y0[mask] = center_y + rr * np.cos(th)
        z0[mask] = rr * np.sin(th)

    sample_circle(left_mask, -L / 2)
    sample_circle(right_mask, L / 2)
    num_h = np.sum(handle_mask)
    x[:, 0] = np.random.randn(Npts) * np.sqrt(sigx2_0)
    H = 2 * z_func(x[:, 0])
    y0[handle_mask] = np.random.uniform(-L / 2, L / 2, size=num_h)
    z0[handle_mask] = np.random.uniform(-H[handle_mask] / 2, H[handle_mask] / 2, size=num_h)

    y[:, 0] = y0
    z[:, 0] = z0

    gx, gy, gz = ux.grid
    ux0, udx = gx[0], gx[1] - gx[0]
    uy0, udy = gy[0], gy[1] - gy[0]
    uz0, udz = gz[0], gz[1] - gz[0]
    ux_hi, uy_hi, uz_hi = gx[-1], gy[-1], gz[-1]
    ux_vals = np.ascontiguousarray(ux.values, dtype=np.float64)
    uy_vals = np.ascontiguousarray(uy.values, dtype=np.float64)
    uz_vals = np.ascontiguousarray(uz.values, dtype=np.float64)

    seeds = np.random.SeedSequence(seed).generate_state(Npts).astype(np.int64)

    _run_dumbbell(x, y, z,
                  ux0, udx, ux_hi, uy0, udy, uy_hi, uz0, udz, uz_hi,
                  ux_vals, uy_vals, uz_vals,
                  dt, sig_s, float(scale_factor), int(sampling_ratio),
                  int(max_iter), L, R, slope, seeds)

    T = np.arange(Nt) * dt * sampling_ratio
    weighted_x = weighted_mean(x)
    weighted_var = weighted_variance(x, weighted_x)
    weighted_skew = weighted_skewness(x, weighted_x, weighted_var)
    weighted_kurt = weighted_kurtosis(x, weighted_x, weighted_var)

    result = {'x': x, 'y': y, 'z': z, 'T': T,
              'weighted_x': weighted_x,
              'weighted_var': weighted_var,
              'weighted_skewness': weighted_skew,
              'weighted_kurtosis': weighted_kurt,
              'Pe0': Pe0, 'D': D, 'U0': U0, 'dt': dt, 'Npts': Npts, 'Nt': Nt,
              'Sampling_ratio': sampling_ratio}

    return result
