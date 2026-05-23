"""Estimate cross-sectional diffusion timescale for dumbbell-shaped channels.

Runs Monte Carlo simulations of pure diffusion in a 2-D dumbbell cross-section
to measure how long particles take to equilibrate between the two lobes.
"""

import numpy as np
from numba import njit, prange


_WALL_EPS_NUDGE = 1e-12
_BISECT_ITERS = 40


@njit(cache=True, inline="always")
def _in_dumbbell_2d(y, z, half_L, R2, half_H):
    """True when (y, z) lies inside the 2-D dumbbell cross-section."""
    left_circle = (y + half_L) ** 2 + z * z <= R2
    right_circle = (y - half_L) ** 2 + z * z <= R2
    handle = (abs(y) <= half_L) and (abs(z) <= half_H) \
        and (not left_circle) and (not right_circle)
    return left_circle or right_circle or handle


@njit(cache=True, inline="always")
def _classify(y, z, half_L, R2, half_H):
    """Return region code for a point: 0 left circle, 1 right circle,
    2 handle, -1 outside."""
    if (y + half_L) ** 2 + z * z <= R2:
        return 0
    if (y - half_L) ** 2 + z * z <= R2:
        return 1
    if abs(y) <= half_L and abs(z) <= half_H:
        return 2
    return -1


@njit(cache=True)
def _run_trial(N_particles, H, D, dt, L, R, max_iter):
    """Monte-Carlo evolution for a single trial.

    Returns the time at which the two lobes reach equal occupation, starting
    from all particles in the left circle.

    Wall reflection uses a state machine: (yi, zi) tracks a strictly-inside
    reference (initialized at sub-step start, epsilon-nudged inside after each
    bounce); the wall crossing on the chord (yi, zi) -> (ye, ze) is found by
    bisection; the remaining displacement is reflected across the outward
    normal. On failure (max_iter exhausted or degenerate normal), the sub-step
    is reverted.
    """
    half_L = 0.5 * L
    R2 = R * R
    half_H = 0.5 * H

    y = np.full(N_particles, -half_L)
    z = np.zeros(N_particles)

    step = np.sqrt(2.0 * D * dt)
    t = 0.0

    n_left = N_particles
    n_right = 0

    while n_left > n_right:
        t += dt
        n_left = 0
        n_right = 0

        for p in range(N_particles):
            yp = y[p]
            zp = z[p]
            yn = yp + step * np.random.standard_normal()
            zn = zp + step * np.random.standard_normal()

            yi = yp
            zi = zp
            ye = yn
            ze = zn
            reverted = False
            for _it in range(max_iter):
                if _in_dumbbell_2d(ye, ze, half_L, R2, half_H):
                    break

                left_circle = (yi < 0.0) and ((yi + half_L) ** 2 + zi * zi <= R2)
                right_circle = (yi > 0.0) and ((yi - half_L) ** 2 + zi * zi <= R2)
                out_handle = (abs(yi) <= half_L) and (abs(zi) <= half_H) \
                    and (not left_circle) and (not right_circle)

                dys = ye - yi
                dzs = ze - zi

                if left_circle or right_circle:
                    cy = -half_L if left_circle else half_L
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
                    ey = yi + s * dys
                    ez = zi + s * dzs
                    ny_ = ey - cy
                    nz_ = ez
                elif out_handle:
                    sign_z = 1.0 if dzs > 0.0 else -1.0
                    s_lo = 0.0
                    s_hi = 1.0
                    for _b in range(_BISECT_ITERS):
                        s_mid = 0.5 * (s_lo + s_hi)
                        z_mid = zi + s_mid * dzs
                        if sign_z * z_mid <= half_H:
                            s_lo = s_mid
                        else:
                            s_hi = s_mid
                    s = s_lo
                    ey = yi + s * dys
                    ez = zi + s * dzs
                    ny_ = 0.0
                    nz_ = sign_z
                else:
                    reverted = True
                    break

                norm = np.sqrt(ny_ * ny_ + nz_ * nz_)
                if norm <= 0.0:
                    reverted = True
                    break
                ny_ /= norm
                nz_ /= norm

                d_y = ye - ey
                d_z = ze - ez
                dot = d_y * ny_ + d_z * nz_
                ye = ey + d_y - 2.0 * dot * ny_
                ze = ez + d_z - 2.0 * dot * nz_

                yi = ey - _WALL_EPS_NUDGE * ny_
                zi = ez - _WALL_EPS_NUDGE * nz_
            else:
                reverted = True

            if reverted:
                ye = yp
                ze = zp

            y[p] = ye
            z[p] = ze

            code = _classify(ye, ze, half_L, R2, half_H)
            if code == 0:
                n_left += 1
            elif code == 1:
                n_right += 1

    return t


@njit(cache=True, parallel=True)
def _run_all_trials(N_particles, H, D, dt, L, R, max_iter, n_trials, seeds):
    """Run ``n_trials`` trials in parallel and return their times."""
    t_array = np.zeros(n_trials)
    for i in prange(n_trials):
        np.random.seed(seeds[i])
        t_array[i] = _run_trial(N_particles, H, D, dt, L, R, max_iter)
    return t_array


def estimate_diffusion_timescale(H_array, D=0.6, dt=1e-3, N_particles=5000,
                                 L=4.0, R=1.0, n_trials=100):
    """Estimate the cross-sectional diffusion timescale for a dumbbell geometry.

    For each handle height H in ``H_array``, runs ``n_trials`` Monte Carlo
    simulations of pure diffusion starting from the left circle.  The timescale
    is the average time for particles to reach equal occupation of both circles.

    Parameters
    ----------
    H_array : array_like
        Array of handle heights to test.
    D : float
        Diffusion coefficient.
    dt : float
        Time step.
    N_particles : int
        Number of particles per trial.
    L : float
        Distance between circle centers.
    R : float
        Radius of each circle.
    n_trials : int
        Number of independent trials per H value.

    Returns
    -------
    dict
        ``{"H_values": H_array, "time_results": time_results}``
    """
    H_array = np.asarray(H_array)
    time_results = np.zeros_like(H_array, dtype=float)

    max_iter = 10

    for q in range(H_array.shape[0]):
        H = float(H_array[q])
        seeds = np.random.randint(0, 2 ** 31 - 1, size=n_trials).astype(np.int64)
        t_array = _run_all_trials(int(N_particles), H, float(D), float(dt),
                                  float(L), float(R), int(max_iter),
                                  int(n_trials), seeds)
        time_results[q] = np.mean(t_array)

    return {
        'H_values': H_array,
        'time_results': time_results,
    }
