"""Wall-crossing utilities for Brownian dynamics simulators.

A trajectory substep is a straight segment ``p(t) = p_prev + t*(p_new - p_prev)``
for ``t in [0, 1]``. A signed gap ``g(t)`` is constructed so that ``g < 0`` means
the particle is inside the channel and ``g > 0`` means outside. The smallest
root ``s`` of ``g`` in ``(0, 1]`` is the wall-crossing fraction.
"""

import numpy as np


def find_crossing(g0, ghalf, g1, default=1.0):
    """Smallest root in ``(0, 1]`` of the quadratic through
    ``(0, g0)``, ``(1/2, ghalf)``, ``(1, g1)``.

    ``g0``, ``ghalf``, ``g1`` are arrays of the same shape. Returns an array of
    crossing fractions. Entries where the quadratic has no root in ``(0, 1]``
    are set to ``default``.
    """
    g0 = np.asarray(g0, dtype=float)
    ghalf = np.asarray(ghalf, dtype=float)
    g1 = np.asarray(g1, dtype=float)

    a = 2.0 * (g0 - 2.0 * ghalf + g1)
    b = -3.0 * g0 + 4.0 * ghalf - g1
    c = g0

    scale = np.abs(b) + np.abs(c) + 1.0
    is_linear = np.abs(a) <= 1e-12 * scale

    with np.errstate(divide='ignore', invalid='ignore'):
        r_lin = np.where(np.abs(b) > 0, -c / b, np.inf)

    disc = b * b - 4.0 * a * c
    has_real = disc >= 0
    sqrt_disc = np.sqrt(np.where(has_real, disc, 0.0))

    q = -0.5 * (b + np.where(b >= 0, sqrt_disc, -sqrt_disc))
    with np.errstate(divide='ignore', invalid='ignore'):
        r1 = np.where(np.abs(a) > 0, q / a, np.inf)
        r2 = np.where(np.abs(q) > 0, c / q, np.inf)

    r1 = np.where(has_real, r1, np.inf)
    r2 = np.where(has_real, r2, np.inf)

    r1 = np.where(is_linear, r_lin, r1)
    r2 = np.where(is_linear, np.inf, r2)

    def select(r):
        return np.where((r > 0) & (r <= 1), r, np.inf)

    s = np.minimum(select(r1), select(r2))
    s = np.where(np.isfinite(s), s, default)
    return s


def reflect(ix, iy, iz, nx, ny, nz):
    """Specular reflection of an incident 3-vector about unit normal ``n``."""
    dot = ix * nx + iy * ny + iz * nz
    return (ix - 2.0 * dot * nx,
            iy - 2.0 * dot * ny,
            iz - 2.0 * dot * nz)
