"""The memory of the dense systems without stellarator symmetry, and a warning before one that may not fit.

Without stellarator symmetry torux holds two square systems as dense arrays of doubles: the Jacobian of the solve's
dense Newton (torux.torus_general.GeneralTorusProblem.newton) and that of the equations at any surface, which the
gradient's adjoint and answer.system_general build (torux.torus_general.GeneralEquations). Each is n x n for n
unknowns, n = 4 (M + 1)(2N + 1) - 4N (one less for the equations at any surface), so its memory grows as (M N)^2:
571 MB at M, N = 32, 32 and 5.0 GB at 48, 64.

Before either is built, torux warns (MemoryWarning) where the peak it estimates is more than half of the machine's
physical memory. The estimate counts the copies of the n x n matrix the code holds at once (COPIES) and the arrays
on its grid, and leaves out what the process held before.
"""
from __future__ import annotations

import os
import sys
import warnings

GB = 2 ** 30
# copies of the n x n Jacobian at the peak of each dense step, as the code allocates them (measured at n = 3,364 to
# 10,659 on QUASR 954 with one coil of each pair at 3% more current: 0.99 to 1.16 for the gradient, 4.54 to 4.57 for a
# Newton step):
#   gradient  the Jacobian alone: built in place (GeneralEquations._jacobian), factorised in place by LAPACK on its
#             transpose's storage, and freed before the product in the source
#   solve     the jitted system's Jacobian and the Galerkin blocks it is joined from, 2.55 together, then
#             jnp.linalg.solve's copy of it to factorise and that copy in LAPACK's layout, 2 more, beside what the
#             system held
#   filtered  the same with the SVD in place of the solve: U, V^T and LAPACK's copies and workspace, 6 more
COPIES = {"gradient": 1.0, "solve": 4.6, "filtered": 8.6}
# doubles a grid point holds while the system is built: the field, its gradient and its vector potential, their
# cylindrical components, the surface's geometry and the four coefficient functions with their FFTs
PER_POINT = 64


class MemoryWarning(UserWarning):
    """A dense system without stellarator symmetry whose estimated peak is more than half of the machine's memory."""


def physical_bytes():
    """The machine's physical memory in bytes, or None where it cannot be read."""
    try:
        n = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        if n > 0:
            return int(n)
    except (AttributeError, ValueError, OSError):
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            class _Status(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = _Status()
            st.dwLength = ctypes.sizeof(_Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return int(st.ullTotalPhys)
        except (AttributeError, OSError):
            pass
    return None


def unknowns(M, N, kind="gradient"):
    """n of the dense system at M, N: the masked cosine and sine coefficients of R and Z, then iota, and for the
    solve's Newton the unfolding parameter too (torux.torus_general)."""
    nco = 2 * (M + 1) * (2 * N + 1) - 2 * N - 1
    return 2 * nco + (1 if kind == "gradient" else 2)


def grid_points(M, N):
    """Points of the oversampled grid of one field period at M, N (torux.torus_general.GeneralBasis)."""
    from .torus import fft_size
    nt = fft_size(max(int(2.5 * M) + 4, 2 * M + 2, 16))
    nph = fft_size(max(int(2.5 * N) + 4, 2 * N + 2, 16)) if N > 0 else 1
    return nt * nph


def dense_bytes(M, N, kind="gradient"):
    """The estimated peak in bytes of a dense step without stellarator symmetry at M, N, beyond what the process
    held before it. kind: 'gradient' (the adjoint, answer.system_general), 'solve' (a dense Newton step) or
    'filtered' (a Newton step by the SVD)."""
    n = unknowns(M, N, kind)
    return int(COPIES[kind] * 8 * n * n) + PER_POINT * 8 * grid_points(M, N)


def warn_if_large(M, N, kind, what, stacklevel=3):
    """Warn (MemoryWarning) where dense_bytes(M, N, kind) is more than half of the machine's memory. True if it did."""
    total = physical_bytes()
    need = dense_bytes(M, N, kind)
    if not total or need <= total / 2:
        return False
    gb = lambda v: f"{v / GB:.2f}" if v < GB else f"{v / GB:.1f}"                      # noqa: E731
    warnings.warn(f"{what} without stellarator symmetry at M, N = {M}, {N} needs about {gb(need)} GB of memory, "
                  f"more than half of this machine's {gb(total)} GB. It grows as the square of M N",
                  MemoryWarning, stacklevel=stacklevel)
    return True
