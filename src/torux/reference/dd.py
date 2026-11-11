"""Double-double (dd) arithmetic for numba, with no FMA dependence.

A dd number is stored in a ``complex128``: ``real`` = high part, ``imag`` = low
part, value = hi + lo with |lo| <= ulp(hi)/2.  This keeps numba code compact
(one value per number, arrays of dd are complex arrays).  NEVER use the native
complex operators on these values: use the functions below.

Error-free transformations: Knuth TwoSum, Dekker/Veltkamp split + TwoProd
(valid on hardware without FMA; numba compiles without fastmath, so LLVM
does not contract a*b+c).  Accuracy of each operation ~ 2^-104 relative
(add/mul/div/sqrt); the elementary functions (exp, log, sin/cos of 2 pi t,
atanh) ~ 1e-31 relative; the tests check every function against mpmath at 50 digits.
"""
from __future__ import annotations

import math

import mpmath
import numpy as np
from numba import njit

_SPLIT = 134217729.0  # 2^27 + 1

# ------------------------------------------------------------------ constants from mpmath (40 digits)


def mp_to_dd(x):
    """mpmath number -> (hi, lo) floats with hi + lo = x to ~1e-32 relative."""
    with mpmath.workdps(60):
        x = mpmath.mpf(x)
        hi = float(x)
        lo = float(x - mpmath.mpf(hi))
    return hi, lo


def dd_to_mp(z):
    """complex128 dd (or (hi, lo)) -> mpf (exact sum)."""
    if isinstance(z, (complex, np.complexfloating)):
        return mpmath.mpf(float(z.real)) + mpmath.mpf(float(z.imag))
    return mpmath.mpf(float(z[0])) + mpmath.mpf(float(z[1]))


def mp_to_c(x):
    hi, lo = mp_to_dd(x)
    return complex(hi, lo)


with mpmath.workdps(60):
    _PI = mp_to_c(mpmath.pi)
    _TWO_PI = mp_to_c(2 * mpmath.pi)
    _LN2 = mp_to_c(mpmath.log(2))
    _SQRT_HALF = mp_to_c(mpmath.sqrt(2) / 2)

PI_HI, PI_LO = _PI.real, _PI.imag
TWO_PI_HI, TWO_PI_LO = _TWO_PI.real, _TWO_PI.imag
LN2_HI, LN2_LO = _LN2.real, _LN2.imag
SQH_HI, SQH_LO = _SQRT_HALF.real, _SQRT_HALF.imag


# ------------------------------------------------------------------ error-free transformations
@njit(cache=True, error_model="numpy")
def two_sum(a, b):
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return s, e


@njit(cache=True, error_model="numpy")
def quick_two_sum(a, b):
    s = a + b
    e = b - (s - a)
    return s, e


@njit(cache=True, error_model="numpy")
def split(a):
    t = _SPLIT * a
    hi = t - (t - a)
    lo = a - hi
    return hi, lo


@njit(cache=True, error_model="numpy")
def two_prod(a, b):
    p = a * b
    ah, al = split(a)
    bh, bl = split(b)
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e


# ------------------------------------------------------------------ dd arithmetic (complex128 carrier)
@njit(cache=True, error_model="numpy")
def D(x):
    """float -> dd."""
    return complex(x, 0.0)


@njit(cache=True, error_model="numpy")
def hi(x):
    return x.real


@njit(cache=True, error_model="numpy")
def add(x, y):
    s, e = two_sum(x.real, y.real)
    t, f = two_sum(x.imag, y.imag)
    e += t
    s, e = quick_two_sum(s, e)
    e += f
    s, e = quick_two_sum(s, e)
    return complex(s, e)


@njit(cache=True, error_model="numpy")
def neg(x):
    return complex(-x.real, -x.imag)


@njit(cache=True, error_model="numpy")
def sub(x, y):
    return add(x, complex(-y.real, -y.imag))


@njit(cache=True, error_model="numpy")
def add_d(x, d):
    s, e = two_sum(x.real, d)
    e += x.imag
    s, e = quick_two_sum(s, e)
    return complex(s, e)


@njit(cache=True, error_model="numpy")
def mul(x, y):
    p, e = two_prod(x.real, y.real)
    e += x.real * y.imag + x.imag * y.real
    p, e = quick_two_sum(p, e)
    return complex(p, e)


@njit(cache=True, error_model="numpy")
def mul_d(x, d):
    p, e = two_prod(x.real, d)
    e += x.imag * d
    p, e = quick_two_sum(p, e)
    return complex(p, e)


@njit(cache=True, error_model="numpy")
def mul_dd(a, b):
    """float * float -> exact dd."""
    p, e = two_prod(a, b)
    return complex(p, e)


@njit(cache=True, error_model="numpy")
def sqr(x):
    p, e = two_prod(x.real, x.real)
    e += 2.0 * x.real * x.imag
    p, e = quick_two_sum(p, e)
    return complex(p, e)


@njit(cache=True, error_model="numpy")
def mul_pow2(x, d):
    """exact multiplication by a power of two."""
    return complex(x.real * d, x.imag * d)


@njit(cache=True, error_model="numpy")
def div(x, y):
    q1 = x.real / y.real
    r = sub(x, mul_d(y, q1))
    q2 = r.real / y.real
    r = sub(r, mul_d(y, q2))
    q3 = r.real / y.real
    q1, q2 = quick_two_sum(q1, q2)
    return add(complex(q1, q2), complex(q3, 0.0))


@njit(cache=True, error_model="numpy")
def div_d(x, d):
    q1 = x.real / d
    p, e = two_prod(q1, d)
    s, f = two_sum(x.real, -p)
    f -= e
    f += x.imag
    q2 = (s + f) / d
    s, e = quick_two_sum(q1, q2)
    return complex(s, e)


@njit(cache=True, error_model="numpy")
def recip(x):
    return div(complex(1.0, 0.0), x)


@njit(cache=True, error_model="numpy")
def sqrt(x):
    if x.real <= 0.0:
        if x.real == 0.0:
            return complex(0.0, 0.0)
        return complex(np.nan, np.nan)
    s0 = math.sqrt(x.real)
    p, e = two_prod(s0, s0)
    r = ((x.real - p) - e) + x.imag
    s, c = quick_two_sum(s0, r / (2.0 * s0))
    return complex(s, c)


@njit(cache=True, error_model="numpy")
def absd(x):
    if x.real < 0.0 or (x.real == 0.0 and x.imag < 0.0):
        return complex(-x.real, -x.imag)
    return x


@njit(cache=True, error_model="numpy")
def is_neg(x):
    return x.real < 0.0 or (x.real == 0.0 and x.imag < 0.0)


@njit(cache=True, error_model="numpy")
def to_f(x):
    return x.real + x.imag


@njit(cache=True, error_model="numpy")
def fma3(x, y, z):
    """x*y + z in dd."""
    return add(mul(x, y), z)


# ------------------------------------------------------------------ elementary functions
@njit(cache=True, error_model="numpy")
def exp(a):
    if a.real > 709.0:
        return complex(np.inf, 0.0)
    if a.real < -745.0:
        return complex(0.0, 0.0)
    k = np.floor(a.real / LN2_HI + 0.5)
    r = sub(a, mul_d(complex(LN2_HI, LN2_LO), k))
    r = mul_pow2(r, 1.0 / 32.0)
    # expm1(r) by Taylor, |r| <= 0.011
    s = r
    p = r
    for i in range(2, 16):
        p = div_d(mul(p, r), float(i))
        s = add(s, p)
        if abs(p.real) < 1e-36 * abs(s.real):
            break
    for i in range(5):  # expm1(2x) = 2 expm1(x) + expm1(x)^2
        s = add(mul_pow2(s, 2.0), sqr(s))
    s = add_d(s, 1.0)
    f = math.ldexp(1.0, int(k))
    return complex(s.real * f, s.imag * f)


@njit(cache=True, error_model="numpy")
def log(a):
    if a.real <= 0.0:
        return complex(np.nan, np.nan)
    y = complex(math.log(a.real), 0.0)
    # one Newton step on exp(y) = a:  y <- y + a exp(-y) - 1
    y = add(y, add_d(mul(a, exp(neg(y))), -1.0))
    return y


@njit(cache=True, error_model="numpy")
def atanh_small(x):
    """atanh(x) by its series; intended for |x| <= 0.05 (13 terms reach 1e-33)."""
    x2 = sqr(x)
    p = x
    s = x
    for k in range(1, 40):
        p = mul(p, x2)
        t = div_d(p, 2.0 * k + 1.0)
        s = add(s, t)
        if abs(t.real) < 1e-34 * abs(s.real):
            break
    return s


@njit(cache=True, error_model="numpy")
def atanh(x):
    if abs(x.real) <= 0.05:
        return atanh_small(x)
    # 0.5 * log((1+x)/(1-x))
    return mul_pow2(log(div(add_d(x, 1.0), sub(complex(1.0, 0.0), x))), 0.5)


@njit(cache=True, error_model="numpy")
def _sin_cos_taylor(th):
    """sin, cos for |th| <= 0.4 (Taylor, 1e-33)."""
    t2 = sqr(th)
    # sin
    s = th
    p = th
    for k in range(1, 20):
        p = div_d(mul(p, t2), -(2.0 * k) * (2.0 * k + 1.0))
        s = add(s, p)
        if abs(p.real) < 1e-36:
            break
    c = complex(1.0, 0.0)
    p = complex(1.0, 0.0)
    for k in range(1, 20):
        p = div_d(mul(p, t2), -(2.0 * k - 1.0) * (2.0 * k))
        c = add(c, p)
        if abs(p.real) < 1e-36:
            break
    return s, c


@njit(cache=True, error_model="numpy")
def cossin2pi(t):
    """(cos 2 pi t, sin 2 pi t) for a dd t."""
    k = np.floor(t.real * 8.0 + 0.5)
    u = add_d(t, -k / 8.0)        # exact: |u| <= 1/16
    th = mul(u, complex(TWO_PI_HI, TWO_PI_LO))
    s, c = _sin_cos_taylor(th)
    m = int(k) % 8
    if m < 0:
        m += 8
    if m == 0:
        return c, s
    if m == 2:
        return neg(s), c
    if m == 4:
        return neg(c), neg(s)
    if m == 6:
        return s, neg(c)
    h = complex(SQH_HI, SQH_LO)
    cm = mul(sub(c, s), h)     # cos(th + pi/4)
    sm = mul(add(s, c), h)     # sin(th + pi/4)
    if m == 1:
        return cm, sm
    if m == 3:
        return neg(sm), cm
    if m == 5:
        return neg(cm), neg(sm)
    return sm, neg(cm)   # m == 7


# ------------------------------------------------------------------ vectors (tuples of 3 dd)
@njit(cache=True, error_model="numpy")
def dot3(ax, ay, az, bx, by, bz):
    return add(add(mul(ax, bx), mul(ay, by)), mul(az, bz))


@njit(cache=True, error_model="numpy")
def cross3(ax, ay, az, bx, by, bz):
    return (sub(mul(ay, bz), mul(az, by)),
            sub(mul(az, bx), mul(ax, bz)),
            sub(mul(ax, by), mul(ay, bx)))


# ------------------------------------------------------------------ numpy helpers (outside numba)
def to_float(z):
    """complex dd array -> float64 array (correctly rounded hi+lo)."""
    z = np.asarray(z)
    return z.real + z.imag


def from_float(x):
    return np.asarray(x, dtype=np.float64).astype(np.complex128)
