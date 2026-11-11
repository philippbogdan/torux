"""A hand-written float64 CUDA kernel for the Fourier-filament sum (B, grad B, A and the half rule), compiled at run
time by NVRTC and launched through the CUDA driver API (ctypes only: no CUDA toolkit, CuPy or numba.cuda needed;
the libraries come with the CUDA wheels of jax / torch and the node's driver).

Same quadrature as torux.field.cpu.filament (the engine's trapezoid nodes; the half rule is twice the sum over the
even nodes), written independently of it:

* targets one per thread, sources streamed through shared memory in tiles of TILE nodes;
* within a tile plain float64 sums in registers, across tiles Neumaier-compensated sums (as the CPU kernels'
  blocks of 32), so a sum of S terms carries the rounding of a TILE-term sum;
* 1/|d| by the float64 rsqrt (max 1 ulp, no division);
* the trace of grad B: every Biot-Savart element's dB_a/dx_b is trace free exactly ((w x e_a)_a = 0 and
  (w x d).d = 0), so dB_z/dz = -(dB_x/dx + dB_y/dy) (one accumulator fewer);
* the source range is split over gridDim.y (enough blocks to fill the card at the solver's target counts); each
  range's compensated partial sum is written separately and the partials are added on the host in a fixed order
  (deterministic: no atomics).

Use: ``k = CudaFilament.get()`` (None when no usable card / driver / NVRTC), then ``k.upload(r, w, wh)`` once per
engine and ``k.evaluate(handle, x, need_grad, need_A)`` -> (T, 18) numpy array in the layout of
torux.field.cpu.filament (B 0:3, A 3:6, G 6:15 row-major dB_a/dx_b, Bh 15:18).

Experimental (torux.fast_family 'cudafil').
"""
from __future__ import annotations

import ctypes
import ctypes.util
import glob
import hashlib
import os
import sys
import threading

import numpy as np

TILE = 128

_SRC = r"""
#define TILE %(TILE)d
extern "C" __global__ void __launch_bounds__(TILE)
%(NAME)s(const double* __restrict__ X, const int T, const double* __restrict__ Sx, const int S,
         const int* __restrict__ ranges, double* __restrict__ out)
{
    __shared__ double sh[6 * TILE];
    const int i = blockIdx.x * TILE + threadIdx.x;
    const int ii = i < T ? i : T - 1;
    const double x0 = X[ii], x1 = X[T + ii], x2 = X[2 * T + ii];
    const int lo = ranges[2 * blockIdx.y], hi = ranges[2 * blockIdx.y + 1];
    double tot[%(NK)d], cmp[%(NK)d];
#pragma unroll
    for (int k = 0; k < %(NK)d; ++k) { tot[k] = 0.0; cmp[k] = 0.0; }
    for (int j0 = lo; j0 < hi; j0 += TILE) {
        const int n = min(TILE, hi - j0);
        __syncthreads();
        if (threadIdx.x < n) {
#pragma unroll
            for (int c = 0; c < 6; ++c) sh[c * TILE + threadIdx.x] = Sx[(size_t)c * S + j0 + threadIdx.x];
        }
        __syncthreads();
        double acc[%(NK)d];
#pragma unroll
        for (int k = 0; k < %(NK)d; ++k) acc[k] = 0.0;
        for (int q = 0; q < n; ++q) {
            const double d0 = x0 - sh[q], d1 = x1 - sh[TILE + q], d2 = x2 - sh[2 * TILE + q];
            const double w0 = sh[3 * TILE + q], w1 = sh[4 * TILE + q], w2 = sh[5 * TILE + q];
            const double ir = rsqrt(d0 * d0 + d1 * d1 + d2 * d2);
            const double ir3 = ir * ir * ir;
            const double c0 = w1 * d2 - w2 * d1, c1 = w2 * d0 - w0 * d2, c2 = w0 * d1 - w1 * d0;
            acc[0] += c0 * ir3; acc[1] += c1 * ir3; acc[2] += c2 * ir3;
#if %(NA)d
            acc[3] += w0 * ir; acc[4] += w1 * ir; acc[5] += w2 * ir;
#endif
#if %(NG)d
            const double f = 3.0 * ir3 * ir * ir;
            const double e0 = c0 * f, e1 = c1 * f, e2 = c2 * f;
            const double u0 = w0 * ir3, u1 = w1 * ir3, u2 = w2 * ir3;
            acc[%(G0)d + 0] -= e0 * d0;
            acc[%(G0)d + 1] += -u2 - e0 * d1;
            acc[%(G0)d + 2] += u1 - e0 * d2;
            acc[%(G0)d + 3] += u2 - e1 * d0;
            acc[%(G0)d + 4] -= e1 * d1;
            acc[%(G0)d + 5] += -u0 - e1 * d2;
            acc[%(G0)d + 6] += -u1 - e2 * d0;
            acc[%(G0)d + 7] += u0 - e2 * d1;
#endif
        }
#pragma unroll
        for (int k = 0; k < %(NK)d; ++k) {
            const double s = tot[k], v = acc[k], u = s + v;
            cmp[k] += (fabs(s) >= fabs(v)) ? ((s - u) + v) : ((v - u) + s);
            tot[k] = u;
        }
    }
    if (i < T) {
        double* o = out + ((size_t)blockIdx.y * T + i) * %(NK)d;
#pragma unroll
        for (int k = 0; k < %(NK)d; ++k) o[k] = tot[k] + cmp[k];
    }
}
"""

# accumulator layouts per kind: (need_grad, need_A) -> (NK, index of G00 or -1, has A)
_KINDS = {(True, True): (14, 6, 1), (True, False): (11, 3, 0), (False, True): (6, -1, 1), (False, False): (3, -1, 0)}


def _kname(g, a):
    return f"torux_fil_g{int(g)}_a{int(a)}"


def _source():
    parts = []
    for (g, a), (nk, g0, na) in _KINDS.items():
        parts.append(_SRC % dict(TILE=TILE, NAME=_kname(g, a), NK=nk, NA=na, NG=int(g), G0=max(g0, 0)))
    return "\n".join(parts)


def _find_lib(names, patterns, with_path=False):
    for n in names:
        try:
            lib = ctypes.CDLL(n)
            return (lib, None) if with_path else lib
        except OSError:
            pass
    roots = [p for p in sys.path if p and os.path.isdir(p)]
    for root in roots:
        for pat in patterns:
            for f in sorted(glob.glob(os.path.join(root, pat)), reverse=True):
                if "builtins" in os.path.basename(f):
                    continue
                try:
                    lib = ctypes.CDLL(f)
                    return (lib, f) if with_path else lib
                except OSError:
                    pass
    return (None, None) if with_path else None


class CudaError(RuntimeError):
    pass


class CudaFilament:
    """The kernel on device 0 of this process (one instance, created on first use)."""

    _inst = None
    _tried = False
    _lock = threading.Lock()
    why_not = None

    @classmethod
    def get(cls):
        with cls._lock:
            if not cls._tried:
                cls._tried = True
                try:
                    cls._inst = cls()
                except Exception as e:  # noqa: BLE001
                    cls._inst = None
                    cls.why_not = repr(e)
            return cls._inst

    # ------------------------------------------------------------------ set-up
    def __init__(self):
        cu = _find_lib(["libcuda.so.1", "libcuda.so"], [])
        if cu is None:
            raise CudaError("no CUDA driver (libcuda.so.1)")
        self.cu = cu
        self._chk(cu.cuInit(0), "cuInit")
        n = ctypes.c_int(0)
        self._chk(cu.cuDeviceGetCount(ctypes.byref(n)), "cuDeviceGetCount")
        if n.value < 1:
            raise CudaError("no CUDA device")
        dev = ctypes.c_int(0)
        self._chk(cu.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
        self.dev = dev
        ctx = ctypes.c_void_p()
        self._chk(cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "cuDevicePrimaryCtxRetain")
        self.ctx = ctx
        self._chk(cu.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
        maj, mnr = ctypes.c_int(), ctypes.c_int()
        self._chk(cu.cuDeviceGetAttribute(ctypes.byref(maj), 75, dev), "attr major")
        self._chk(cu.cuDeviceGetAttribute(ctypes.byref(mnr), 76, dev), "attr minor")
        nsm = ctypes.c_int()
        self._chk(cu.cuDeviceGetAttribute(ctypes.byref(nsm), 16, dev), "attr sm count")
        self.cc = (maj.value, mnr.value)
        self.n_sm = nsm.value
        name = ctypes.create_string_buffer(256)
        cu.cuDeviceGetName(name, 256, dev)
        self.name = name.value.decode(errors="replace")
        self.image, self.image_kind = self._compile()
        mod = ctypes.c_void_p()
        self._chk(cu.cuModuleLoadData(ctypes.byref(mod), ctypes.c_char_p(self.image)), "cuModuleLoadData")
        self.mod = mod
        self.fn = {}
        for key in _KINDS:
            f = ctypes.c_void_p()
            self._chk(cu.cuModuleGetFunction(ctypes.byref(f), mod, _kname(*key).encode()), "cuModuleGetFunction")
            self.fn[key] = f
        self._bufs = {}

    def _chk(self, rc, what):
        if rc != 0:
            s = ctypes.c_char_p()
            try:
                self.cu.cuGetErrorString(rc, ctypes.byref(s))
                msg = s.value.decode() if s.value else ""
            except Exception:  # noqa: BLE001
                msg = ""
            raise CudaError(f"{what}: CUDA error {rc} {msg}")

    def _compile(self):
        """NVRTC: a cubin for this card's architecture (no driver JIT), else PTX for the driver to JIT; cached
        on disk next to the jax cache (keyed by source, architecture and NVRTC version)."""
        nv, nv_path = _find_lib(["libnvrtc.so.12", "libnvrtc.so.13", "libnvrtc.so.11.2", "libnvrtc.so"],
                                ["nvidia/cuda_nvrtc/lib/libnvrtc.so*", "nvidia/cu13/lib/libnvrtc.so*",
                                 "nvidia/*/lib/libnvrtc.so*", "torch/lib/libnvrtc*.so*"], with_path=True)
        if nv is None:
            raise CudaError("no NVRTC library")
        self.nvrtc_path = nv_path
        if nv_path:
            # NVRTC opens its builtins library by name at compile time: load the copy next to it first
            for f in sorted(glob.glob(os.path.join(os.path.dirname(nv_path), "libnvrtc-builtins*.so*"))):
                try:
                    ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
        maj, mnr = ctypes.c_int(), ctypes.c_int()
        nv.nvrtcVersion(ctypes.byref(maj), ctypes.byref(mnr))
        src = _source().encode()
        arch = f"{self.cc[0]}{self.cc[1]}"
        key = hashlib.sha256(src + arch.encode() + f"{maj.value}.{mnr.value}".encode()).hexdigest()[:16]
        cdir = os.environ.get("TORUX_CUDA_CACHE") or os.path.join(os.getcwd(), "cache", "cuda")
        for kind in ("cubin", "ptx"):
            fn = os.path.join(cdir, f"fil_{key}.{kind}")
            if os.path.exists(fn):
                try:
                    return open(fn, "rb").read(), kind
                except OSError:
                    pass
        out = None
        self.compile_log = ""
        for kind, opt in (("cubin", f"--gpu-architecture=sm_{arch}"), ("ptx", f"--gpu-architecture=compute_{arch}")):
            if kind == "cubin" and not hasattr(nv, "nvrtcGetCUBIN"):
                continue
            prog = ctypes.c_void_p()
            rc = nv.nvrtcCreateProgram(ctypes.byref(prog), ctypes.c_char_p(src), b"torux_fil.cu", 0, None, None)
            if rc != 0:
                raise CudaError(f"nvrtcCreateProgram {rc}")
            try:
                opts = (ctypes.c_char_p * 3)(opt.encode(), b"--std=c++14", b"--fmad=true")
                rc = nv.nvrtcCompileProgram(prog, 3, opts)
                if rc != 0:
                    sz = ctypes.c_size_t()
                    nv.nvrtcGetProgramLogSize(prog, ctypes.byref(sz))
                    log = ctypes.create_string_buffer(sz.value + 1)
                    nv.nvrtcGetProgramLog(prog, log)
                    self.compile_log += f"[{kind} rc {rc}] " + log.value.decode(errors="replace")
                    continue
                sz = ctypes.c_size_t()
                if kind == "cubin":
                    nv.nvrtcGetCUBINSize(prog, ctypes.byref(sz))
                    buf = ctypes.create_string_buffer(sz.value)
                    nv.nvrtcGetCUBIN(prog, buf)
                else:
                    nv.nvrtcGetPTXSize(prog, ctypes.byref(sz))
                    buf = ctypes.create_string_buffer(sz.value)
                    nv.nvrtcGetPTX(prog, buf)
                out = (buf.raw, kind)
                break
            finally:
                nv.nvrtcDestroyProgram(ctypes.byref(prog))
        if out is None:
            raise CudaError("NVRTC compile failed: " + self.compile_log[:2000])
        try:
            os.makedirs(cdir, exist_ok=True)
            tmp = os.path.join(cdir, f".fil_{key}.{out[1]}.{os.getpid()}")
            open(tmp, "wb").write(out[0])
            os.replace(tmp, os.path.join(cdir, f"fil_{key}.{out[1]}"))
        except OSError:
            pass
        return out

    # ------------------------------------------------------------------ memory
    def _alloc(self, nbytes):
        p = ctypes.c_uint64()
        self._chk(self.cu.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t(max(int(nbytes), 8))), "cuMemAlloc")
        return p

    def _free(self, p):
        try:
            self.cu.cuMemFree_v2(p)
        except Exception:  # noqa: BLE001
            pass

    def _h2d(self, p, a):
        a = np.ascontiguousarray(a)
        self._chk(self.cu.cuMemcpyHtoD_v2(p, a.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(a.nbytes)), "HtoD")

    def _d2h(self, a, p):
        self._chk(self.cu.cuMemcpyDtoH_v2(a.ctypes.data_as(ctypes.c_void_p), p, ctypes.c_size_t(a.nbytes)), "DtoH")

    def _scratch(self, tag, nbytes):
        b = self._bufs.get(tag)
        if b is None or b[1] < nbytes:
            if b is not None:
                self._free(b[0])
            b = (self._alloc(int(nbytes * 1.25) + 64), int(nbytes * 1.25) + 64)
            self._bufs[tag] = b
        return b[0]

    # ------------------------------------------------------------------ sources
    def upload(self, r, w, wh):
        """The engine's nodes (r, w as (S, 3), wh the half-rule factors) on the card, reordered even (wh == 2)
        first, then odd (wh == 0).  Returns a handle, or None when wh is not the 2 / 0 pattern (the CPU kernel
        then serves)."""
        r = np.asarray(r, float); w = np.asarray(w, float); wh = np.asarray(wh, float)
        ev = np.flatnonzero(wh == 2.0); od = np.flatnonzero(wh == 0.0)
        if len(ev) + len(od) != len(wh) or len(wh) == 0:
            return None
        idx = np.concatenate([ev, od])
        Sx = np.ascontiguousarray(np.concatenate([r[idx].T, w[idx].T], 0))      # (6, S)
        p = self._alloc(Sx.nbytes)
        self._h2d(p, Sx)
        return dict(ptr=p, S=len(idx), Se=len(ev), nbytes=Sx.nbytes, owner=self)

    def release(self, h):
        if h is not None and h.get("ptr") is not None:
            self._free(h["ptr"])
            h["ptr"] = None

    def _ranges(self, S, Se, T):
        """Source ranges (within the even half, then the odd half) so that blocks x ranges fill the card."""
        nb = max(1, (T + TILE - 1) // TILE)
        want = max(1, int(np.ceil(8 * self.n_sm / nb)))          # ~8 blocks per SM
        per_half = max(1, min(want // 2 + (want % 2), 64))
        out = []
        for lo, hi in ((0, Se), (Se, S)):
            n = hi - lo
            k = max(1, min(per_half, (n + TILE - 1) // TILE))
            edges = lo + (np.arange(k + 1) * n) // k
            for a, b in zip(edges[:-1], edges[1:]):
                out.append((int(a), int(b), lo == 0))
        return out

    def evaluate(self, h, x, need_grad=True, need_A=True):
        """(T, 18): B, A, G (row-major dB_a/dx_b), Bh at the targets x (T, 3)."""
        x = np.asarray(x, float).reshape(-1, 3)
        T = x.shape[0]
        out = np.zeros((T, 18))
        if T == 0:
            return out
        key = (bool(need_grad), bool(need_A))
        nk, g0, na = _KINDS[key]
        rng = self._ranges(h["S"], h["Se"], T)
        R = len(rng)
        xs = np.ascontiguousarray(x.T)
        px = self._scratch("x", xs.nbytes)
        self._h2d(px, xs)
        ra = np.array([[a, b] for a, b, _ in rng], np.int32).ravel()
        pr = self._scratch("r", ra.nbytes)
        self._h2d(pr, ra)
        po = self._scratch("o", R * T * nk * 8)
        cT, cS = ctypes.c_int(T), ctypes.c_int(h["S"])
        args = [ctypes.c_uint64(px.value), cT, ctypes.c_uint64(h["ptr"].value), cS, ctypes.c_uint64(pr.value),
                ctypes.c_uint64(po.value)]
        params = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
        nb = (T + TILE - 1) // TILE
        self._chk(self.cu.cuLaunchKernel(self.fn[key], nb, R, 1, TILE, 1, 1, 0, None, params, None), "launch")
        self._chk(self.cu.cuCtxSynchronize(), "sync")
        part = np.empty((R, T, nk))
        self._d2h(part, po)
        even = np.array([e for _, _, e in rng])
        # fixed-order host sums of the range partials (few terms; the ranges' own sums are compensated)
        Pe = part[even].sum(0) if even.any() else np.zeros((T, nk))
        Po = part[~even].sum(0) if (~even).any() else np.zeros((T, nk))
        tot = Pe + Po
        out[:, 0:3] = tot[:, 0:3]
        out[:, 15:18] = 2.0 * Pe[:, 0:3]
        if na:
            out[:, 3:6] = tot[:, 3:6]
        if need_grad:
            out[:, 6:14] = tot[:, g0:g0 + 8]
            out[:, 14] = -(out[:, 6] + out[:, 10])
        return out
