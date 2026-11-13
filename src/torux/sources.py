"""Every way a physicist holds coils, as a source.

    torux.load("quasr:954")       QUASR design 954, downloaded once into torux's cache and read from there after
    torux.load(biot_savart)       a live SIMSOPT BiotSavart, a Coil or a list of Coils
    torux.load(coil_set)          a live DESC CoilSet, MixedCoilSet or single coil
    torux.load("coils.json")      a file, read as torux.load.load reads it

A live object is written by its own library's serialisation (SIMSOPT's JSON, DESC's HDF5) into a temporary
file, and that file is read by the reader the package already has for it, so the conversion is the tested one
and what it cannot convert exactly it says.  Everything else goes to torux.load.load unchanged.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time

from .load import load as _base_load

QUASR_URL = "https://quasr.flatironinstitute.org/simsopt_serials/{group:04d}/serial{n:07d}.json"
QUASR_SITE = "https://quasr.flatironinstitute.org"

# "quasr:954", "QUASR 954", "quasr_0000954", "quasr/1,100,073", "quasr:serial0000954.json"
_QUASR = re.compile(r"^\s*quasr\s*[:#/_ -]?\s*(?:serial)?\s*([0-9][0-9,_ ]*?)\s*(?:\.json)?\s*$", re.IGNORECASE)
_GROUPED = re.compile(r"^[0-9]{1,3}(?:([,_ ])[0-9]{3})?(?:\1[0-9]{3})*$")


def load(path, *more, nfp=None, kind=None, flux=None, which=None, name=None, currents=None):
    """The source of a field: a coil or magnet file, a QUASR design ("quasr:954"), a live SIMSOPT BiotSavart, Coil
    or list of Coils, a live DESC CoilSet, MixedCoilSet or coil, or several of these whose parts add up.

    nfp    field periods, for formats that do not state them, and the symmetry to verify
    kind   'stellarator' (default), 'tokamak' or 'dipole'
    flux   a full toroidal flux in Wb to ask for, kept with the source as its request
    which  for a SIMSOPT file holding several coil sets: the one to take, by name or index
    name   the source's id (default: the first file's name)
    currents  for a SIMSOPT file whose Current objects state two different values: 'saved', the value each object
           held when the file was saved, or 'dofs', the value SIMSOPT reads back. Without it such a file is refused

    A DESC coil set's field comes out larger than DESC's own by 1.3 parts in 10^10: torux takes mu_0 = 4 pi 1e-7
    H/m, as SIMSOPT and VMEC do, and DESC takes CODATA's."""
    first, made = _as_file(path)
    rest = [_as_file(p) for p in more]
    temps = [t for t in [made] + [m for _, m in rest] if t is not None]
    try:
        if name is None and made is not None:
            name = made["name"]
        src = _base_load(first, *[f for f, _ in rest], nfp=nfp, kind=kind, flux=flux, which=which,
                                name=name, currents=currents)
        files = src.provenance.get("files")
        if isinstance(files, list):
            # a temporary file says nothing once it is gone: name the object it was written from instead
            by_path = {t["path"]: t for t in temps if t.get("temporary")}
            src.provenance["files"] = [by_path[f]["what"] if f in by_path else f for f in files]
        for t in temps:
            if "quasr" in t:
                src.provenance["quasr"] = t["quasr"]
        return src
    except ImportError as e:
        lib = getattr(e, "name", None)
        if lib in ("h5py", "netCDF4"):
            raise ImportError(f'reading {first} needs h5py: pip install ".[hdf5]" in the torux folder') from e
        raise
    except NotImplementedError as e:
        live = [t for t in temps if t.get("temporary")]
        if live:
            raise ValueError(f"{live[0]['what']} holds a part torux cannot convert exactly ({e}). "
                             f"{live[0]['hint']}") from e
        raise
    finally:
        for t in temps:
            if t.get("temporary"):
                shutil.rmtree(t["dir"], ignore_errors=True)


# --------------------------------------------------------------------------- what a source argument is
def quasr_number(spec):
    """The design number a string such as "quasr:954" names, or None when the string does not name one."""
    if not isinstance(spec, str):
        return None
    m = _QUASR.match(spec)
    if not m:
        if re.match(r"^\s*quasr\s*[:#/ -]", spec, re.IGNORECASE):
            raise ValueError(f"{spec!r}: write a QUASR design's number as digits, for example 'quasr:954'")
        return None
    digits = m.group(1).strip()
    if not _GROUPED.match(digits) and not digits.isdigit():
        raise ValueError(f"{spec!r}: write a QUASR design's number as digits, for example 'quasr:954'")
    n = int(re.sub(r"[,_ ]", "", digits))
    if not 1 <= n <= 9_999_999:
        raise ValueError(f"{spec!r}: QUASR numbers its designs from 1 to 9,999,999")
    return n


def _as_file(x):
    """(a path torux.load.load reads, and what was made for it or None)."""
    if isinstance(x, (str, os.PathLike)):
        if isinstance(x, str) and not os.path.exists(x):
            n = quasr_number(x)
            if n is not None:
                path = quasr_file(n)
                return path, dict(path=path, name=f"quasr_{n:07d}",
                                  quasr=dict(id=n, url=QUASR_URL.format(group=n // 1000, n=n)))
        if not os.path.exists(x):
            raise FileNotFoundError(f"{os.fspath(x)}: no such file: give the path of a coil or magnet file, or "
                                    f"quasr:NUMBER for a QUASR design")
        if os.path.isdir(x):
            raise IsADirectoryError(f"{os.fspath(x)} is a folder: give a file in it")
        return x, None
    mod = type(x).__module__ or ""
    if isinstance(x, (list, tuple)) and x:
        mod = type(x[0]).__module__ or ""
    if mod.split(".")[0] == "simsopt":
        return _simsopt_file(x)
    if mod.split(".")[0] == "desc":
        return _desc_file(x)
    raise TypeError(f"torux cannot read a source of type {type(x).__name__}: give a file's path, "
                    f"'quasr:NUMBER', a SIMSOPT BiotSavart, Coil or list of Coils, or a DESC CoilSet, MixedCoilSet or "
                    f"coil")


# --------------------------------------------------------------------------- QUASR
def quasr_file(n):
    """The path of QUASR design ``n`` in torux's cache, downloaded first if it is not there yet."""
    from . import _cache_dir
    n = int(n)
    path = os.path.join(_cache_dir(), "quasr", f"serial{n:07d}.json")
    if os.path.exists(path):
        return path
    url = QUASR_URL.format(group=n // 1000, n=n)
    try:
        data = _download(url)
    except _NotThere:
        raise ValueError(f"QUASR has no design {n:,} ({url} does not exist): check the number at "
                         f"{QUASR_SITE}") from None
    except OSError as e:
        raise ConnectionError(f"QUASR design {n:,} is not in torux's cache and could not be downloaded ({e}): "
                              f"download {url} by hand and save it as {path}") from e
    try:
        ok = "simsopt_objs" in json.loads(data)
    except ValueError:
        ok = False
    if not ok:
        raise ConnectionError(f"{url} did not send a QUASR design: download it by hand and save it as {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".download-")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)          # whole or not at all, even with several processes downloading at once
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


class _NotThere(Exception):
    pass


def _download(url, tries=3):
    """The bytes at ``url``. _NotThere for a page that does not exist, OSError when the network fails."""
    import urllib.error
    import urllib.request
    from . import __version__
    req = urllib.request.Request(url, headers={"User-Agent": f"torux/{__version__}"})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404, 410):
                raise _NotThere(url) from None
            err = e
        except (urllib.error.URLError, OSError) as e:
            err = e
            if isinstance(e, urllib.error.URLError) and not isinstance(getattr(e, "reason", None), TimeoutError):
                break                  # no route or no name: trying again will not help
        if i + 1 < tries:
            time.sleep(2 ** i)
    raise OSError(getattr(err, "reason", None) or err)


# --------------------------------------------------------------------------- live objects
def _temp(what, name, suffix, hint):
    d = tempfile.mkdtemp(prefix="torux-")
    return dict(dir=d, path=os.path.join(d, "source" + suffix), name=name, what=what, hint=hint, temporary=True)


def _simsopt_file(x):
    try:
        import simsopt
        from simsopt.field import BiotSavart, Coil
    except ImportError as e:  # pragma: no cover
        raise ImportError(f"reading a SIMSOPT object needs SIMSOPT itself: pip install simsopt ({e})") from e
    if isinstance(x, BiotSavart):
        obj, what, name = x, "a SIMSOPT BiotSavart object", "biotsavart"
    elif isinstance(x, Coil):
        obj, what, name = [x], "a SIMSOPT Coil object", "coil"
    elif isinstance(x, (list, tuple)) and all(isinstance(c, Coil) for c in x):
        obj, what, name = list(x), f"a list of {len(x)} SIMSOPT Coil objects", "coils"
    else:
        raise ValueError(f"torux reads SIMSOPT's BiotSavart, a Coil or a list of Coils. This is a "
                         f"{type(x).__name__}")
    t = _temp(what, name, ".json", "SIMSOPT can save it with simsopt.save(obj, 'coils.json') for a look at "
                                   "what it holds")
    try:
        simsopt.save(obj, t["path"])
    except BaseException:
        shutil.rmtree(t["dir"], ignore_errors=True)
        raise
    return t["path"], t


def _desc_file(x):
    try:
        from desc.coils import CoilSet, _Coil
    except ImportError as e:  # pragma: no cover
        raise ImportError(f"reading a DESC object needs DESC itself: pip install desc-opt ({e})") from e
    if isinstance(x, (list, tuple)):
        raise ValueError("give DESC coils as one coil set: desc.coils.MixedCoilSet(*coils)")
    if isinstance(x, CoilSet):
        what = f"a DESC {type(x).__name__}"
    elif isinstance(x, _Coil):
        what = f"a DESC {type(x).__name__}"
    else:
        raise ValueError(f"torux reads DESC's CoilSet, MixedCoilSet or a single coil. This is a {type(x).__name__}")
    try:
        import h5py  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(f'reading a DESC coil set needs h5py: pip install ".[hdf5]" in the torux folder ({e})') from e
    name = (getattr(x, "name", "") or "").strip() or "desc_coils"
    t = _temp(what, name, ".h5", "torux reads DESC's FourierPlanarCoil, FourierRZCoil and FourierXYZCoil exactly: "
                                 "convert the others with DESC's to_FourierXYZ() first, an approximation DESC "
                                 "makes and you check")
    try:
        x.save(t["path"])
    except BaseException:
        shutil.rmtree(t["dir"], ignore_errors=True)
        raise
    return t["path"], t
