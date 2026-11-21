"""The command line.

    torux solve SOURCE [SOURCE ...] [--flux WB | --minor-radius M | --volume M3 | --largest] [--budget S] [--nfp N]
                [--kind K] [--gradients] [--no-verify] [--trace-lines N] [--save FILE ...] [--mpol N] [--ntor N]
                [--plot FILE] [--boozer] [--coils NAME] [--currents saved|dofs] [--widths S [--widths-jobs N]]
    torux check SOURCE [SOURCE ...] SURFACE [--flux WB] [--nfp N] [--which NAME] [--coils NAME]
                [--currents saved|dofs] [--surface-currents saved|dofs] [--mismatched-coils]
    torux many SOURCE [SOURCE ...] --table FILE [--flux WB | --minor-radius M | --volume M3 | --largest] [--budget S]
               [--jobs N] [--save DIR] [--coils NAME] [--currents saved|dofs] [--widths S [--widths-jobs N]]
    torux render SOURCE [SOURCE ...] FILE [--flux WB | --minor-radius M | --volume M3 | --largest] [--budget S]
                 [--nfp N] [--kind K] [--dark] [--closed] [--coils NAME] [--currents saved|dofs]
    torux poincare SOURCE [SOURCE ...] [FILE] [--planes DEG [DEG ...]] [--lines N] [--turns N] [--start R,Z]
                   [--dark] [--nfp N] [--kind K] [--coils NAME] [--currents saved|dofs]
    torux version

solve prints the answer on stdout and, with --save, writes it: *.json (everything), input.<name> (VMEC input, cut
to --mpol and --ntor when they are given), *.vtk (the surface with |B|). --save may be given more than once.  A
search (a size, the largest) says on stderr what it asks, how long it may take and each request as it ends.  The exit
code is 0 for a verified surface, a certified or proved no-surface or a verified range, 2 for anything undetermined,
1 for an error, a command line that cannot be read included.  render solves as solve does, prints the answer, and
writes its picture (answer.render), with the same exit code, or 1 where no picture could be written.
--coils and --currents are torux.load's which= and currents= for a SIMSOPT JSON source: which coil set of a file
holding several (with several files, --coils once per file, - for one that needs none), and which of a Current
object's two values where they differ.  In many, given once they apply to every design, and given once per SOURCE,
in order, each to its own design, - for one that needs none.  poincare prints its lines (where each starts, its
rotational transform, whether it left) and with FILE draws them, and where the lines take more than half a minute
says on stderr about how long.  --start is given once per line.  check's --which is the surface's, and
--surface-currents and --mismatched-coils are check's surface_currents= and mismatched_coils=True.  render and
poincare check that FILE can be written before any work, and Ctrl-C ends them in one line, with code 130.  Each
subcommand is a pair of functions in COMMANDS.
"""
from __future__ import annotations

import argparse
import sys


def _rows(rows):
    from .answer import _table
    return _table(rows)


def report(res, request=None):
    """The outcome of a solve as a few aligned lines (an Answer's own words)."""
    from .answer import Answer
    if isinstance(res, Answer):
        return str(res)
    ev = res.evidence or {}
    t = f"{res.wall_s:.1f} s" if res.wall_s < 100 else f"{res.wall_s:.0f} s"
    if res.status.startswith(("surface", "range")) and res.surface is not None:
        ind = ev.get("independent") or {}
        sol = ev.get("solver") or {}
        s = res.surface
        head = {"surface": "verified", "range": "the outermost verified surface the source holds"}.get(
            res.status, str(ev.get("unverified", "not verified")))
        rows = [(res.status.replace("_", " "), head),
                ("toroidal flux", f"{abs(s.flux):.12g} Wb"),
                ("rotational transform", f"{res.iota:.12g}"),
                ("Fourier series", f"M, N = {s.M}, {s.N}")]
        if sol.get("bn_off") is not None:
            rows.append(("field across it", f"{sol['bn_off']:.1e} of |B|"))
        chk = ind.get("bn_sup", ind.get("bn_max"))
        if chk is not None:
            rows.append(("independent check", f"{chk:.1e} of |B|, flux within {ind.get('flux_rel_err', float('nan')):.0e}"))
        rows.append(("time", t))
        return _rows(rows)
    if res.status == "no_surface":
        c = ev.get("certificate") or {}
        p = ev.get("proof") or {}
        kind = {"edge_barrier": "edge barrier", "island_band": "island band"}.get(
            str(c.get("kind", "")).replace("_manifold", "").replace("_pairs", ""), None)
        rows = [("no surface", f"certified ({kind})" if kind else "proved (" + str(p.get("verdict", "")).replace("_", " ").lower() + ")")]
        if request is not None:
            rows.append(("requested flux", f"{abs(request):.9g} Wb"))
        if c.get("F_up") is not None:
            rows.append(("largest possible", f"{c['F_up']:.9g} Wb"))
        elif c.get("interval"):
            rows.append(("none between", f"{c['interval'][0]:.9g} and {c['interval'][1]:.9g} Wb"))
        if p.get("arguments"):
            rows.append(("arguments", ", ".join(p["arguments"])))
        rows.append(("time", t))
        return _rows(rows)
    why = ev.get("unverified") or ev.get("error") or ev.get("status_at_deadline") or \
        "neither a surface nor a certificate within the budget"
    return _rows([(res.status, str(why)), ("time", t)])


def _solve_args(sub):
    s = sub.add_parser("solve", help="the flux surface asked for, verified, or the certificate that none exists",
                       description="The flux surface of the source asked for by one of --flux, --minor-radius, "
                                   "--volume or --largest. With none, the source's own request, or its largest "
                                   "surface when it carries none.")
    s.add_argument("files", nargs="+", metavar="SOURCE",
                   help="a coil or magnet file, or quasr:NUMBER. Several files whose parts add up are one source")
    _question_args(s)
    s.add_argument("--gradients", action="store_true", help="also the gradients of iota in every source parameter")
    widths_args(s)
    s.add_argument("--no-verify", action="store_true", help="skip the independent verification: the answer then rests "
                                                            "on the solver's own checks")
    s.add_argument("--trace-lines", type=int, default=0, metavar="N", help="also follow this many field lines with "
                                                                           "the independent tracer (minutes)")
    s.add_argument("--save", action="append", default=[], metavar="FILE",
                   help="write the answer: *.json, input.<name> (VMEC) or *.vtk. May be given more than once")
    s.add_argument("--mpol", type=int, metavar="N", help="cut the VMEC input written to MPOL = N: modes m < N, as "
                                                         "VMEC counts them (default: every mode)")
    s.add_argument("--ntor", type=int, metavar="N", help="cut the VMEC input written to NTOR = N: modes |n| <= N "
                                                         "(default: every mode)")
    s.add_argument("--plot", metavar="FILE", help="draw the surface inside its coils to FILE: .png, .pdf or .svg")
    s.add_argument("--boozer", action="store_true", help="also the Boozer spectrum of |B| on the surface and its "
                                                         "quasi-symmetry")
    from .check import add_source_choices
    add_source_choices(s)
    s.add_argument("--out", help=argparse.SUPPRESS)


def widths_args(s):
    """--widths and --widths-jobs (torux.widths), on solve and many."""
    s.add_argument("--widths", type=float, metavar="S", help="with --largest: close in on every island chain of the "
                                                             "answer with verified surfaces either side, in S seconds "
                                                             "of its own (each chain's bracket)")
    s.add_argument("--widths-jobs", type=int, metavar="N", help="requests the closing in runs at once, each on "
                                                                "TORUX_THREADS cores of its own (default 1)")


def widths_options(a):
    """torux.solve's widths_s and widths_jobs from --widths and --widths-jobs ({} without them); ValueError for
    --widths-jobs without --widths."""
    if a.widths is None:
        if a.widths_jobs is not None:
            raise ValueError("--widths-jobs is the requests at once of --widths: give --widths with it")
        return {}
    return dict(widths_s=a.widths, widths_jobs=a.widths_jobs)


def _say(text):
    print(f"torux: {text}", file=sys.stderr, flush=True)


def _question_args(s):
    """The question, as solve asks it: the flux surface by one of four, the budget, and how to read the source."""
    ask = s.add_mutually_exclusive_group()
    ask.add_argument("--flux", type=float, metavar="WB", help="the surface enclosing this full toroidal flux, in Wb")
    ask.add_argument("--minor-radius", type=float, metavar="M", help="the surface of this minor radius, in m")
    ask.add_argument("--volume", type=float, metavar="M3", help="the surface enclosing this volume, in m^3")
    ask.add_argument("--largest", action="store_true", help="the largest surface torux can verify for the source")
    s.add_argument("--budget", type=float, metavar="S", help="seconds for the whole question (default: stated from "
                                                             "the source's size)")
    s.add_argument("--nfp", type=int, metavar="N", help="field periods, where the format does not state them")
    s.add_argument("--kind", choices=("stellarator", "tokamak", "dipole"), help="what the source is (default: "
                                                                                "stellarator)")


def _ask(a, rendering=False, **options):
    """The answer to the question on the command line, saying on stderr what it asks and how long it may take."""
    import torux
    from . import ask
    ask.one_question(a.flux, a.minor_radius, a.volume, a.largest)
    if a.budget is not None and not (a.budget > 0 and a.budget < float("inf")):
        raise ValueError(f"--budget {a.budget:g}: give the seconds for the whole question as a positive number")
    from .check import source_choices
    src = torux.load(*a.files, nfp=a.nfp, kind=a.kind, **source_choices(a))
    if rendering and src.has_surface_current:
        from .answer import RENDER_SHEET
        raise ValueError(RENDER_SHEET)
    own = a.flux if a.flux is not None else (src.request or {}).get("flux_Wb")
    if own is not None and a.minor_radius is None and a.volume is None and not a.largest:
        from .api import estimate_budget          # one request. A search says what it asks itself
        given = a.budget is not None
        b = a.budget if given else estimate_budget(src, own)
        _say(f"solving {src.id} at {abs(float(own)):.9g} Wb" + (" (its own request)" if a.flux is None else "")
             + f": up to {b:.0f} s, " + ("the budget given" if given else "a budget stated from the source's "
                                                                          "size since none was given"))
    ask.say = _say
    try:
        return torux.solve(src, a.flux, minor_radius=a.minor_radius, volume=a.volume, largest=a.largest,
                           budget_s=a.budget, **options)
    finally:
        ask.say = None


def _code(res):
    return 0 if res.status in ("surface", "no_surface", "range") else 1 if res.status == "error" else 2


def _solve(a):
    import json
    import warnings
    # numpy's notes on the solver's own arithmetic (a NaN it goes on to discard) are not the user's to read: stderr
    # carries what is asked and each request's outcome
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"torux(\.|$)")
    from .answer import save_format
    try:
        cut = a.mpol is not None or a.ntor is not None
        if cut and not any(save_format(p) == "vmec" for p in a.save):
            raise ValueError("--mpol and --ntor cut a VMEC input file: give one to write, --save input.<name>")
        for p in a.save:
            save_format(p)
        if a.plot:
            from .plot import check_path
            check_path(a.plot)
        res = _ask(a, verify=not a.no_verify, trace_lines=a.trace_lines, gradients=a.gradients,
                   **widths_options(a))
    except Exception as e:  # noqa: BLE001
        print(f"error   {e}", file=sys.stderr)
        return 1
    print(res)
    code = _code(res)
    if a.boozer:
        try:
            print("\n" + str(res.boozer()))
        except ValueError as e:
            print(f"no Boozer spectrum   {e}", file=sys.stderr)
    for p in a.save:
        try:
            if save_format(p) == "vmec" and cut:
                moved = res.save(p, mpol=a.mpol, ntor=a.ntor)
                _say(f"{p}: cut to " + ", ".join(f"{k} = {v}" for k, v in (("MPOL", a.mpol), ("NTOR", a.ntor))
                                                  if v is not None)
                     + f": no point of the surface moved by more than {moved:.3e} m")
            else:
                res.save(p)
        except ValueError as e:
            print(f"not saved   {p}: {e}", file=sys.stderr)
        except OSError as e:
            print(f"error   {e}", file=sys.stderr)
            code = 1
    if a.plot:
        try:
            with warnings.catch_warnings(record=True) as said:
                warnings.simplefilter("always", UserWarning)
                res.plot(a.plot)
            for w in said:                  # what the picture leaves out, on stderr as a line of its own
                _say(f"{a.plot}: {w.message}")
        except ValueError as e:             # nothing to draw: the answer stands, and so does its exit code
            print(f"not drawn   {a.plot}: {e}", file=sys.stderr)
        except (ImportError, OSError) as e:
            print(f"error   {e}", file=sys.stderr)
            code = 1
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res.to_dict(), fh, allow_nan=False)
    return code


def _render_args(sub):
    r = sub.add_parser("render", help="the answer as one path-traced picture in the look of the README: minutes",
                       description="Solves the source as solve does, prints the answer, and draws it to FILE as "
                                   "answer.render does: its coils in copper and the surface, or the certificate that "
                                   "there is none, in light, a wedge of the plasma cut away in front of one "
                                   "cross-section. Minutes of path tracing. An answer with nothing of the plasma to "
                                   "draw is drawn as its coils alone. A proof that a source which is not a "
                                   "stellarator has no surface keeps no coils: it is not drawn, and the code is 1.")
    r.add_argument("files", nargs="+", metavar="SOURCE",
                   help="a coil or magnet file, or quasr:NUMBER. Several files whose parts add up are one source")
    r.add_argument("file", metavar="FILE", help="the picture to write: a .png file")
    _question_args(r)
    r.add_argument("--dark", action="store_true", help="for a dark page")
    r.add_argument("--closed", action="store_true", help="the machine closed, with no cross-section opened")
    from .check import add_source_choices
    add_source_choices(r)


def _stopped(run):
    """run(a), where Ctrl-C ends the command in one line, with the code a shell gives a command Ctrl-C ended (130)."""
    def wrapped(a):
        try:
            return run(a)
        except KeyboardInterrupt:
            print("stopped: nothing written", file=sys.stderr)
            return 130
    wrapped.__doc__ = run.__doc__
    return wrapped


@_stopped
def _render(a):
    import os
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"torux(\.|$)")
    from .plot import check_folder
    from .render import _source
    try:
        if os.path.splitext(a.file)[1].lower() != ".png":
            raise ValueError(f"cannot render to {a.file!r}: name a .png file")
        check_folder(a.file)
        res = _ask(a, rendering=True)
    except Exception as e:  # noqa: BLE001
        print(f"error   {e}", file=sys.stderr)
        return 1
    print(res, flush=True)                  # the answer stands before the minutes of tracing, into a pipe too
    try:
        _source(res)
    except ValueError as e:                 # nothing to draw, said before any work
        print(f"not drawn   {a.file}: {e}", file=sys.stderr)
        return 1
    _say(f"rendering {a.file}: minutes of path tracing")
    try:
        res.render(a.file, dark=a.dark, cut=not a.closed)
    except ValueError as e:
        print(f"not drawn   {a.file}: {e}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"error   {e}", file=sys.stderr)
        return 1
    return _code(res)


def _check_args(sub):
    from .check import add_command
    add_command(sub)


def _check(a):
    from .check import run_command
    return run_command(a)


def _many_args(sub):
    from .batch import add_parser
    add_parser(sub)


def _many(a):
    from .batch import main
    return main(a)


def _poincare_args(sub):
    s = sub.add_parser("poincare", help="field lines through planes of constant toroidal angle, and their picture",
                       usage="torux poincare SOURCE [SOURCE ...] [FILE] [--planes DEG [DEG ...]] [--lines N] "
                             "[--turns N] [--start R,Z] [--dark] [--nfp N] [--kind K] [--coils NAME] "
                             "[--currents saved|dofs]",
                       description="Field lines of the source's own field, traced through planes of constant toroidal "
                                   "angle: where each line starts, its rotational transform and whether it left the "
                                   "machine. With FILE, the picture. torux chooses where lines start, how many and "
                                   "how far they go, unless told. A SurfaceCurrent source requires --start.")
    s.add_argument("files", nargs="+", metavar="SOURCE",
                   help="a coil or magnet file, or quasr:NUMBER (several files whose parts add up are one source), "
                        "then optionally FILE: the picture, .png, .pdf or .svg")
    s.add_argument("--planes", nargs="+", metavar="DEG",
                   help="the planes' toroidal angles in degrees (default: the start, a quarter and a half of a field "
                        "period)")
    s.add_argument("--lines", type=int, metavar="N", help="lines started from the magnetic axis to the edge (default "
                                                          "16), besides those torux adds in island chains")
    s.add_argument("--turns", type=float, metavar="N", help="toroidal turns each line is followed (default: 600 "
                                                            "field periods)")
    s.add_argument("--start", action="append", metavar="R,Z", help="a line starting here (m) in the first plane, "
                                                                   "instead of torux's choice. May be given more "
                                                                   "than once")
    s.add_argument("--dark", action="store_true", help="draw FILE for a dark page")
    s.add_argument("--nfp", type=int, metavar="N", help="field periods, where the format does not state them")
    s.add_argument("--kind", choices=("stellarator", "tokamak", "dipole"), help="what the source is (default: "
                                                                                "stellarator)")
    from .check import add_source_choices
    add_source_choices(s)


# names of pictures: the last argument named so is the picture asked for, and one torux cannot draw is refused as such
PICTURES = (".png", ".pdf", ".svg", ".jpg", ".jpeg", ".gif", ".tif", ".tiff", ".bmp", ".webp", ".eps", ".ps")


def _degrees(values, files):
    """--planes' angles, and the arguments after them that are not angles moved back to the files (argparse gives
    --planes every argument up to the next option)."""
    out = []
    for i, v in enumerate(values):
        try:
            out.append(float(v))
        except ValueError:
            rest = values[i:]
            if not out or any(_number(r) for r in rest):
                raise ValueError(f"--planes takes the planes' angles in degrees: {v!r} is not one") from None
            files += rest
            break
    return out


def _number(v):
    try:
        float(v)
        return True
    except ValueError:
        return False


@_stopped
def _poincare(a):
    import math
    import os
    import torux
    files, picture = list(a.files), None
    try:
        degrees = None if a.planes is None else _degrees(a.planes, files)
        if len(files) > 1 and os.path.splitext(files[-1])[1].lower() in PICTURES:
            picture = files.pop()
        if picture is not None:
            from .plot import check_folder, check_path
            check_path(picture)
            check_folder(picture)
        starts = None
        if a.start:
            try:
                starts = [tuple(float(v) for v in s.split(",")) for s in a.start]
            except ValueError:
                starts = [()]
            if any(len(s) != 2 for s in starts):
                raise ValueError("--start takes R,Z in metres, for example --start 1.1,0")
        from .check import source_choices
        from . import ask
        src = torux.load(*files, nfp=a.nfp, kind=a.kind, **source_choices(a))
        ask.say = _say
        try:
            p = torux.poincare(src, planes=None if degrees is None else [math.radians(v) for v in degrees],
                               lines=a.lines, turns=a.turns, starts=starts)
        finally:
            ask.say = None
    except Exception as e:  # noqa: BLE001
        print(f"error   {e}", file=sys.stderr)
        return 1
    print(p)
    print("\n" + p.lines_table())
    if picture is not None:
        there = os.path.exists(picture)
        try:
            p.plot(picture, dark=a.dark)
        except (ImportError, OSError, ValueError) as e:
            print(f"error   {e}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            if not there and os.path.exists(picture):
                os.remove(picture)              # half a picture is not one
            raise
    return 0


def _version_args(sub):
    sub.add_parser("version", help="print the version")


def _version(a):
    import torux
    print(torux.__version__)
    return 0


# subcommand: (add its parser to the subparsers, run it on the parsed arguments and return the exit code)
COMMANDS = {"solve": (_solve_args, _solve), "check": (_check_args, _check), "many": (_many_args, _many),
            "render": (_render_args, _render), "poincare": (_poincare_args, _poincare),
            "version": (_version_args, _version)}


class _Parser(argparse.ArgumentParser):
    """A command line torux cannot read is an error, code 1: argparse's own 2 is the code of an undetermined answer."""

    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def main(argv=None):
    ap = _Parser(prog="torux", description="Vacuum flux surfaces of any external magnetic source.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for add, _ in COMMANDS.values():
        add(sub)
    a = ap.parse_args(argv)
    return COMMANDS[a.cmd][1](a)


if __name__ == "__main__":
    sys.exit(main())
