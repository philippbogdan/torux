"""The independent evaluator shares no code with the solver: it imports nothing of torux outside itself."""
import ast
import os

import torux.reference as ref


def _imports(path, package):
    """Absolute names of the modules a file imports."""
    out = set()
    for node in ast.walk(ast.parse(open(path).read())):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package.split(".")
            if node.level:
                base = base[:len(base) - (node.level - 1)]
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            if node.module:
                out.add(mod)                                    # from x.y import name: the module x.y
            else:
                out.update(mod + "." + a.name for a in node.names)   # from . import name: the module .name
    return out


def test_reference_imports_nothing_of_the_solver():
    d = os.path.dirname(ref.__file__)
    files = [f for f in sorted(os.listdir(d)) if f.endswith(".py")]
    assert len(files) > 10
    for f in files:
        for mod in _imports(os.path.join(d, f), "torux.reference"):
            if mod == "torux" or mod.startswith("torux."):
                assert mod == "torux.reference" or mod.startswith("torux.reference."), (f, mod)


def _code_lines(path):
    """The significant lines of a Python file, whitespace removed: no comments, docstring-only or short lines."""
    out = []
    for line in open(path).read().splitlines():
        s = "".join(line.split("#", 1)[0].split())
        if len(s) >= 24 and not s.startswith(("import", "from", '"""', "'''")):
            out.append(s)
    return out


def test_sheet_reference_shares_no_code_with_the_solver_sheet():
    """The reference's SurfaceCurrent evaluator (sheet.py, sheet_dd.py) is covered by the import test above and
    repeats no run of three significant lines of the solver's (torux/field/surface_current.py)."""
    d = os.path.dirname(ref.__file__)
    mine = [os.path.join(d, f) for f in ("sheet.py", "sheet_dd.py")]
    assert all(os.path.exists(f) for f in mine)
    for f in mine:
        assert not any(m.startswith("torux.field") for m in _imports(f, "torux.reference")), f
    theirs = os.path.join(os.path.dirname(d), "field", "surface_current.py")
    t = _code_lines(theirs)
    shingles = {tuple(t[i:i + 3]) for i in range(len(t) - 2)}
    for f in mine:
        m = _code_lines(f)
        shared = [m[i:i + 3] for i in range(len(m) - 2) if tuple(m[i:i + 3]) in shingles]
        assert not shared, (f, shared[:2])
        assert not set(m) & set(t) - {"returnout"}, (f, sorted(set(m) & set(t))[:5])
