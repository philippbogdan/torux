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
