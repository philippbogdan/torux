"""The time budget of one analysis, visible to every loop under it.

An analysis entry point (analyse, analyse_design, analyse_general) opens a scope with its deadline; the Newton
iterations, resolution raises, refinements and continuations check it at every step, so a stated budget holds to
about one iteration (a continuation step near a chain is several high-resolution Newton solves: checked only
between steps, a 600 s budget ran 984 s).  Scopes nest: the inner deadline is the earlier of the two."""
from __future__ import annotations

import contextlib
import functools
import inspect
import threading
import time

_local = threading.local()


def deadline():
    """The current absolute deadline (time.time() scale) or None."""
    return getattr(_local, "deadline", None)


def expired(margin=0.0):
    d = deadline()
    return d is not None and time.time() > d - margin


def earliest(d):
    """The earlier of d and the current scope's deadline (None = no deadline)."""
    c = deadline()
    if d is None:
        return c
    return d if c is None else min(c, d)


@contextlib.contextmanager
def scope(d):
    """Run the block with deadline d (absolute; None leaves the enclosing one)."""
    old = deadline()
    _local.deadline = earliest(d) if d is not None else old
    try:
        yield
    finally:
        _local.deadline = old


def budgeted(fn=None, *, extra=None):
    """Decorator for entry points with a budget_s argument: the call runs in a scope ending budget_s (plus the
    argument named `extra`, e.g. a separate range budget) from its start; a dict result gets
    budget_exhausted = True if the deadline passed before it returned."""
    if fn is None:
        return functools.partial(budgeted, extra=extra)
    sig = inspect.signature(fn)

    def arg(bound, name):
        return bound.arguments.get(name, sig.parameters[name].default)

    @functools.wraps(fn)
    def wrapper(*args, **kw):
        bound = sig.bind_partial(*args, **kw)
        b = arg(bound, "budget_s")
        d = None
        if b:
            d = time.time() + float(b) + (float(arg(bound, extra) or 0.0) if extra else 0.0)
        with scope(d):
            out = fn(*args, **kw)
        if isinstance(out, dict) and d is not None and time.time() > d:
            out["budget_exhausted"] = True
        return out
    return wrapper
