"""Exceptions that say the machine failed, not the source: disk full, quota, I/O error, read-only file system.

The analysis stages that record their own errors re-raise these, so that a failing machine is never recorded as
an undetermined answer for a source it has nothing to do with."""
import errno

ENV_ERRNOS = (errno.ENOSPC, errno.EDQUOT, errno.EIO, errno.EROFS)


def environment_failure(e):
    """True if e, or an exception it was raised from, is an OSError with one of ENV_ERRNOS."""
    seen = set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if isinstance(e, OSError) and e.errno in ENV_ERRNOS:
            return True
        e = e.__cause__ or e.__context__
    return False
