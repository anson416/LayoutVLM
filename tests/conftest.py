"""Test fixtures for the VLM-unreliability integration tests.

The vendored ``vlmunr_bpa`` performs OS file-descriptor-level stdout/stderr
redirection (``dup2`` + ``close``) which is fundamentally incompatible with
pytest's capture machinery (both fight over fds 1/2 and over the ``sys.stdout``
object).  To let the bpy smoke test run under a normal ``pytest tests/`` (no
``-s`` required), we snapshot the *real* terminal fds before pytest installs its
capture, and expose them so the smoke test can run bpa against genuine fds.
"""

import os
import sys

# Snapshot the genuine stdout/stderr fds at import time (plugin load happens
# before per-test fd capture is active for the duration of a test body, but to
# be safe we dup whatever fds 1/2 currently point at).
_REAL_STDOUT_FD = os.dup(1)
_REAL_STDERR_FD = os.dup(2)


def real_terminal_fds():
    """Return fresh dups of the genuine terminal stdout/stderr fds."""

    return os.dup(_REAL_STDOUT_FD), os.dup(_REAL_STDERR_FD)
