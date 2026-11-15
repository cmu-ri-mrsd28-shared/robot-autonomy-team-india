"""Console messages with the source filename and line of the caller."""

import os
import sys


def log(*values, sep=" ", flush=True):
    caller = sys._getframe(1)
    source = f"{os.path.basename(caller.f_code.co_filename)}:{caller.f_lineno}"
    for line in sep.join(str(value) for value in values).split("\n"):
        print(f"{source} {line}", flush=flush)
