#!/usr/bin/env python3
"""Run the standalone Franka task. Default: dry run; --execute commands hardware."""
from franka_lab.runner import main

if __name__ == '__main__':
    raise SystemExit(main())
