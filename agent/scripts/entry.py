"""PyInstaller entry point wrapper.

Calls huginn.cli:main() so PyInstaller can bundle the CLI.
"""

import sys
from huginn.cli import main

if __name__ == "__main__":
    sys.exit(main())
