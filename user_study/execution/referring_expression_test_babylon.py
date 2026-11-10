"""Compatibility launcher for the renamed Study 1 application.

New code and operator instructions should use ``study1_referring_expression``.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from user_study.execution.study1_referring_expression import *  # noqa: F401,F403
from user_study.execution.study1_referring_expression import main


if __name__ == "__main__":
    main()
