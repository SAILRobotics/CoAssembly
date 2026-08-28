"""Compatibility launcher for the renamed Study 1 application.

New code and operator instructions should use ``study1_referring_expression``.
"""

from study1_referring_expression import *  # noqa: F401,F403
from study1_referring_expression import main


if __name__ == "__main__":
    main()
