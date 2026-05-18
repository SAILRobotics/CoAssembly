import sys
import math
import numpy as np
import os
import time
import pandas as pd

HOME = os.path.expanduser("~")

    
def check_dir(folder, generate=True):
    if os.path.isdir(folder):
        return True
    else:
        if generate:
            os.makedirs(folder)
            return True
        else:
            return False

def to_normalized_path(path: str) -> str:
    """
    Converts a given path to a Windows-compatible normalized path.
    """
    return os.path.normpath(path)
