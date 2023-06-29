import os
import sys
import pandas as pd
import numpy as np
import dill
import pickle

from src.exception import CustomException


def save_object(filepath, obj):

    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)

        with open(filepath, "wb") as filehandle:
            pickle.dump(obj, filehandle)

    except Exception as e:
        raise CustomException(e,sys)