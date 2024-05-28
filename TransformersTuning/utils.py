import os
import shutil
import random
import numpy as np
import torch

def set_seed(seed):
    """
    Sets the seed to make everything deterministic, for reproducibility of experiments
    Parameters:
    seed: the number to set the seed to
    Return: None
    """
    # Random seed
    random.seed(seed)
    # Numpy seed
    np.random.seed(seed)

    # Torch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # os seed
    os.environ['PYTHONHASHSEED'] = str(seed)

def remove_previous_model(folder):
    dirs = [x for x in os.listdir(folder) if os.path.isdir(folder+os.sep+x)]
    for x in dirs:
        shutil.rmtree(folder+os.sep+x, ignore_errors=False, onerror=None)