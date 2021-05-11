import json
import torch
import random
import numpy as np


def load_config_file(config_file):
    all_params = json.load(open(config_file))
    return all_params


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
