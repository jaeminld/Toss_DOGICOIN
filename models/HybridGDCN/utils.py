import torch
import random
import os
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prob_to_logit(p):
    eps = 1e-7
    p = np.clip(p, eps, 1-eps)
    return np.log(p / (1 - p))


def logit_to_prob(l):
    return 1 / (1 + np.exp(-l))