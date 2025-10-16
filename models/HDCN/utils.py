import random
import os
import numpy as np
import torch

class RandomnessController:
    """Control all sources of randomness for reproducibility"""
    
    @staticmethod
    def fix_seeds(seed: int) -> None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class DeviceManager:
    """Manage device selection and info"""
    
    @staticmethod
    def get_device() -> torch.device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"  Computing device: {device}")
        return device