from abc import ABC, abstractmethod
import pandas as pd

    
class BaseExecutor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self, train_data):
        return None
    
    @abstractmethod
    def test(self, test_data, config):
        return None