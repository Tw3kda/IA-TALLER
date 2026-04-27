from abc import ABC, abstractmethod

import numpy as np


class Policy(ABC):

    @abstractmethod
    def act(self, s: np.ndarray) -> int:
        pass
