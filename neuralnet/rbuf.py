from typing import Deque
import numpy as np
import random

class RBUF:

    def __init__(self, max_size=1000000):
        self.buffer = Deque()
        self.max_size = max_size

    def get_random_batch(self, batch_size):
        
        #make get halv the buffer elements
        get = int(len(self.buffer) / 2)

        if batch_size > len(self.buffer):
            return self.buffer

        weights = np.linspace(0, 1, len(self.buffer))
        return random.choices(self.buffer, weights=weights, k=get)

    def add(self, case):
        if len(self.buffer) > self.max_size:
            self.buffer.popleft()
        self.buffer.append(case)