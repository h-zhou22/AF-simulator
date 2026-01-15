import random
from request import Request
from batch import Batch
from typing import List, Dict, Tuple
from collections import deque
import math

class FFN:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.current_busy = False
        self.current_ending = -1
        self.buffer = deque()
    
    def load_batch(self, current_time, batch:Batch):
        self.buffer.append(batch)
        
    def cycle_work(self, current_time, alpha_F, beta_F):
        if self.current_busy:
            if current_time < self.current_ending:
                return
            self.current_busy = False
        if self.buffer():
            batch = self.buffer.pop()
            self.current_ending = batch.FFN_processing(current_time, alpha_F, beta_F)
            self.current_busy = True
