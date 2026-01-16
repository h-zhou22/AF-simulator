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
        # 当batch进入buffer时，设置status=6（等待FFN worker分配）
        # 如果worker空闲，会在下一个cycle的cycle_work中立即处理
        # 如果worker忙碌，会保持status=6直到worker空闲
        batch.status = 6  # Waiting for allocation in FFN
        # 记录FFN侧等待状态开始
        if len(batch.timeline) == 0 or batch.timeline[-1][0] != 'WF':
            batch.timeline.append(('WF', current_time, -1, batch.current_round))
            batch.last_waiting_record_time = -1  # Reset for immediate first sample
        
    def cycle_work(self, current_time, alpha_F, beta_F, sample_interval=5):
        if self.current_busy:
            if current_time < self.current_ending:
                # Worker忙碌时，计算等待时间并记录
                expected_waiting_time = self.current_ending - current_time
                for batch in self.buffer:
                    if batch.status == 6:
                        # 计算并记录该次将会等待的时间（一次等待只采一次样）
                        batch.record_waiting_time(current_time, expected_waiting_time, 'WF')
                return
            self.current_busy = False
        if self.buffer:
            batch = self.buffer.pop()
            # 如果batch之前处于等待状态(status=6)，记录等待结束
            if batch.status == 6:
                if len(batch.timeline) > 0 and batch.timeline[-1][0] == 'WF':
                    wait_start = batch.timeline[-1][1]
                    waiting_duration = current_time - wait_start
                    batch.timeline[-1] = ('WF', wait_start, current_time, batch.current_round)
                    print(f"[Batch {batch.bids}] Time {current_time}: FFN Waiting Ended, Actual Waiting Time = {waiting_duration} cycles")
            self.current_ending = batch.FFN_processing(current_time, alpha_F, beta_F)
            self.current_busy = True
