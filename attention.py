import random
from request import Request
from batch import Batch
from typing import List, Dict, Tuple

class Server:
    def __init__(self, server_id, num_batches, batches: dict[int,Batch]):
        self.num_batches = num_batches
        self.batches = batches
        assert len(batches) == num_batches
        self.server_id = server_id
        self.current_busy = False

    def load_request_to_batch(self, current_time, batch_id, request:Request):
        self.batches[batch_id].load_request(current_time, request)

    def find_available_batch(self)-> List[Tuple[int, int, int, int]]:
        available_batches = []
        for batch_id, batch in self.batches.items():
            # 使用 has_free_slot() 方法统一判断逻辑，避免重复代码和不一致
            if batch.has_free_slot(0):  # current_time参数在这里不重要，只是检查条件
                available_batches.append((batch.num_req, batch.length, batch_id, self.server_id))
        return available_batches
    
    def cycle_work(self, current_time, stats, FFN_worker, alpha_T, beta_T):
        for batch_id, batch in self.batches.items():
            if batch.status == 3:  # A2F传输中
                if current_time >= batch.current_ending:
                    batch.A2F_transmission_end(current_time)
                    FFN_worker.load_batch(current_time, batch)
                    
            elif batch.status == 4:  # F2A传输中
                if current_time >= batch.current_ending:
                    batch.F2A_transmission_end(current_time)
                    batch.do_new_round(current_time, stats)  # stats is None for now
            elif batch.status == 1:  # Attention processing
                if current_time >= batch.current_ending:
                    batch.A2F_transmission(current_time, alpha_T, beta_T)
                    self.current_busy = False
            elif batch.status == 2:  # FFN processing (在FFN worker中)
                if current_time >= batch.current_ending:
                    batch.F2A_transmission(current_time, alpha_T, beta_T)

    def attention_work(self, current_time, alpha_A, beta_A, sample_interval=5):
        # 检查当前是否有正在处理的batch（status=1）
        if self.current_busy:
            # 找到当前正在处理的batch，获取其结束时间
            current_processing_ending = -1
            for batch_id, batch in self.batches.items():
                if batch.status == 1:
                    if current_time >= batch.current_ending:
                        # Attention处理完成，在cycle_work中会开始A2F传输
                        self.current_busy = False
                        break
                    else:
                        current_processing_ending = batch.current_ending
                        break
            
            # 如果worker仍然忙碌，检查等待中的batch并计算等待时间
            if self.current_busy and current_processing_ending > 0:
                expected_waiting_time = current_processing_ending - current_time
                for batch_id, batch in self.batches.items():
                    if batch.status == 5:
                        # 计算并记录该次将会等待的时间（一次等待只采一次样）
                        batch.record_waiting_time(current_time, expected_waiting_time, 'WA')
                return
        
        # Worker空闲，处理等待中的batch（与FFN对称）
        for batch_id, batch in self.batches.items():
            if batch.status == 5:  # Waiting for allocation in attention
                # Worker空闲，等待时间为0，记录等待结束（在Attention_processing中处理）
                # 开始处理
                batch.Attention_processing(current_time, alpha_A, beta_A)
                self.current_busy = True
                break  # 一次只处理一个batch