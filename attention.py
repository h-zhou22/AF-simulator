import random
import math
from request import Request
from batch import Batch
from typing import List, Dict, Tuple

class Server:
    def __init__(self, server_id, num_batches, batch_size, unit_FFN_time, batches: dict[int,Batch]):
        self.num_batches = num_batches
        self.batches = batches
        self.batch_size = batch_size
        assert len(batches) == num_batches
        self.server_id = server_id
        self.unit_FFN_time = unit_FFN_time
        self.current_busy = False

        #self.ceiling = True

    def load_request_to_batch(self, current_time, batch_id, request:Request):
        self.batches[batch_id].load_request(current_time, request)

    def find_available_batch(self)-> List[Tuple[int, int, int, int]]:
        available_batches = []
        for batch_id, batch in self.batches.items():
            if batch.num_req < batch.batch_size:
                if batch.use_length_limit:
                    if batch.length > batch.length_limit:
                        continue
                available_batches.append((batch.num_req, batch.length, batch_id, self.server_id))
        return available_batches
    
    def cycle_work(self, current_time, stats, FFN_worker, alpha_T, beta_T):
        for batch_id, batch in self.batches.items():
            # if batch.status == 5: # Waiting for allocation in attention
            #     if self.current_busy == False:
            #         batch.Attention_processing(current_time, alpha_F, beta_F)
            #         self.current_busy = True
            if batch.status == 3:
                # 传输时间无需取整
                if current_time >= batch.current_ending:
                    batch.F2A_transmission_end(current_time)
                    batch.do_new_round(current_time, stats)  # stats is None for now
            elif batch.status == 4: # Waiting for allocation in FFN
                if current_time >= batch.current_ending:
                    batch.A2F_transmission_end(current_time)
                    FFN_worker.load_batch(current_time, batch)
                    #batch.FFN_processing(current_time, alpha_F, beta_F)
            elif batch.status == 1:
                if batch.attention_now:
                    continue # Should be done in attention_work
                if current_time >= batch.current_ending:
                    batch.A2F_transmission(current_time, alpha_T, beta_T)
                    self.current_busy = False
            elif batch.status == 2:
                if current_time >= batch.current_ending:
                    batch.F2A_transmission(current_time, alpha_T, beta_T)

    def attention_work(self, current_time, alpha_A, beta_A):
        for batch_id, batch in self.batches.items():
            if batch.status == 5: # Waiting for allocation in attention
                if self.current_busy == False:
                    batch.Attention_processing(current_time, alpha_A, beta_A)
                    self.current_busy = True    
            elif batch.status == 1:
                if not batch.attention_now:
                    continue
                batch.attention_now = False
                if self.current_busy == False:
                    batch.Attention_processing(current_time, alpha_A, beta_A)
                    self.current_busy = True
                else:
                    batch.status = 5

    def compute_other_batch_cost(self, batch_id, alpha_A, beta_A):
    # 用于近似计算同组其他Batch的Attention用时, 用于FFN阶段的allocation参考
    # 每轮计算开始时均需更新. 运输时间可以省略
        batch = self.batches[batch_id]
        tot_len = 0
        for batch_id in self.batches:
            if batch_id == batch_id:
                continue
            other_batch = self.batches[batch_id]
            tot_len += other_batch.length
        batch.other_batch_cost = alpha_A*tot_len + beta_A

    def compute_total_cost(self, alpha_A, beta_A):
        # return the total cost of all batches in the current server
        tot_len = 0
        for batch_id in self.batches:
            batch = self.batches[batch_id]
            tot_len += batch.length
        return alpha_A*tot_len + beta_A

    def compute_total_unit_cost(self,alpha_A, beta_A):
        total_cost = self.compute_total_cost(alpha_A, beta_A)
        total_unit = total_cost / (self.unit_FFN_time*self.num_batches)
        return total_unit