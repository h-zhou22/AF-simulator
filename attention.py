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