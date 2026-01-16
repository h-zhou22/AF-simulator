import random
import math
from request import Request
from typing import List, Dict, Tuple

class Batch:
    def __init__(self, bids, batch_size,  use_length_limit=False, length_limit=0):
        self.bids = bids  # List of request IDs in the batch
        self.requests :List[Request] = []  # Requests in the batch
        self.batch_size = batch_size # Maximal number of requests allowed
        self.length = 0
        self.num_req = 0
        self.use_length_limit = use_length_limit
        self.length_limit = length_limit
        self.ever_served_request = 0

        self.status = 0
        # 0: Empty
        # 1: Attention processing, 2: FFN processing
        # 3: A2F transfer, 4 F2A transfer
        # 5: Waiting for allocation in attention 
        # 6: Waiting for allocation in FFN
        self.current_ending = 0 # Time to finish current stage
        self.attention_now = False # Just finish last-round work
        self.doing_FFN = False # Just arrive at FFN instance

        self.round_cost:list[int] = []
        self.A_arrival:list[int] = []
        self.current_A_arrival:int = 0 
        self.A_finish:list[int] = []
        self.F_arrival:list[int] = []
        self.F_finish:list[int] = []
        self.Acost:list[int] = []
        self.Fcost:list[int] = []
        

    def load_request(self, current_time, request:Request):
        self.requests.append(request)
        request.start_processing(current_time, self.bids)
        self.length += request.length
        self.num_req += 1
        if self.status == 0:
            self.status = 1
            self.attention_now = True
        

    def finish_request(self, current_time, request:Request)-> bool:
        if request not in self.requests:
            raise ValueError("Request not in batch")
            #return False
        self.ever_served_request += 1
        self.requests.remove(request)
        self.length -= (request.length-1)
        self.num_req -= 1
        if self.num_req == 0:
            self.status = 0
            #raise ValueError("Ever reached here")
        return True
        
    def Attention_processing(self, current_time, alpha_A, beta_A):
        # t_A(T)=alpha_A*T+beta_A
        self.status = 1
        current_ending = current_time + alpha_A*self.length + beta_A
        self.current_ending = math.ceil(current_ending)

        self.Acost.append(self.current_ending - current_time)

    def FFN_processing(self, current_time, alpha_F, beta_F) -> int:
        # t_F(T)=alpha_F*T+beta_F
        self.status = 2
        current_ending = current_time + alpha_F*self.num_req + beta_F
        self.current_ending = math.ceil(current_ending)

        self.Fcost.append(self.current_ending-current_time)

        return self.current_ending

    def A2F_transmission(self, current_time, alpha_T, beta_T):
        # t_T(T)=alpha_T*T+beta_T
        self.status = 3
        current_ending = current_time + alpha_T*self.num_req + beta_T
        self.current_ending = math.ceil(current_ending)
        
        self.A_finish.append(current_time) 

    def F2A_transmission(self, current_time, alpha_T, beta_T):
        # t_T(T)=alpha_T*T+beta_T
        self.status = 4
        current_ending = current_time + alpha_T*self.num_req + beta_T
        self.current_ending = math.ceil(current_ending)

        self.F_finish.append(current_time)
        

    def F2A_transmission_end(self,current_time):
        self.status = 1  # Waiting for allocation in attention
        self.attention_now = True
        self.A_arrival.append(current_time) # One longer since final round
        self.round_cost.append((current_time-self.current_A_arrival))
        self.current_A_arrival = current_time

    def A2F_transmission_end(self,current_time):
        self.status = 6  # Waiting for allocation in attention
        self.F_arrival.append(current_time)

    def do_new_round(self, current_time, stats):
        self.collect_makespan(current_time)
        for request in self.requests:
            flag = request.do_new_round(current_time, stats)
            if flag:
                self.length += 1
            else:
                self.finish_request(current_time, request)

    def update_info(self, current_time):
        return self.num_req, self.length

    def has_free_slot(self, current_time)->bool:
        if self.batch_size <= self.num_req:
            return False
        if self.use_length_limit:
            if self.length >= self.length_limit:
                return False
        return True

    def collect_makespan(self, current_time):
        # TODO
        None