import random
import math
from request import Request
from typing import List, Dict, Tuple

class Batch:
    def __init__(self, batch_id, batch_size, FFN_unit_time,  use_length_limit=False, length_limit=0):
        self.batch_id = batch_id  # List of request IDs in the batch
        self.requests :List[Request] = []  # Requests in the batch
        self.batch_size = batch_size # Maximal number of requests allowed
        self.length = 0
        self.num_req = 0
        self.use_length_limit = use_length_limit
        self.length_limit = length_limit
        self.ever_served_request = 0

        self.other_batch_FFN_unit_cost = 0
        self.FFN_unit_cost = FFN_unit_time

        self.status = 0
        # 0: Empty
        # 1: Attention processing, 2: FFN processing
        # 3: F2A transfer, 4 A2F transfer
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
        
        self.num_F_unittime = 0

        self.ceiling = True
        # 用时是否需要上取整,注意代码各处应保持相同
        # 在取整的情况下, current_ending总是整数

        # 需要被交换, 尚未完成交换，需要丢弃已经load尚在排队的FFN工作
        self.being_swapped = False

        self.mapped_FFN_id = -1
        self.FFN_level = -1

    def append_request(self, current_time,  request:Request):
        self.requests.append(request)
        request.start_processing(current_time, self.batch_id)
        self.length += request.length
        self.num_req += 1

    def start_processing_from_empty(self, current_time):
        if self.status == 0:
            self.status = 1
            self.attention_now = True

    def load_request(self, current_time, request:Request):
        self.append_request(current_time, request)
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
        if self.ceiling:
            self.current_ending = math.ceil(current_ending)
        else:
            self.current_ending = current_ending

        self.Acost.append(self.current_ending - current_time)

    def FFN_processing(self, current_time, alpha_F, beta_F, current_ending):
        # t_F(T)=alpha_F*T+beta_F
        # 在ceiling的情况下总是整数
        self.status = 2
        current_cost = alpha_F*self.num_req + beta_F
        
        if self.ceiling:
        # 此时开始时间需要取整,总用时对应亦取整
            current_ending = current_time + current_cost
            self.current_ending = math.ceil(current_ending)
            current_cost = self.current_ending - current_time
        else:
        # 此时开始时间可以是float
            current_ending = current_ending + current_cost
            self.current_ending = current_ending

        self.Fcost.append(current_cost)

        return self.current_ending

    def A2F_transmission(self, current_time, alpha_T, beta_T):
        # t_T(T)=alpha_T*T+beta_T
        self.status = 4
        current_ending = current_time + alpha_T*self.num_req + beta_T
        if self.ceiling:
            self.current_ending = math.ceil(current_ending)
        else:
            self.current_ending = current_ending
        if not self.being_swapped:
            self.A_finish.append(current_time) 
        else:
            self.being_swapped = False

    def F2A_transmission(self, current_time, alpha_T, beta_T):
        # t_T(T)=alpha_T*T+beta_T
        self.status = 3
        current_ending = current_time + alpha_T*self.num_req + beta_T
        if self.ceiling:
            self.current_ending = math.ceil(current_ending)
        else:
            self.current_ending = current_ending

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

    def updated_info(self, current_time):
        return self.num_req, self.length

    def has_free_slot(self, current_time)->bool:
        if self.batch_size <= self.num_req:
            return False
        if self.use_length_limit:
            if self.length >= self.length_limit:
                return False
        return True

    def compute_num_F_unit_time(self, alpha_A, beta_A)-> float:
        attention_cost = alpha_A*self.length + beta_A
        self.num_F_unittime = attention_cost / self.FFN_unit_cost
        return self.num_F_unittime

    def update_other_batch_FFN_unit_cost(self, num_F_unit_time):
        self.other_batch_FFN_unit_cost = num_F_unit_time

    def collect_makespan(self, current_time):
        # TODO
        None

    def discard_loading_FFN(self):
        self.F_arrival.pop()

    def reactivate_after_match_switching(self, current_time, alpha_T, beta_T):
        if self.status == 6:
            # 推入了等待区一次，重新load会产生一次额外的FFN记次
            self.discard_loading_FFN()
        if self.status == 6 or self.status == 4:
        # 这种情况下需要重新分配给新的FFN worker完成此轮的传输
        # 在这种情况下重新传输到新的FFN_worker开始下一阶段的传输
            self.being_swapped = True
            self.A2F_transmission(current_time, alpha_T, beta_T)
    
    def map_to_FFN(self, FFN_id, FFN_level):
        self.mapped_FFN_id = FFN_id
        self.FFN_level = FFN_level