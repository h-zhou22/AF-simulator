import random
import math
from request import Request
from typing import List, Dict, Tuple

class Batch:
    def __init__(self, batch_id, batch_size, FFN_unit_time,  use_length_limit=False, length_limit=0, dynamic_matching = False, batch_size_limit = 128):
        self.batch_id = batch_id  # List of request IDs in the batch
        self.server_id =  -1
        self.requests :List[Request] = []  # Requests in the batch
        self.batch_size = batch_size # Maximal number of requests allowed
        self.batch_size_limit = batch_size_limit # 在没有固定Batch size的情况下以此来表示最大允许承载的request数量
        self.length = 0
        self.waiting_tot_length = 0
        # 正在处理的以及刚刚load尚未加入队列的
        self.num_req = 0
        self.num_buffered_req = 0
        # 是否使用长度限制, 曾经处理过的与添加过的request数量
        self.use_length_limit = use_length_limit
        self.length_limit = length_limit
        self.ever_served_request = 0
        self.ever_entered_request = 0

        self.other_batch_cost = 0
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
        self.arranger = None

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
        self.served_type = -1
        """served_type定义与agent_type完全相同 0 <1024, 1 4096, 2 8192
        3 16384, 4 Short generation, 5-7为predicted 1024, 4096, 8192。 -1为普通的batch"""

        self.dynamic_matching = dynamic_matching

        self.waiting_buffer = [] # 等待本轮完成后进行load的request

    def is_due(self, current_time) -> bool:
        """current_ending 落在 [current_time, current_time+1) 内视为本 cycle 到期."""
        return current_time + 1 > self.current_ending


    def append_request(self, current_time,  request:Request):
        self.requests.append(request)
        request.start_processing(current_time, self.batch_id)
        # 此处均为正在处理的长度与数量
       
        self.length += request.length
        self.num_req += 1
        # 在加入Buffer的时候已经完成数量的更新的包含了buffer里面的数量

    def count_free_slot(self):
        cnt = self.batch_size - self.num_req - self.num_buffered_req
        assert cnt >= 0
        return cnt

    def has_free_slot(self, current_time)->bool:
        cnt = self.count_free_slot()
        if cnt <= 0:
            return False
        if self.use_length_limit:
            if self.length >= self.length_limit:
                return False
        return True

    def start_processing_from_empty(self, current_time):
        if self.status == 0:
            self.status = 1
            self.attention_now = True

    def load_request(self, current_time, request:Request):
        request.status = 2 # 开始随着Batch一同处理
        self.append_request(current_time, request)
        if self.status == 0:
            self.status = 1
            self.attention_now = True
        

    def finish_request(self, current_time, request:Request)-> bool:
        if request not in self.requests:
            raise ValueError("Request not in batch")
            #return False
        if request.status == 3:
            # 被evict掉的request, 长度已经扣除, 只需调整batch size
            self.num_req -= 1
            self.requests.remove(request)
            return True

        self.ever_served_request += 1
        self.requests.remove(request)
        self.length -= (request.length-1)
        self.num_req -= 1
        if self.num_req == 0:
            self.status = 0
            #raise ValueError("Ever reached here")
        if self.arranger is not None:
            self.arranger.update_score(request)
            
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
            start_processing = current_time
            end_processing = start_processing + current_cost
            self.current_ending = math.ceil(end_processing)
            current_cost = self.current_ending - current_time
        else:
        # 此时开始时间可以是float
        # 
            start_processing = max(current_ending, current_time)
            end_processing = start_processing + current_cost
            self.current_ending = current_ending

        #print("Fcost: ",current_cost)
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
        self.append_requests_from_waiting_buffer(current_time)
        for request in list(self.requests):
            flag = request.do_new_round(current_time, stats)
            if flag:
                self.length += 1
            else:
                self.finish_request(current_time, request)
        for request in self.waiting_buffer:
            # 将还在处于更新状态的request进行更新
            if request.status == 5:
                if current_time + 1 > request.loading_finished_time:     # 改
                    request.status = 1
            else:
                assert(request.status == 1)

    def updated_info(self, current_time):
        return self.num_req, self.length

    
    
    # def has_free_slot_in_dynamic_size(self, current_time)->bool:
    #     if self.batch_size_limit <= self.num_req:
    #         return False
    #     return True

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
    
    def append_requests_from_waiting_buffer(self, current_time):
        for request in list(self.waiting_buffer):
            if request.status == 5:
                continue
            elif request.status == 1:
                self.load_request(current_time, request)
                self.waiting_buffer.remove(request)
                self.waiting_tot_length -= request.length
                self.num_buffered_req -= 1
            else:
                raise Exception("The requests entered batches should have their status changed.")
            
    # 注意, 一定需要检查过逻辑才能加入waiting Buffer
    def append_request_to_waiting_buffer(self, current_time, request: Request):
        self.waiting_buffer.append(request)
        self.waiting_tot_length += request.length
        self.num_buffered_req += 1
        request.loading_to_Batch_buffer(current_time)
        test_print = True
        if test_print:
            print("Batch {} append request to waiting buffer, time: {}".format(self.batch_id, current_time))
            #self.print_debug_information()

    def map_to_FFN(self, FFN_id, FFN_level):
        self.mapped_FFN_id = FFN_id
        self.FFN_level = FFN_level

    # 对于有效预测 , 是否可以加入这个Batch
    def type_loadable(self, current_time, request_type: int)->bool:
        if not self.has_free_slot(current_time):
            return False
        if self.served_type == -1:
            # 基础Batch， 支持任何种类的请求
            return True
        if request_type <= 3:
            if request_type != self.served_type:
                return False
        elif 5<= request_type <= 7:
            if (request_type-self.served_type) % 5 != 0:
                return False
                
        return True
        

    # def load_request(self, current_time, request: Request):
    #     self.requests.append(request)
    #     self.num_req += 1
    #     self.length += request.length
    #     request.loading_finished_time = current_time + request.loading_time
    #     request.status = 5

    def find_evil_requests(self, current_time):
        evil_requests = []
        for request in self.requests:
            if request.use_target_length:
                if request.target_length < request.length:
                    evil_requests.append(request)
        return evil_requests
    
    def find_longest_request(self, current_time):
        max_length = 0
        longest_request = None
        for request in self.requests:
            if request.length > max_length:
                max_length = request.length
                longest_request = request
            elif request.length == max_length:
                # 如果存在多个最长的request， 那么把初始长度最长的丢出去
                if request.original_len > longest_request.original_len:
                    longest_request = request
        return longest_request
    
    def mark_request_evicted(self, request: Request):
        request.marked_eviction = True
        self.length -= request.length

    def print_info(self):
        print("Batch ID: ", self.batch_id)
        print("Status: ", self.status)
        print("Num_req: ", self.num_req)
        print("Ever served requests: ", self.ever_served_request)
        print("Length: ", self.length)
        print("Length: ", self.length)
        
        print("Acost: ", self.Acost)
        print("Fcost: ", self.Fcost)
        print("A_finish: ", self.A_finish)
        print("F_finish: ", self.F_finish)
        print("A_arrival: ", self.A_arrival)
        print("F_arrival: ", self.F_arrival)
        print("round_cost: ", self.round_cost)

    def print_debug_information(self):
        print("Batch ID: {}, served type: {}, ever served req:{}".format(self.batch_id, self.served_type, self.ever_served_request))
        print("Satus:{}, Batch size:{}, num_req:{}, num_req_in_buffer: {}".format(self.status, self.batch_size, self.num_req, self.num_buffered_req))
        test_print_for_request = False
        if test_print_for_request:
            for req in self.requests:
                req.print_debug_information()
            for req in self.waiting_buffer:
                req.print_debug_information()