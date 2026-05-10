import random
import math
from request import Request
from batch import Batch
from typing import List, Dict, Tuple

class Server:
    def __init__(self, server_id, num_batches, batch_size, unit_FFN_time, 
                 batches: dict[int,Batch], memory_capacity = 960000, dynamic_matching = False):
        self.num_batches = num_batches
        self.batches = batches
        self.batch_size = batch_size

        assert len(batches) == num_batches
        self.server_id = server_id
        self.unit_FFN_time = unit_FFN_time
        self.weight = 0
        self.current_busy = False

        # 按照逐个Batch进行AF匹配还是整个Server一同进行AF匹配
        self.batch_matched = False
        self.mapped_FFN_id = -1
        self.FFN_level = -1
        self.balanced_FFN = 0 # 当前分配的FFN_worker是否满足其需要。0恰当, 1过多, -1过少
        #self.FFN_server_cnt = 0
        """ 上一个完成的Batch的ID, 在整个server换组时方便考虑哪个在前
        以开始FFN工作的时间为准, 因为最后已经开始的FFNwork不会因换组而中止"""
        self.last_finished_batch_id = -1
        self.first_finished_batch_id = -1
        #self.ceiling = True

        self.dynamic_matching = dynamic_matching
        self.memory_capacity = memory_capacity
        # Attention里面不维护静态的memory size，每次需动态查询
        #self.memory_used = 0
        self.typed_server = False
        self.served_type = []
        for batch in batches.values():
            self.served_type.append(batch.served_type)
        self.batch_size_upper_bound = 64
        self.dynamic_space = 4096

        self.requests_to_be_evicted = []

    # 只判断是否能够装得下
    def judge_capable(self, request: Request)-> bool:
        memory_used = self.compute_memory_usage()
        if memory_used + request.length > self.memory_capacity:
            return False
        else:
            return True

    def compute_memory_usage(self):
        memory_used = 0
        for batch in self.batches.values():
            memory_used += (batch.length+batch.waiting_tot_length)
        return memory_used
    
    def load_request_to_batch(self, current_time, batch_id, request:Request):
        # 不维护静态的memory大小(为了更好的代码兼容度)
        self.batches[batch_id].append_request_to_waiting_buffer(current_time, request)
        # self.batches[batch_id].load_request(current_time, request)

    def find_available_batch(self)-> List[Tuple[int, int, int, int]]:
        available_batches = []
        for batch_id, batch in self.batches.items():
            if batch.num_req < batch.batch_size:
                if batch.use_length_limit:
                    if batch.length > batch.length_limit:
                        continue
                available_batches.append((batch.num_req, batch.length, batch_id, self.server_id))
        return available_batches

    def cycle_work(self, current_time, stats, FFN_worker=None, alpha_T=0, beta_T=1, score_table = None):
        for batch_id, batch in self.batches.items():
            # if batch.status == 5: # Waiting for allocation in attention
            #     if self.current_busy == False:
            #         batch.Attention_processing(current_time, alpha_F, beta_F)
            #         self.current_busy = True
            if batch.status == 3:
                # 传输时间无需取整
                if batch.is_due(current_time):
                    batch.F2A_transmission_end(current_time)
                    batch.do_new_round(current_time, stats)  # stats is None for now
            elif batch.status == 4: # Waiting for allocation in FFN
                if self.dynamic_matching:
                    continue
                if batch.is_due(current_time):
                    batch.A2F_transmission_end(current_time)
                    FFN_worker.load_batch(current_time, batch)
                    #batch.FFN_processing(current_time, alpha_F, beta_F)
            elif batch.status == 1:
                if batch.attention_now:
                    continue # Should be done in attention_work
                if batch.is_due(current_time):
                    #raise Exception("Batch {} in server {} is still in attention".format(batch_id, self.server_id))
                    batch.A2F_transmission(current_time, alpha_T, beta_T)
                    self.current_busy = False
            elif batch.status == 2:
                if batch.is_due(current_time):
                    batch.F2A_transmission(current_time, alpha_T, beta_T)
                    self.first_finished_batch_id = self.last_finished_batch_id
                    self.last_finished_batch_id = batch_id
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
        for bid in self.batches:
            if bid == batch_id:
                continue
            other_batch = self.batches[bid]
            tot_len += other_batch.length
        batch.other_batch_cost = alpha_A*tot_len + beta_A
        """注意,这里考虑每个server只同时维护两个Batch"""
        batch.other_batch_FFN_unit_cost = batch.other_batch_cost / self.unit_FFN_time

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

    def map_to_FFN(self, FFN_id, FFN_level):
        self.mapped_FFN_id = FFN_id
        self.FFN_level = FFN_level
        # for batch in self.batches.values():
        #     batch.map_to_FFN(FFN_id, FFN_level)

    def update_FFN_level(self, alpha_A, beta_A):
        #print("Unit FFN cost: {}, Attention cost: {}".format(self.unit_FFN_time, self.compute_total_cost(alpha_A, beta_A)))
        self.weight = self.compute_total_unit_cost(alpha_A, beta_A)

    def reactivate_batches(self, current_time, alpha_A, beta_A, alpha_T, beta_T):
        for batch in self.batches.values():
            batch.reactivate_after_match_switching(current_time, alpha_T, beta_T)

    def evict_requests(self, current_time):
        # 无需返回, 只负责将一些requests加入self.requests_to_be_evicted
        server_longest_req = None
        for batch in self.batches.values():
            if batch.served_type != -1:
                evil_reqs = batch.find_evil_requests(current_time)
                if evil_reqs is not None:
                    for req in evil_reqs:
                        self.requests_to_be_evicted.append(req)
                else: # 在这种情况下, 需要添加最长的request
                    if batch.served_type == 4:
                        continue
                    longest_req = batch.find_longest_request(current_time)
                    self.requests_to_be_evicted.append(longest_req)

    def evict_out_requests(self, current_time):
        evict_out = []
        for req in self.requests_to_be_evicted:
            if req.status == 3:
                evict_out.append(req)
        for req in evict_out:
            self.requests_to_be_evicted.remove(req)
        return evict_out

    def print_debug_information(self):
        print("Server ID: {}, server capacity:{}".format(self.server_id, self.memory_capacity))
        batch_print = True
        if batch_print:
            for batch_id, batch in self.batches.items():
                batch.print_debug_information()