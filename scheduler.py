import random
import math
from request import Request
from batch import Batch
from attention import Server
from typing import List, Dict, Tuple
from FFN import FFN, dynamic_FFN

class BasicScheduler:
# AF之间固定匹配不再更改
    def __init__(self, servers:List[Server], FFN_workers:List[FFN], stats, buffer, stored_batches:Dict[int, Batch], alpha_A, beta_A, alpha_T, beta_T, alpha_F, beta_F, initially_full=False):
        self.servers = servers
        self.buffer = buffer
        self.stored_batches = stored_batches
        self.num_servers = len(servers)
        self.FFN_workers = FFN_workers
        self.num_FFN = len(FFN_workers)
        self.stats = stats
        self.alpha_A = alpha_A
        self.beta_A = beta_A
        self.alpha_T = alpha_T
        self.beta_T = beta_T
        self.alpha_F = alpha_F
        self.beta_F = beta_F
        self.AF_match = Dict[int, int] # server_id -> FFN_id

        self.initially_full = initially_full
        if self.initially_full:
            self.do_initialize_filling()
        self.match_AF()

    def do_initialize_filling(self):
        if self.buffer().size() < self.num_batches*self.batch_size:
            raise ValueError("Buffer size is smaller than total batch capacity")
        for batch in self.stored_batches.values():
            if batch.num_req < batch.batch_size:
                request = self.buffer.pop()
                batch.append_request(0, request)
                # load_request加入之后会立刻开始处理, 所以初始化都用append

    def match_AF(self):
        # match each server to a given FFN worker
        for i in range (self.num_servers):
            server = self.servers[i]
            FFN_worker = self.FFN_workers[i % self.num_FFN]
            server.FFN_worker = FFN_worker
            self.AF_match[server.server_id] = FFN_worker.worker_id

    def do_cycle_work(self, current_time):
            for server_id in len(self.servers):
            # 这里假定了server和FFN的ID是按照顺序排列的，后续Debug的时候需注意
                server = self.servers[server_id]
                FFN_server = self.FFN_workers[self.AF_match[server_id]]
                server.cycle_work(current_time, self.stats, FFN_server, self.alpha_T, self.beta_T)

            available_batches : List[Tuple[int, int, int, int]] = []
            for server in self.servers:
                extend_batches = server.find_available_batch()
                available_batches.extend(extend_batches)

            while available_batches and self.buffer:
                request = self.buffer.pop()
                best_batch_info = min(available_batches)
                batch_id0 = best_batch_info[2]
                server_id0 = best_batch_info[3]
                best_batch = self.stored_batches[batch_id0]
                target_server = self.servers[server_id0]
                target_server.load_request_to_batch(current_time, best_batch_info[2], request)
                available_batches.remove(best_batch_info)
                if best_batch.has_free_slot(current_time):
                    info0, info1 = best_batch.updated_info(current_time=current_time)
                    new_info = (info0, info1, batch_id0, server_id0)
                    available_batches.append(new_info)
            
            for server in self.servers:
                server.attention_work(current_time, self.alpha_A, self.beta_A)
            for FFN_worker in self.FFN_workers:
                FFN_worker.cycle_work(current_time, self.alpha_F, self.beta_F)

class PipelineScheduler:
# AF之间存在固定匹配，可能会被动态修改
    def __init__(self, servers:List[Server], FFN_workers:List[FFN], stats, buffer, stored_batches:Dict[int, Batch], alpha_A, beta_A, alpha_T, beta_T, alpha_F, beta_F, initially_full=False):
        self.servers = servers
        self.batch_size = servers[0].batch_size
        self.buffer = buffer
        self.stored_batches = stored_batches
        self.num_batches = len(stored_batches)
        self.num_servers = len(servers)
        self.FFN_workers = FFN_workers
        self.num_FFN = len(FFN_workers)
        self.stats = stats
        self.alpha_A = alpha_A
        self.beta_A = beta_A
        self.alpha_T = alpha_T
        self.beta_T = beta_T
        self.alpha_F = alpha_F
        self.beta_F = beta_F
        self.AF_match = Dict[int, int] # server_id -> FFN_id

        self.Batch_FFN_unit_count = Dict[int, int]
        self.server_FFN_unit_count = Dict[int, int]

        # 单个FFN worker至多同时为多少个Attention server提供服务, 在此调整
        self.max_AF_ratio = 5
        self.AF_ratio_status = [0] * self.max_AF_ratio

        self.initially_full = initially_full
        # 先把所有的Attention装满
        if self.initially_full:
            self.do_initialize_filling()
        # TODO: 初始化AF匹配策略以及后续AF匹配的动态调整

        self.do_initial_matching()

    def do_initialize_filling(self):
        if self.buffer().size() < self.num_batches*self.batch_size:
            raise ValueError("Buffer size is smaller than total batch capacity")
        for batch in self.stored_batches.values():
            while batch.num_req < batch.batch_size:
                request = self.buffer.pop()
                batch.append_request(0, request)
                # load_request加入之后会立刻开始处理, 所以初始化都用append
            batch.start_processing_from_empty(0)

    def do_initial_matching(self):
        # 先记录初始时AF用时比例分配初始资源
        self.do_initial_FFN_unit_recording()
        for server in self.servers:
            None
        #TODO

    def do_initial_FFN_unit_recording(self):
        for batch in self.stored_batches.values():
            self.Batch_FFN_unit_count[batch.batch_id] = batch.compute_num_F_unit_time(self.alpha_A, self.beta_A)
        for server in self.serevrs:
            self.server_FFN_unit_count[server.server_id] = server.compute_total_unit_cost(self.alpha_A, self.beta_A)

    def match_AF(self):
        # match each server to a given FFN worker
        for i in range (self.num_servers):
            server = self.servers[i]
            FFN_worker = self.FFN_workers[i % self.num_FFN]
            server.FFN_worker = FFN_worker
            self.AF_match[server.server_id] = FFN_worker.worker_id

    def do_cycle_work(self, current_time):
            for server_id in len(self.servers):
            # 这里假定了server和FFN的ID是按照顺序排列的，后续Debug的时候需注意
                server = self.servers[server_id]
                FFN_server = self.FFN_workers[self.AF_match[server_id]]
                server.cycle_work(current_time, self.stats, FFN_server, self.alpha_T, self.beta_T)

            available_batches : List[Tuple[int, int, int, int]] = []
            for server in self.servers:
                extend_batches = server.find_available_batch()
                available_batches.extend(extend_batches)

            while available_batches and self.buffer:
                request = self.buffer.pop()
                best_batch_info = min(available_batches)
                batch_id0 = best_batch_info[2]
                server_id0 = best_batch_info[3]
                best_batch = self.stored_batches[batch_id0]
                target_server = self.servers[server_id0]
                target_server.load_request_to_batch(current_time, best_batch_info[2], request)
                available_batches.remove(best_batch_info)
                if best_batch.has_free_slot(current_time):
                    info0, info1 = best_batch.updated_info(current_time=current_time)
                    new_info = (info0, info1, batch_id0, server_id0)
                    available_batches.append(new_info)
            
            for server in self.servers:
                server.attention_work(current_time, self.alpha_A, self.beta_A)
            for FFN_worker in self.FFN_workers:
                FFN_worker.cycle_work(current_time, self.alpha_F, self.beta_F)

            

class DynamicScheduler:
    def __init__(self, servers:List[Server], FFN_workers:List[dynamic_FFN], stats, alpha_A, beta_A, alpha_T, beta_T, alpha_F, beta_F):
        self.servers = servers
        self.FFN_workers = FFN_workers
        self.stats = stats
        self.alpha_A = alpha_A
        self.beta_A = beta_A
        self.alpha_T = alpha_T
        self.beta_T = beta_T
        self.alpha_F = alpha_F
        self.beta_F = beta_F