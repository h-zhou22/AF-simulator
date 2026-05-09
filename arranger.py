import heapq

from request import Request
from attention import Server
from stats import StatsCollector
from collections import deque
from typing import List, Tuple, Dict
from batch import Batch

class GlobalArranger:
    def __init__(self, servers, stored_batches, costly_loading: bool = False):
        self.servers = servers
        self.stored_batches = stored_batches
        self.buffer = deque()  # waiting requests
        self.num_servers = len(servers)
        self.num_batches = len(stored_batches)
        self.num_req_inque = 0
        self.num_req_served = 0
        self.num_req_evicted = 0
        self.costly_loading = costly_loading

    # def server_ready(self, servers):
    #     # Arrange the initial servers, TODO
    #     self.servers = servers
    #     self.num_servers = len(servers)

    def inqueue_request(self, request: Request):
        """
        A new request arrives, add it to the buffer.
        """
        #print("Request inque")
        #request.print_debug_information()
        self.buffer.append(request)
        self.num_req_inque += 1
        self.num_req_served += 1

    def do_initial_filling(self, batch: Batch):
        while self.num_req_inque and batch.num_req < batch.batch_size:
            request = self.buffer.popleft()
            self.num_req_inque -= 1
            # 注意, 初始的request直接绕开了buffer的程序, 是为了保证模拟的初始化正常
            request.status = 2
            batch.load_request(0, request)
        

    def evict_all_requests(self, requests: list[Request]):
        """
        A list of requests are evicted from server, add them back to the buffer.
        """
        for request in requests:
            self.evict_request(request)

    def evict_request(self, request: Request):
        """
        A request is evicted from server, add it back to the buffer.
        """
        print("Request {} evicted".format(request.rid))
        self.buffer.appendleft(request)
        self.num_req_inque += 1
        self.num_req_evicted += 1

    def try_allocation(self, current_time, request: Request):
        """
        Try assigning the request to a unit.
        Strategy: choose the unit with the maximum remaining capacity.
        Return True if assigned, else False.
        """
        
        best_server = None
        max_free_capacity = -1
        #print("Allocation gong on")
        for server in self.servers:    
            memory_used = server.compute_memory_usage()    
            if server.memory_capacity < request.length + memory_used:
                continue
            flag = False
            for batch in server.batches.values():
                if batch.has_free_slot(current_time):
                    flag = True
            if not flag: # 没有有容量的Batch也是不行的
                continue

            if server.memory_capacity - memory_used > max_free_capacity:
                max_free_capacity = server.memory_capacity - memory_used
                best_server = server

        if best_server is not None:
            # Directly appending to the server
            max_free_slot = -1
            best_batch = None
            for batch in best_server.batches.values():
                free_slot = batch.count_free_slot()
                if free_slot > max_free_slot:
                    max_free_slot = free_slot
                    best_batch = batch
            # 注意, 这里都先加入等待区
            # 原先这里的含义是直接加入Batch且保证此时刚好可以一起开始
            best_server.load_request_to_batch(current_time,best_batch.batch_id, request)
            #print("Request loaded: ", request.rid)
            return True
        else:
            return False
        
    def arrange_requests(self, current_time, allow_further_find: bool = False):
        if not self.buffer:
            #print("Empty buffer")
            return None
        if not allow_further_find:
            allocated = True
            while allocated:
                if not self.buffer:
                    break
                request = self.buffer[0]
                allocated = self.try_allocation(current_time, request)
                if allocated:
                    self.buffer.popleft()
                    self.num_req_inque -= 1
        else: 
            allocated = True
            while allocated:
                if not self.buffer:
                    return None
                for idx in range(len(self.buffer)):
                    request = self.buffer[idx]
                    allocated = self.try_allocation(current_time,request)
                    if allocated:
                        self.buffer.remove(request)
                        self.num_req_inque -= 1
                        break
                if not allocated:
                    break

class GreedyArranger:
    def __init__(self, servers, costly_loading: bool = False):
        self.servers = servers
        self.buffer = []  # Use heapq to construct a priority queue
        self.num_servers = len(servers)
        self.num_req_inque = 0
        self.num_req_served = 0
        self.num_req_evicted = 0
        self.costly_loading = costly_loading
        

    def do_initial_filling(self, batch: Batch):
        while self.num_req_inque and batch.num_req < batch.batch_size:
            _, _, rid, request = self.buffer.pop()
            self.num_req_inque -= 1
            # 注意, 初始的request直接绕开了buffer的程序, 是为了保证模拟的初始化正常
            request.status = 2
            batch.load_request(0, request)

    def inqueue_request(self, request: Request):
        """
        A new request arrives, add it to the buffer.
        """
        heapq.heappush(self.buffer, (request.length, request.target_length, request.rid, request))
        self.num_req_inque += 1
        self.num_req_served += 1

    def try_allocation(self, current_time, request: Request):
        """
        Try assigning the request to a unit.
        Strategy: choose the unit with the maximum remaining capacity.
        Return True if assigned, else False.
        """
        
        best_server = None
        max_free_capacity = -1
        for server in self.servers:    
            memory_used = server.compute_memory_usage()    
            if server.memory_capacity < request.length + memory_used:
                continue
            flag = False
            for batch in server.batches.values():
                if batch.has_free_slot(current_time):
                    flag = True
            if not flag: # 没有有容量的Batch也是不行的
                continue

            if server.memory_capacity - memory_used > max_free_capacity:
                max_free_capacity = server.memory_capacity - memory_used
                best_server = server

        if best_server is not None:
            # Directly appending to the server
            max_free_slot = -1
            best_batch = None
            for batch in best_server.batches.values():
                free_slot = batch.count_free_slot()
                if free_slot > max_free_slot:
                    max_free_slot = free_slot
                    best_batch = batch
            # 注意, 这里都先加入等待区
            # 原先这里的含义是直接加入Batch且保证此时刚好可以一起开始
            best_server.load_request_to_batch(current_time,best_batch.batch_id, request)
            return True
        else:
            return False
    
    def evict_all_requests(self, requests: list[Request]):
        """
        A list of requests are evicted from server, add them back to the buffer.
        """
        for request in requests:
            self.evict_request(request)

    def evict_request(self, request: Request):
        """
        A request is evicted from server, add it back to the buffer.
        """
        heapq.heappush(self.buffer, (request.length, request.target_length, request.rid, request))
        self.num_req_inque += 1
        self.num_req_evicted += 1
    

    def arrange_requests(self, current_time, allow_further_find: bool = False):
        while self.buffer:
            _, _, _, request = self.buffer[0]  # peek shortest request

            allocated = self.try_allocation(current_time, request)
            if allocated:
                heapq.heappop(self.buffer)
                self.num_req_inque -= 1
            else:
                # shortest one cannot be allocated → no need to try longer ones
                break

class MultitypeArranger:
    """
    Requests are classified by their original length:
      - Type 4: 2049 - 4096
      - Type 3: 1025 - 2048
      - Type 2: 513  - 1024
      - Type 1: 257  - 512
      - Type 0: 1    - 256

    Arrange order: from larger type to smaller type.
    """
    def __init__(self, servers, costly_loading: bool = False):
        self.servers = servers
        self.buffer = {
            0: deque(),  # 1 - 1024
            1: deque(),  # 1025 - 4096
            2: deque(),  # 4097 - 8192
            3: deque(),  # long requests
            4: deque(),  # Predicted short request (20 rounds)
            5: deque(),  # Prediction 1024
            6: deque(),  # Prediction 4096
            7: deque(),  # Prediction 4096-8192
        }
        self.num_servers = len(servers)
        self.num_req_inque = 0
        self.num_req_served = 0
        self.num_req_evicted = 0
        self.costly_loading = costly_loading
        
        self.predictor_scores = []
        for _ in range(1000):
            self.predictor_scores.append(0)

        # 各个type的trequest的对应生成长度
        # 如果更改了实验配置, 此处也需要进行修改
        self.predicted_type_corresponding_length = [0, 20, 1024, 4096]

    def do_initial_filling(self, batch: Batch):
        if batch.served_type == -1:
            for qid in (0, 1, 2, 4, 5, 6, 7, 3):
                while self.buffer[qid] and batch.num_req < batch.batch_size:
                    request = self.buffer[qid].popleft()
                    request.status = 2
                    batch.load_request(0, request)
                    self.num_req_inque -= 1
                if batch.num_req >= batch.batch_size:
                    break
        elif batch.served_type == 4:
                qid = 4
                while self.buffer[qid] and batch.num_req < batch.batch_size:
                    request = self.buffer[qid].popleft()
                    request.status = 2
                    batch.load_request(0, request)
                    self.num_req_inque -= 1
        elif batch.served_type == 5:
            for qid in (5, 4):
                while self.buffer[qid] and batch.num_req < batch.batch_size:
                    request = self.buffer[qid].popleft()
                    request.status = 2
                    batch.load_request(0, request)
                    self.num_req_inque -= 1
                if batch.num_req >= batch.batch_size:
                    break
        elif batch.served_type == 6:
            for qid in (6, 4):
                while self.buffer[qid] and batch.num_req < batch.batch_size:
                    request = self.buffer[qid].popleft()
                    request.status = 2
                    batch.load_request(0, request)
                    self.num_req_inque -= 1
                if batch.num_req >= batch.batch_size:
                    break
        elif batch.served_type == 7:
            for qid in (7, 4):
                while self.buffer[qid] and batch.num_req < batch.batch_size:
                    request = self.buffer[qid].popleft()
                    request.status = 2
                    batch.load_request(0, request)
                    self.num_req_inque -= 1
                if batch.num_req >= batch.batch_size:
                    break
        elif batch.served_type == 1:
            for qid in (6, 1, 4):
                while self.buffer[qid] and batch.num_req < batch.batch_size:
                    request = self.buffer[qid].popleft()
                    request.status = 2
                    batch.load_request(0, request)
                    self.num_req_inque -= 1
                if batch.num_req >= batch.batch_size:
                    break
        elif batch.served_type == 2:
            for qid in (2, 7, 4, 1, 6):
                while self.buffer[qid] and batch.num_req < batch.batch_size:
                    request = self.buffer[qid].popleft()
                    request.status = 2
                    batch.load_request(0, request)
                    self.num_req_inque -= 1
                if batch.num_req >= batch.batch_size:
                    break
        elif batch.served_type == 3:
            for qid in (3, 2, 7):
                while self.buffer[qid] and batch.num_req < batch.batch_size:
                    request = self.buffer[qid].popleft()
                    request.status = 2
                    batch.load_request(0, request)
                    self.num_req_inque -= 1
                if batch.num_req >= batch.batch_size:
                    break

    def length_classify(self, request: Request) -> int:
        # 根据现在的长度而非original length进行划分
        L = request.length
        
        if L <= 1024:
                return 0
        elif L<=4096:
                return 1
        elif L <= 8192:
                return 2
        else:
                return 3
       
    def type_classify(self, request: Request) -> int:
        # 分类
        L = request.length
        predicted_type = request.predicted_type
        predictor = request.agent_belong
        if predicted_type <=0 or predictor < 0 or self.predictor_scores[predictor] <= -5: 
            if L <= 1024:
                return 0
            elif L<=4096:
                return 1
            elif L <= 8192:
                return 2
            else: # long requests
                return 3
        else:
            if self.predictor_scores[predictor] > -5:
                return predicted_type + 3
            else:
                raise ValueError("Predictor score error")

        
    def inqueue_request(self, request: Request):
        """
        A new request arrives, add it to the buffer.
        """
        qid = self.type_classify(request)
        self.buffer[qid].append(request)
        self.num_req_inque += 1
        self.num_req_served += 1

    def evict_all_requests(self, requests: list[Request]):
        """
        A list of requests are evicted from server, add them back to the buffer.
        """
        for request in requests:
            self.evict_request(request)

    def evict_request(self, request: Request):
        """
        A request is evicted from server, add it back to the buffer.
        """
        level = self.length_classify(request)
        agent_id = request.agent_belong
        if self.predictor_scores[agent_id] <= -5:
            self.buffer[level].append(request)
        elif request.predicted_type <= 0:
            self.buffer[level].appendleft(request)
        else:
            assert request.predicted_type <= 3
            predicted_upper_limit = self.predicted_type_corresponding_length[request.predicted_type] + request.original_len
            if request.length <= predicted_upper_limit:
                self.buffer[level].appendleft(request)
            else:
                self.buffer[level].append(request)
        
        self.num_req_inque += 1
        self.num_req_evicted += 1

    def try_allocation(self, current_time, request, qid):
        """
        Try assigning the request to a unit.
        Strategy: choose the unit with the maximum remaining capacity.
        Return True if assigned, else False.
        """
        best_server = None
        max_free_capacity = -1
        for server in self.servers:
            flag = False
            if server.typed_server:
                for batch in server.batches.values():
                    if batch.type_loadable(current_time, qid):
                        flag = True
                        break
                if not flag:
                    continue
            memory_used = server.compute_memory_usage()    
            if server.memory_capacity < request.length + memory_used:
                continue
            free_memory_capacity = server.memory_capacity - memory_used
            if free_memory_capacity > max_free_capacity:
                max_free_capacity = free_memory_capacity
                best_server = server

        if best_server is not None:
            # Directly appending to the server
            max_free_slot = -1
            best_batch = None
            for batch in best_server.batches.values():
                if batch.type_loadable(current_time, qid):
                    free_slot = batch.count_free_slot()
                    if free_slot > max_free_slot:
                        max_free_slot = free_slot
                        best_batch = batch
            # 注意, 这里都先加入等待区
            # 原先这里的含义是直接加入Batch且保证此时刚好可以一起开始
            if best_batch is None:
                print("Server {}, no satisfied batch.".format(best_server.server_id))
                raise ValueError("Server {}, no satisfied batch.".format(best_server.server_id))
            best_server.load_request_to_batch(current_time,best_batch.batch_id, request)
            return True
        else:
            return False
    
    def find_maximal_capacity(self, current_time, qid: int):
        max_capacity = -1
        for server in self.servers:
            if server.typed_server:
                flag = False
                for batch in server.batches.values():
                    if batch.type_loadable(current_time, qid):
                        flag = True
                        break
                if not flag:
                    continue
            
            free_capacity = server.memory_capacity - server.compute_memory_usage()
            if free_capacity > max_capacity:
                max_capacity = free_capacity
        return max_capacity

    def arrange_requests(self, current_time, allow_further_find: bool = False) :

        for qid in (4, 7, 6, 5, 3, 2, 1, 0):
            max_capacity = self.find_maximal_capacity(current_time, qid)
            queue = self.buffer[qid]
            if not queue:
                continue
             
            # Not allow further find, each time check the head of the queue
            allocated = True
            while allocated:
                if not queue:
                    break
                request = queue[0]
                if self.try_allocation(current_time, request, qid):
                    queue.popleft()
                    self.num_req_inque -= 1
                    # max_capacity = self.find_maximal_capacity(current_time)
                    # if max_capacity < (256 << qid):
                    #     break
                    continue
                allocated = False
        
# class Multitype_server_Arranger:
#     def __init__(self, servers, costly_loading: bool = False, vip_arrangement: bool = False):
#         self.servers = servers
#         self.buffer = {
#             0: deque(),  # 1 - 256
#             1: deque(),  # 257 - 512
#             2: deque(),  # 513 - 1024
#             3: deque(),  # 1025 - 2048
#             4: deque(),  # 2049 - 4096
#             5: deque()   # >4096
#         }
#         self.num_servers = len(servers)
#         self.num_req_inque = 0
#         self.num_req_served = 0
#         self.num_req_evicted = 0
#         self.costly_loading = costly_loading
#         self.remaining_status = [0,0,0,0,0,0]
#         self.use_vip = vip_arrangement

#     def inqueue_request(self, request: Request):
#         """
#         A new request arrives, add it to the buffer.
#         """
#         qid = self.length_classify(request)
#         # 暂缓, 此条件下有BUG
#         if (not request.normal_client) and self.use_vip:
#             qid = 5
#         self.buffer[qid].append(request)
#         self.num_req_inque += 1
#         self.num_req_served += 1
#         self.remaining_status[qid] += 1

#     def length_classify(self, request: Request) -> int:
#         # 根据现在的长度而非original length进行划分
#         L = request.length
#         if L <= 256:
#             return 0
#         elif L<=512:
#             return 1
#         elif L <= 1024:
#             return 2
#         elif L <= 2048:
#             return 3
#         elif L <= 4096:
#             return 4
#         # 暂时不支持5级
#         else:
#             return 4

#     def evict_all_requests(self, requests: list[Request]):
#         """
#         A list of requests are evicted from server, add them back to the buffer.
#         """
#         for request in requests:
#             self.evict_request(request)

#     def evict_request(self, request: Request):
#         """
#         A request is evicted from server, add it back to the buffer.
#         """
#         # 这里原先采用的是根据长度划分的level
#         # 在request有优先级的时候应该优先按照request自身的函数划分level
#         level = self.length_classify(request)
        
#         if (not request.normal_client) and self.use_vip:
#             # VIP level
#             level = 5
#         self.buffer[level].appendleft(request)
#         self.num_req_inque += 1
#         self.num_req_evicted += 1
#         self.remaining_status[level] += 1

#     def arrange_requests(self, current_time, allow_further_find: bool = False):
#         for qid_new_2 in range(0, 6, 1):
#         # 1222 Update: shortest first!
#         # 最高优先级最先
#             qid_new = 0
#             if qid_new_2 == 0:
#                 qid_new = 5
#             else:
#                 qid_new = qid_new_2 -1
#             qid = qid_new
#             queue = self.buffer[qid_new]
#             if not queue:
#                 continue
#             if allow_further_find:
#                 raise NotImplementedError("Multitype server arranger with further find not implemented yet.")
#             else:
#                 allocated = True
#                 if qid_new !=5:
#                     while allocated:
#                         if not queue:
#                             break
#                         request = queue[0]
#                         # Status, remaining capacity
#                         current_best = (4, 0)
#                         best_server = None
#                         for server in self.servers:
#                             if server.bottle_neck_level > qid_new:
#                                 # Should not consider this server since the request
#                                 continue
#                             arrange_status = server.judge_loadable(qid, request.length, request.normal_client)
#                             if arrange_status == 0:
#                                 # Failed, continue
#                                 continue
#                             free_capacity = server.free_capacity
#                             if (arrange_status, -free_capacity) < current_best:
#                                 current_best = (arrange_status, -free_capacity)
#                                 best_server = server
#                         if best_server is not None:
#                             best_server.load_request(current_time, request, costly_loading=self.costly_loading, loading_flag = current_best[0])
#                             queue.popleft()
#                             self.num_req_inque -= 1
#                             self.remaining_status[qid_new] -= 1
#                         else:
#                             test_print = False
#                             # if self.num_req_inque == 328 and test_print:
#                             #     print("Debugging")
#                             #     print("Queue ID: ", qid)
#                             #     print("REQUEST ID:", request.rid)
#                             #     print("Request length:", request.length)
#                             allocated = False
#                 else:
#                     while allocated:
#                         if not queue:
#                             break

#                         allocated = False  # 本轮是否成功分配过

#                         # 遍历整个队列（最多一轮）
#                         for idx in range(len(queue)):
#                             request = queue[idx]
#                             qid = request.divide_by_length()

#                             current_best = (4, 0)
#                             best_server = None

#                             for server in self.servers:
#                                 if server.bottle_neck_level > qid_new:
#                                     continue

#                                 arrange_status = server.judge_loadable(
#                                     qid, request.length, request.normal_client
#                                 )
#                                 if arrange_status == 0:
#                                     continue

#                                 free_capacity = server.free_capacity
#                                 if free_capacity + server.unprotected_updates < 0:
#                                     print("Server ID:", server.server_id)
#                                     print("Server capacity:", server.free_capacity)
#                                     print("Server usage:", server.used_capacity)
#                                     print("Server reserved: ", server.reserved_capacity)
#                                     print("Server diffrenece: ", server.difference_capacity)
#                                     print("Server difference2: ", server.difference_capacity_low)

#                                     raise ValueError("Free capacity negative!")
#                                 if (arrange_status, -free_capacity) < current_best:
#                                     current_best = (arrange_status, -free_capacity)
#                                     best_server = server
#                                 # else:
#                                 #     print("status and server_id:", arrange_status, server.server_id, server.free_capacity)

#                             if best_server is not None:
#                                 best_server.load_request(
#                                     current_time,
#                                     request,
#                                     costly_loading=self.costly_loading,
#                                     loading_flag=current_best[0],
#                                 )
#                                 queue.remove(request)     # ⚠ 注意：不能用 popleft
#                                 self.num_req_inque -= 1
#                                 self.remaining_status[qid_new] -= 1
#                                 allocated = True
#                                 break   # 本轮已成功，跳出 for，回到 while

#     # 如果 for 整轮都没有成功分配，allocated 仍为 False
#     # while allocated 会自然退出