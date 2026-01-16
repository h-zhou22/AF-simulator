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
        self.last_waiting_record_time = -1  # 上次记录waiting的时间，用于定期采样

        self.round_cost:list[int] = []
        self.A_arrival:list[int] = []
        self.current_A_arrival:int = 0 
        self.A_finish:list[int] = []
        self.F_arrival:list[int] = []
        self.F_finish:list[int] = []
        self.Acost:list[int] = []
        self.Fcost:list[int] = []
        
        # 时间轴数据：记录每个阶段的开始和结束时间
        # 格式: [(stage_type, start_time, end_time, round_id), ...]
        # stage_type: 'A'=Attention, 'C1'=A2F传输, 'F'=FFN, 'C2'=F2A传输, 'W'=等待
        self.timeline: list[tuple] = []
        self.current_round = 0
        
        # 记录batch_size（num_req）随时间的变化
        # 格式: [(time, num_req), ...]
        self.batch_size_history: list[tuple] = []
        # 记录total token count（length）随时间的变化
        # 格式: [(time, length), ...]
        self.token_count_history: list[tuple] = []
        # 记录Attention waiting time随时间的变化
        # 格式: [(time, waiting_time), ...]
        self.attention_waiting_time_history: list[tuple] = []
        # 记录FFN waiting time随时间的变化
        # 格式: [(time, waiting_time), ...]
        self.ffn_waiting_time_history: list[tuple] = []
        # 记录status随时间的变化
        # 格式: [(time, status), ...]
        self.status_history: list[tuple] = []
        

    def load_request(self, current_time, request:Request):
        self.requests.append(request)
        request.start_processing(current_time, self.bids)
        self.length += request.length
        self.num_req += 1
        # 记录batch_size和token_count变化
        self.batch_size_history.append((current_time, self.num_req))
        self.token_count_history.append((current_time, self.length))
        # 调试输出
        if len(self.batch_size_history) % 10 == 0 or self.num_req > 20:  # 每10次或num_req超过20时输出
            print(f"[DEBUG Batch {self.bids}] load_request: time={current_time}, num_req={self.num_req}, "
                  f"ever_served={self.ever_served_request}, batch_size_limit={self.batch_size}")
        if self.status == 0:
            self.status = 5  # Waiting for allocation in attention (与FFN对称)
            # 记录Attention侧等待状态开始
            if len(self.timeline) == 0 or self.timeline[-1][0] != 'WA':
                self.timeline.append(('WA', current_time, -1, 0))
                self.last_waiting_record_time = -1  # Reset for immediate first sample
            # 初始化第一轮
            self.current_round = 0
        

    def finish_request(self, current_time, request:Request)-> bool:
        if request not in self.requests:
            raise ValueError("Request not in batch")
            #return False
        self.ever_served_request += 1
        self.requests.remove(request)
        self.length -= (request.length-1)
        self.num_req -= 1
        # 记录batch_size和token_count变化
        self.batch_size_history.append((current_time, self.num_req))
        self.token_count_history.append((current_time, self.length))
        # 调试输出
        if len(self.batch_size_history) % 10 == 0 or self.num_req < 0:  # 每10次或num_req异常时输出
            print(f"[DEBUG Batch {self.bids}] finish_request: time={current_time}, num_req={self.num_req}, "
                  f"ever_served={self.ever_served_request}, batch_size_limit={self.batch_size}")
        if self.num_req == 0:
            self.status = 0
            #raise ValueError("Ever reached here")
        return True
        
    def record_waiting_time(self, current_time, expected_waiting_time, waiting_type='WA'):
        """
        记录等待时间（该次将会等待多长时间）
        current_time: 当前时间
        expected_waiting_time: 预计等待时间（可以算出来）
        waiting_type: 'WA' for Attention waiting, 'WF' for FFN waiting
        """
        # 检查是否已经记录过这次等待（一次等待只采一次样）
        if self.last_waiting_record_time >= 0:
            # 检查是否还在同一个等待阶段
            if len(self.timeline) > 0 and self.timeline[-1][0] == waiting_type and self.timeline[-1][2] < 0:
                # 还在同一个等待阶段，不重复记录
                return
        
        # 记录等待时间到历史中
        if waiting_type == 'WA':
            self.attention_waiting_time_history.append((current_time, expected_waiting_time))
            print(f"[Batch {self.bids}] Time {current_time}: Attention Expected Waiting Time = {expected_waiting_time} cycles")
        elif waiting_type == 'WF':
            self.ffn_waiting_time_history.append((current_time, expected_waiting_time))
            print(f"[Batch {self.bids}] Time {current_time}: FFN Expected Waiting Time = {expected_waiting_time} cycles")
        
        self.last_waiting_record_time = current_time
    
    def Attention_processing(self, current_time, alpha_A, beta_A):
        # t_A(T)=alpha_A*T+beta_A
        self.status = 1
        current_ending = current_time + alpha_A*self.length + beta_A
        self.current_ending = current_ending

        self.Acost.append(self.current_ending - current_time)
        
        # 如果之前处于等待状态(status=5)，记录等待结束
        if len(self.timeline) > 0 and self.timeline[-1][0] == 'WA':
            wait_start = self.timeline[-1][1]
            waiting_duration = current_time - wait_start
            self.timeline[-1] = ('WA', wait_start, current_time, self.current_round)
            print(f"[Batch {self.bids}] Time {current_time}: Attention Waiting Ended, Actual Waiting Time = {waiting_duration} cycles")
            # 重置last_waiting_record_time，为下一次等待做准备
            self.last_waiting_record_time = -1
        
        # 记录Attention处理阶段到timeline
        self.timeline.append(('A', current_time, self.current_ending, self.current_round))

        return self.current_ending

    def FFN_processing(self, current_time, alpha_F, beta_F) -> int:
        # t_F(T)=alpha_F*T+beta_F
        self.status = 2
        current_ending = current_time + alpha_F*self.num_req + beta_F
        self.current_ending = current_ending

        self.Fcost.append(self.current_ending-current_time)
        self.timeline.append(('F', current_time, self.current_ending, self.current_round))
        return self.current_ending

    def A2F_transmission(self, current_time, alpha_T, beta_T):
        # t_T(T)=alpha_T*T+beta_T
        self.status = 3
        current_ending = current_time + alpha_T*self.length + beta_T
        self.current_ending = current_ending
        
        self.A_finish.append(current_time)
        # 记录时间轴：A2F传输
        self.timeline.append(('C1', current_time, self.current_ending, self.current_round)) 

    def F2A_transmission(self, current_time, alpha_T, beta_T):
        # t_T(T)=alpha_T*T+beta_T
        self.status = 4
        current_ending = current_time + alpha_T*self.length + beta_T
        self.current_ending = current_ending

        self.F_finish.append(current_time)
        # 记录时间轴：F2A传输
        self.timeline.append(('C2', current_time, self.current_ending, self.current_round))
        

    def F2A_transmission_end(self,current_time):
        self.status = 5  # Waiting for allocation in attention (与FFN对称)
        self.A_arrival.append(current_time) # One longer since final round
        self.round_cost.append((current_time-self.current_A_arrival))
        self.current_A_arrival = current_time
        # 记录Attention侧等待状态开始
        if len(self.timeline) == 0 or self.timeline[-1][0] != 'WA':
            self.timeline.append(('WA', current_time, -1, self.current_round))
            self.last_waiting_record_time = -1  # Reset for immediate first sample
        # 新的一轮开始
        self.current_round += 1

    def A2F_transmission_end(self,current_time):
        # 不在这里设置status=6，而是在FFN worker检查buffer时设置
        self.F_arrival.append(current_time)

    def do_new_round(self, current_time, stats):
        self.collect_makespan(current_time)
        for request in self.requests:
            flag = request.do_new_round(current_time, stats)
            if flag:
                self.length += 1
                # 记录token_count变化（每个request生成新token）
                self.token_count_history.append((current_time, self.length))
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