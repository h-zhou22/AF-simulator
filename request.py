import random
#from stats import StatsCollector

class Request:
    def __init__(self, rid, arrival_time, length, max_possible_length, next_token_prob, seed=42, 
                 request_mode="default", LI=None, LO=None, p=None, q=None):
        self.rid = rid
        self.arrival = arrival_time
        self.max_possible_length = max_possible_length
        self.next_token_prob = next_token_prob
        
        # 新的模式参数
        self.request_mode = request_mode
        self.LI = LI  # 输入长度（identical模式）
        self.LO = LO  # 输出长度（identical模式）
        self.p = p    # geometric分布概率（输出）
        self.q = q    # geometric分布概率（输入，仅用于geometric_input_output模式）

        self.original_len = length
        self.length = length  # current generated length
        self.rounds = 0  # number of rounds processed
        
        # 用于identical模式：记录已生成的输出token数
        self.generated_output_tokens = 0

        self.proc_end_times = []  # list of processing end times for each round

        self.start_processing_time = None  # time when processing starts
        self.completion_time = None  # time when request is completed
        self.finished = False  # whether the request is finished

        self.rng = random.Random(seed + rid)  # random generator for this request
        self.batch_id = None  # batch id the request is assigned to
        # Statistics
        self.cyc_used = 0  # total cycles used

    
    def do_new_round(self, current_time, stats):
    # Increase the length and decide whether to continue generating tokens
        self.rounds += 1
        self.proc_end_times.append(current_time)
        
        # 根据不同的模式决定是否继续生成token
        should_continue = False
        
        if self.request_mode == "identical":
            # identical模式：固定输出长度LO
            self.length += 1
            self.generated_output_tokens += 1
            if self.generated_output_tokens < self.LO:
                should_continue = True
            else:
                # 完成输出
                should_continue = False
                
        elif self.request_mode == "geometric_output":
            # geometric_output模式：输入是identical的，输出服从Geo(p)
            # 每个时刻有p的概率增长1，1-p的概率变为0（停止）
            self.length += 1
            if self.rng.random() < self.p:
                should_continue = True
            else:
                # 1-p概率停止
                should_continue = False
                
        elif self.request_mode == "geometric_input_output":
            # geometric_input_output模式：输入和输出都是geometric分布
            # 输入用q控制，输出用p控制
            self.length += 1
            if self.rng.random() < self.p:
                should_continue = True
            else:
                # 1-p概率停止
                should_continue = False
        else:
            # default模式：原有逻辑
            self.length += 1
            if self.rng.random() < self.next_token_prob and self.length < self.max_possible_length:
                should_continue = True
            else:
                should_continue = False
        
        if should_continue:
            return True
        else:
            # Finished request
            self.completion_time = current_time
            self.finished = True
            # 统计各项数据，加入统计队列等待statistic worker处理
            self.count_statistics(stats)
            return False

    def start_processing(self, current_time, batch_id):
        self.start_processing_time = current_time
        self.batch_id = batch_id

    def count_statistics(self, stats):
        # information collecion
        self.cyc_used = self.completion_time - self.start_processing_time
        stats.record(self)

    def mark_completion(self, t):
        self.completion_time = t

    def prepare_for_eviction(self):
        self.prepare_for_eviction = True

    
    
    

