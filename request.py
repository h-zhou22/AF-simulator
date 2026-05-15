import random
#from stats import StatsCollector

class Request:
    def __init__(self, rid, arrival_time, length, max_possible_length, next_token_prob, seed=42, use_max_length_limit=True, 
                 use_min_length_limit=False, min_length_limit = 4096, fixed_generation_round = False, fixed_generation_len = 20, alpha_L = 1.0, beta_L=1):
        self.rid = rid
        self.arrival = arrival_time
        self.max_possible_length = max_possible_length
        self.next_token_prob = next_token_prob

        self.use_max_length_limit = use_max_length_limit
        self.use_min_length_limit = use_min_length_limit
        self.min_length_limit = min_length_limit
       
        self.fixed_generation_round = fixed_generation_round
        self.fixed_generation_len = fixed_generation_len
        self.use_fixed_final_length = False
        self.fixed_final_length = length + fixed_generation_len

        self.original_len = length
        self.length = length  # current generated length
        self.rounds = 0  # number of rounds processed

        self.proc_end_times = []  # list of processing end times for each round

        self.start_processing_time = None  # time when processing starts
        self.completion_time = None  # time when request is completed
        self.finished = False  # whether the request is finished

        self.rng = random.Random(seed + rid)  # random generator for this request
        self.batch_id = None  # batch id the request is assigned to
        # Statistics
        self.cyc_used = 0  # total cycles used
        # 通常为1，在启用Multitype_req的时候为1-4。其中1为普通, 2有最小输出长度要求, 3为超长, 4为20轮的超长
        # 5 为被agent modify过的情况
        self.req_type = 1

        self.agent_id = -1
        self.agent_belong = -1
        self.predictable = False
        self.actual_type = 0
        
        self.predicted_type = -1
        self.predicted_length = 2147483647
        
        self.use_target_length = False
        self.target_length = 2147483647 # 目标输出长度
        
        self.marked_eviction = False
        self.status = 0 # 0:未开始, 1: 已分配至server, 尚未分配至batch, 2:正在随batch一同处理, 3: 已经被evict， 4：Finished, 5: 正在loading

        self.alpha_L = alpha_L
        self.beta_L = beta_L
        self.loading_finished_time = -1
        # self.loading_time = self.alpha_L * self.length + self.beta_L
        # 初始的request默认已经装填在Batch当中, 不需要load耗时, 但evict之后失去特权
        # 对这些request, 计算时长的时候需要增加初始的load用时
        # ---- MoE ----
        self.is_MoE = False             # generator 设置
        self.expert_ids = []            # 长度 4 的 distinct expert id 列表
        self.completed_experts = 0      # 已完成的 expert 任务数, MoE worker 完成时 +1
        self.remaining_unfinished_experts = 0  # 剩余未完成的 expert 任务数, MoE worker 完成时 -1

        self.extra_reduce = 0          # 拥塞方案 2 累计的虚降数
        self.dispatch_time = -1        # MoE dispatch 时的 current_time, 用于方案 2 计时
        self.time_bound = 0            # 方案 2 的阈值, dispatch 时由 scheduler 计算
        self.task_locations = {}       # eid -> (sub_q_idx) 反向索引, 用于 O(1) 主动迁移

    def do_new_round(self, current_time, stats):
    # Increase the length and decide whether to continue generating tokens
        self.length += 1
        self.rounds += 1
        self.proc_end_times.append(current_time)
        #print("Request ID: {}, Current length: {}")
        # 被evict的情况特殊判断
        if self.marked_eviction:
            self.status = 3
            return False
        if self.use_min_length_limit and self.length < self.min_length_limit:
            return True
        elif self.use_target_length :
            if self.length < self.target_length:
                return True   # 强制继续到 target
            else:
                self.finish_request(current_time, stats)
                return False
        # Agent 使用
        elif self.use_fixed_final_length:
            #print("Fixed final length: {}".format(self.fixed_final_length))
            if self.length < self.fixed_final_length:
                return True
            else:
                self.finish_request(current_time, stats)
                return False
        # Multi-type使用
        elif self.fixed_generation_round :
            #print("Fixed generation rounds: {}".format(self.fixed_generation_len))
            if self.rounds < self.fixed_generation_len:
                return True
            else:
                self.finish_request(current_time, stats)
                return False
        elif self.use_max_length_limit and self.length >= self.max_possible_length:
            print("Max possible length: {}, Length:{}, Use max limitations:{}".format(self.max_possible_length, self.length, self.use_max_length_limit))
            self.finish_request(current_time, stats)
            return False
        elif self.rng.random() < self.next_token_prob:
            return True
        else:
            # Finished request
            self.finish_request(current_time, stats)
            # 统计各项数据，加入统计队列等待statistic worker处理
            # print("Request ID: {}, Finished cycle: {}".format(self.rid, current_time))
            return False

    def finish_request(self, current_time, stats):
        self.completion_time = current_time
        print("Request {} finished at time {}, org length {}, length {}".format(self.rid, current_time, self.original_len, self.length))
        print("Actual type {}, Predicted type {}".format(self.actual_type, self.predicted_type))
        self.status = 4
        self.finished = True

        self.count_statistics(stats)

    def start_processing(self, current_time, batch_id):
        self.start_processing_time = current_time
        self.batch_id = batch_id

    def count_statistics(self, stats):
        # information collecion
        self.status = 4
        self.finished = True

        self.count_statistics(stats)

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
        self.marked_eviction = True

    def loading_to_Batch_buffer(self, current_time):
        # if self.status == 0:
        #     self.loading_finished_time = current_time
        #     print("Current time:{}".format(current_time))
        #     raise Exception("Initial request should be directly appended using append_request in Batch")
        # else:
        self.loading_finished_time = current_time + (self.alpha_L*self.length + self.beta_L)
        self.status = 5

    @property
    def priority_level(self):
        """调度优先级标签. 0 = 最紧迫 (sub_q[0]); 4 = 最低优先级."""
        return max(0, self.remaining_unfinished_experts - self.extra_reduce)

    def print_debug_information(self):
        print("Request ID: {}, Length: {}, Original length: {}, Rounds:{} ".format(self.rid, self.length, self.original_len,self.rounds))
        print("Batch: {}, Agent type:{}, Predicted type:{}, Status:{}".format(self.batch_id, self.actual_type, self.predicted_type, self.status))

    
    

