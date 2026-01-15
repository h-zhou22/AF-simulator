import random
#from stats import StatsCollector

class Request:
    def __init__(self, req_id, arrival_time, length, max_possible_length, next_token_prob, seed=42):
        self.req_id = req_id
        self.arrival = arrival_time
        self.max_possible_length = max_possible_length
        self.next_token_prob = next_token_prob

        self.original_len = length
        self.length = length  # current generated length
        self.rounds = 0  # number of rounds processed

        self.proc_end_times = []  # list of processing end times for each round

        self.start_processing_time = None  # time when processing starts
        self.completion_time = None  # time when request is completed
        self.finished = False  # whether the request is finished

        self.rng = random.Random(seed + req_id)  # random generator for this request
        self.batch_id = None  # batch id the request is assigned to
        # Statistics
        self.cyc_used = 0  # total cycles used

    
    def do_new_round(self, current_time, stats):
    # Increase the length and decide whether to continue generating tokens
        self.length += 1
        self.rounds += 1
        self.proc_end_times.append(current_time)
        if self.rng.random() < self.next_token_prob and self.length < self.max_possible_length:
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

    
    
    

