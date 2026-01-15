import random
from request import Request

class UniformGenerator:
    """
    以固定速度、特定时间分布生成 Request。
    initial length 均匀分布在 [1, max_length]
    """

    def __init__(self, 
                next_token_prob: float = 0.99,
                seed=42, 
                rate=1, 
                max_length=4096, 
                num_per_cyc = 1, 
                maximal_generation = 10000
                ):
        """
        :param rate: 每多少个 cycle 生成一个 request（例如 rate=5 表示每 5 cycle 生成一个）
        :param max_length: 最大 request 初始长度
        """
        self.next_token_prob = next_token_prob 
        self.rng = random.Random(seed)
        self.seed = seed
        self.rate = rate
        self.num_per_cyc = num_per_cyc  
        self.max_length = max_length
        self.next_request_id = 0  # 全局自增 ID
        self.global_time = 0
        self.gen_tot = 0
        self.maximal_generation = maximal_generation
        

    def generate_length(self):
        """
        生成一个均匀分布的 request 长度
        """
        return self.rng.randint(1, self.max_length)

    def step(self, global_time):
        """
        根据当前 global_time 判断是否生成新 request。
        如果该 cycle 生成，则返回一个 Request 实例；
        否则返回 None。
        """
        self.global_time = global_time
        if global_time % self.rate != 0:
            return None
        requests = []
        #if self.num_per_cyc > 1:
        
        for _ in range(self.num_per_cyc):
                if self.gen_tot >= self.maximal_generation:
                    break
                length = self.generate_length()
                new_req = Request(rid=self.next_request_id, arrival_time=global_time, length=length, max_possible_length=self.max_length, attention_rate=1, next_token_prob=self.next_token_prob, seed=self.seed)
                new_req.generated_time = global_time
                self.next_request_id += 1

                requests.append(new_req)
                self.gen_tot += 1
                
        return requests
    

class UniformRandomGenerator:
    """
    以固定速度、固定概率生成 Request。
    initial length 均匀分布在 [1, max_length]
    """

    def __init__(self, 
                next_token_prob: float = 0.99,
                seed=42, 
                rate=1, 
                max_length=4096, 
                num_per_cyc = 1, 
                maximal_generation = 10000,
                basic_length = 0
                ):
        """
        :param rate: 每多少个 cycle 生成一个 request（例如 rate=5 表示每 5 cycle 生成一个）
        :param max_length: 最大 request 初始长度
        """
        self.next_token_prob = next_token_prob
        self.rng = random.Random(seed)
        self.seed = seed
        self.rate = rate
        # 同一个cycle以相同的概率多次尝试生成request
        self.num_per_cyc = num_per_cyc  
        self.max_length = max_length
        self.next_request_id = 0  # 全局自增 ID
        self.global_time = 0
        self.gen_tot = 0
        self.maximal_generation = maximal_generation
        
        self.basic_length = basic_length
        
        assert self.basic_length <= self.maximal_generation

    def generate_length(self):
        """
        生成一个均匀分布的 request 长度
        """
        return self.rng.randint(1, self.max_length)

    def step(self, global_time):
        """
        根据当前 global_time 判断是否生成新 request。
        如果该 cycle 生成，则返回一个 Request 实例；
        否则返回 None。
        """
        self.global_time = global_time
        requests = []
        #if self.num_per_cyc > 1:
        
        while self.gen_tot < self.basic_length:
                if self.gen_tot >= self.maximal_generation:
                    break
                
                
                length = self.generate_length()
                new_req = Request(rid=self.next_request_id, arrival_time=global_time, length=length, max_possible_length=self.max_length, attention_rate=1, next_token_prob=self.next_token_prob, seed=self.seed)
                new_req.generated_time = global_time
                self.next_request_id += 1

                requests.append(new_req)
                self.gen_tot += 1

        for _ in range(self.num_per_cyc):
                if self.gen_tot >= self.maximal_generation:
                    break
                
                if self.rng.random() > self.rate:
                    continue
                length = self.generate_length()
                new_req = Request(rid=self.next_request_id, arrival_time=global_time, length=length, max_possible_length=self.max_length, attention_rate=1, next_token_prob=self.next_token_prob, seed=self.seed)
                new_req.generated_time = global_time
                self.next_request_id += 1

                requests.append(new_req)
                self.gen_tot += 1
                
        return requests