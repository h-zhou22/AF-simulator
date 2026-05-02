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
                new_req = Request(rid=self.next_request_id, arrival_time=global_time, length=length, max_possible_length=self.max_length,  next_token_prob=self.next_token_prob, seed=self.seed)
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
        
        # print("Basic Length, Maximal Generation: ",self.basic_length, self.maximal_generation)
        assert self.basic_length <= self.maximal_generation

    def generate_length(self):
        """
        生成一个均匀分布的 request 长度
        """
        return self.rng.randint(1, self.max_length)

    def do_initial_generation(self):
        requests = []
        while self.gen_tot < self.basic_length:
                if self.gen_tot >= self.maximal_generation:
                    break
                
                
                length = self.generate_length()
                new_req = Request(rid=self.next_request_id, arrival_time=0, length=length, max_possible_length=self.max_length,  next_token_prob=self.next_token_prob, seed=self.seed)
                new_req.generated_time = 0
                self.next_request_id += 1

                requests.append(new_req)
                self.gen_tot += 1
        return requests

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
                new_req = Request(rid=self.next_request_id, arrival_time=global_time, length=length, max_possible_length=self.max_length,  next_token_prob=self.next_token_prob, seed=self.seed)
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
                new_req = Request(rid=self.next_request_id, arrival_time=global_time, length=length, max_possible_length=self.max_length,  next_token_prob=self.next_token_prob, seed=self.seed)
                new_req.generated_time = global_time
                self.next_request_id += 1

                requests.append(new_req)
                self.gen_tot += 1
                
        return requests
    
class MultitypeRandomGenerator:
    """
    四类 request 混合生成:
      type 1: 普通 — 初始长度 uniform [1, max_length], 默认停止规则 (next_token_prob)
      type 2: 长尾 — 初始长度 uniform [1, max_length], 必须达到 type2_target_length 才允许停
      type 3: 巨型 — 初始长度 big_max_length, 默认停止规则
      type 4: 短促爆发 — 初始长度 uniform [max_length, big_max_length], 恰好 type4_max_rounds 轮
    """

    def __init__(self,
                 next_token_prob: float = 0.99,
                 seed: int = 42,
                 type_seed: int = 1337,        # 决定 type 抽样的独立 RNG 种子
                 rate: float = 1.0,
                 max_length: int = 4096,       # type 1/2 的初始长度上限
                 big_max_length: int = 32768,  # type 3/4 的初始长度上限
                 num_per_cyc: int = 1,
                 maximal_generation: int = 10000,
                 basic_length: int = 0,
                 type_probs=(0.7, 0.2, 0.02, 0.08),
                 type2_target_length: int = 4096,
                 type4_max_rounds: int = 20):
        assert len(type_probs) == 4
        assert all(p >= 0 for p in type_probs)
        assert sum(type_probs) > 0
        assert max_length <= big_max_length
        #对于长request而言, 其初始长度就有可能突破最长生成限制
        #在此情况下, 最短限制优先判断
        #assert basic_length <= maximal_generation

        self.next_token_prob = next_token_prob
        self.seed = seed
        self.rng = random.Random(seed)              # 控制长度抽样
        self.type_rng = random.Random(type_seed)    # 独立控制类型抽样

        self.rate = rate
        self.num_per_cyc = num_per_cyc
        self.max_length = max_length
        self.big_max_length = big_max_length
        self.next_request_id = 0
        self.global_time = 0
        self.gen_tot = 0
        self.maximal_generation = maximal_generation
        self.basic_length = basic_length

        self.type_probs = type_probs
        self.type2_target_length = type2_target_length
        self.type4_max_rounds = type4_max_rounds

        # 生成各类型计数, 仅用于事后查看分布
        self.type_counts = [0, 0, 0, 0]

    # ---------- 类型抽样 ----------
    def _pick_type(self) -> int:
        return self.type_rng.choices([1, 2, 3, 4], weights=self.type_probs, k=1)[0]

    # ---------- 各类型的 Request 构造 ----------
    def _build_request(self, req_type: int, arrival_time: int) -> Request:
        if req_type == 1:
            length = self.rng.randint(1, self.max_length)
            req = Request(rid=self.next_request_id, arrival_time=arrival_time,
                          length=length, max_possible_length=self.max_length,
                          next_token_prob=self.next_token_prob, seed=self.seed)
        elif req_type == 2:
            length = self.rng.randint(1, self.max_length)
            req = Request(rid=self.next_request_id, arrival_time=arrival_time,
                          length=length, max_possible_length=self.max_length,
                          next_token_prob=self.next_token_prob, seed=self.seed,
                          use_max_length_limit=False, use_min_length_limit=True,
                          min_length_limit=self.type2_target_length)
        elif req_type == 3:
            length = self.rng.randint(self.max_length, self.big_max_length)
            req = Request(rid=self.next_request_id, arrival_time=arrival_time,
                          length=length, max_possible_length=self.big_max_length,
                          next_token_prob=self.next_token_prob, seed=self.seed,
                          use_max_length_limit=False, use_min_length_limit=False)
        elif req_type == 4:
            length = self.big_max_length
            req = Request(rid=self.next_request_id, arrival_time=arrival_time,
                          length=length, max_possible_length=self.big_max_length,
                          next_token_prob=self.next_token_prob, seed=self.seed,
                          use_max_length_limit=False, use_min_length_limit=False,
                          fixed_generation_round= True, fixed_generation_len=self.type4_max_rounds)
        else:
            raise ValueError(f"Unknown req_type {req_type}")

        req.req_type = req_type            # 标在 request 上, 方便统计
        req.generated_time = arrival_time
        self.next_request_id += 1
        self.type_counts[req_type - 1] += 1
        return req

    # ---------- 与 UniformRandomGenerator 接口对齐的两个方法 ----------
    def do_initial_generation(self):
        """与 UniformRandomGenerator 同名方法对齐: 一次性生成 basic_length 个 request, 全部 arrival_time=0."""
        requests = []
        while self.gen_tot < self.basic_length:
            if self.gen_tot >= self.maximal_generation:
                break
            req_type = self._pick_type()
            req = self._build_request(req_type, arrival_time=0)
            requests.append(req)
            self.gen_tot += 1
        return requests

    def step(self, global_time):
        self.global_time = global_time
        requests = []

        # 第一段: prefill basic_length (与 UniformRandomGenerator 行为一致)
        while self.gen_tot < self.basic_length:
            if self.gen_tot >= self.maximal_generation:
                break
            req_type = self._pick_type()
            req = self._build_request(req_type, arrival_time=global_time)
            requests.append(req)
            self.gen_tot += 1

        # 第二段: 每个 cycle 按 rate 概率生成
        for _ in range(self.num_per_cyc):
            if self.gen_tot >= self.maximal_generation:
                break
            if self.rng.random() > self.rate:
                continue
            req_type = self._pick_type()
            req = self._build_request(req_type, arrival_time=global_time)
            requests.append(req)
            self.gen_tot += 1

        return requests