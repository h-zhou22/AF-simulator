import random
from request import Request

import random

import random

class PredictionAgent:
    def __init__(self, p=0.5, seed=42, type_weight = [0.6, 0.04, 0.04, 0.04, 0.2, 0.08]):
        """
        :param p: request 是可预测 (predictable) 的概率
        :param seed: 固定随机种子确保实验可重复
        """
        self.p = p
        self.rng = random.Random(seed)
        self.type_weight = type_weight

    def _generate_agent_info(self):
        """
        生成 agent_id 并根据新的权重分配类型 (Modulo 100)
        """
        # 生成 0-1000 之间的随机数
        agent_id = self.rng.randint(0, 199)
        remainder = agent_id % 100
        
        # 根据余数区间分配类型，对应权重: 0.9, 0.01, 0.01, 0.01, 0.05, 0.02
        if 0 <= remainder <= 59:
            agent_type = 0
        elif 60 <= remainder <= 63:
            agent_type = 1
        elif 64<= remainder <= 67:
            agent_type = 2
        elif 68<= remainder <= 71:
            agent_type = 3
        elif 72 <= remainder <= 91:
            agent_type = 4
        else: # 92-99
            agent_type = 5
            
        return agent_id, agent_type

    def _get_target_length(self, actual_type, origin_length):
        """
        根据实际输出轮数确定真实类别
        """
        if actual_type == 0:
            return 0 # 无预测
        elif actual_type == 1:
            return origin_length + 20
        elif actual_type == 2:
            # Final length 1024
            target_length = 1024
            return target_length
        elif actual_type == 3:
            target_length = self.rng.randint(1025, 4096)
            return target_length
        elif actual_type == 4:
            target_length = origin_length + 4096
            return target_length
        else:
            raise ValueError("Invalid actual_type")
        
    def _get_actual_type(self):
        target_type = self.rng.choice([1, 2, 3, 4])
        if  target_type == 1:
            return 1
        elif target_type == 2:
            return 2
        elif target_type == 3:
            return 3
        else:
            return 4

    def process_request(self, request: Request):
        """
        核心处理函数：为 request 注入所有预测相关的属性
        """
        # 1. 身份分配
        agent_id, agent_type = self._generate_agent_info()
        request.agent_id = agent_id
        request.agent_belong = agent_type
        
        # 2. 预测属性标注
        request.predictable = self.rng.random() < self.p
        if request.predictable:
            actual_type = self._get_actual_type()
            target_length = self._get_target_length(actual_type, request.original_len)
            request.actual_type = actual_type 
            self.rework_request(request, target_length)
        
        # 3. 模拟 Agent 预测行为
        request.predicted_type = -1
        
        if agent_type == 0:
            pass # 不进行预测
            
        elif agent_type == 1:
            request.predicted_type = 1 # 永远预测最短
            request.predicted_length = request.length + 20
            
        elif agent_type == 2:
            request.predicted_type = 4 # 永远预测最长
            request.predicted_length = request.length + 4096
            
        elif agent_type == 3:
            # 可预测则准确，不可预测则 (1-p)*p 概率随机预测最长/最短
            if request.predictable:
                request.predicted_type = actual_type
                request.predicted_length = self.reasonable_length_guess(request.original_len, actual_type)
            else:
                # 对剩下的request， 以prob p的概率进行随机猜测
                prob = self.p
                if self.rng.random() < prob:
                    request.predicted_type = self.rng.choice([1, 2, 3, 4])
                    request.predicted_length = self.reasonable_length_guess(request.original_len, request.predicted_type)
                    
        elif agent_type == 4:
            # 仅对可预测的请求进行准确预测
            if request.predictable:
                request.predicted_type = actual_type
                request.predicted_length = self.reasonable_length_guess(request.original_len, actual_type)
                
        elif agent_type == 5:
            # 对可预测请求有 80% 准确率
            if request.predictable:
                if self.rng.random() < 0.8:
                    request.predicted_type = actual_type
                    request.predicted_length = self.reasonable_length_guess(request.original_len, actual_type)
                else:
                    # 随机选择预测
                    request.predicted_type = self.rng.choice([1, 2, 3, 4])
                    request.predicted_length = self.reasonable_length_guess(request.original_len, request.predicted_type)

        return request
    
    def reasonable_length_guess(self, origin_length, request_type):
        if request_type == 1:
            return origin_length + 20
        elif request_type == 2:
            return 1024
        elif request_type == 3:
            return 4096
        elif request_type == 4:
            return origin_length + 4096
        else:
            raise ValueError("Invalid request_type")

    # 固定轮数的request需清洗其原有的特征 
    def rework_request(self, request: Request, target_length: int):
        new_up_bound = min(4096, target_length-1)
        new_length = request.original_len
        if 2<= request.actual_type <= 3:
            new_length = self.rng.randint(1, new_up_bound)
        request.original_len = new_length
        request.length = new_length
        request.target_length = target_length
        request.use_target_length = True
        request.use_max_length_limit = False
        request.use_min_length_limit = False
        request.req_type = 5
        
        
class UniformGenerator:
    """
    以固定速度、特定时间分布生成 Request。
    initial length 均匀分布在 [1, max_length]
    """

    def __init__(self, 
                next_token_prob: float = 0.99,
                seed=42, 
                agent_seed = 42,
                rate=1, 
                max_length=4096, 
                num_per_cyc = 1, 
                maximal_generation = 10000,
                use_prediction_agent = False,
                alpha_L = 1.0,
                beta_L = 1.0,
                ):
        """
        :param rate: 每多少个 cycle 生成一个 request（例如 rate=5 表示每 5 cycle 生成一个）
        :param max_length: 最大 request 初始长度
        """
        self.next_token_prob = next_token_prob 
        self.rng = random.Random(seed)
        self.seed = seed
        self.agent_seed = agent_seed
        self.rate = rate
        self.num_per_cyc = num_per_cyc  
        self.max_length = max_length
        self.next_request_id = 0  # 全局自增 ID
        self.global_time = 0
        self.gen_tot = 0
        self.maximal_generation = maximal_generation
        self.use_prediction_agent = use_prediction_agent
        predictable_probability = 0.5
        self.alpha_L = alpha_L
        self.beta_L = beta_L
        self.agent_generator = PredictionAgent(p=predictable_probability, seed = agent_seed)
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
                if self.use_prediction_agent:
                    self.agent_generator.process_request(new_req, length)
                
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
                agent_seed = 42,
                rate=1, 
                max_length=4096, 
                num_per_cyc = 1, 
                maximal_generation = 10000,
                basic_length = 0,
                use_prediction_agent = False,
                alpha_L = 1.0,
                beta_L = 1.0,
                ):
        """
        :param rate: 每多少个 cycle 生成一个 request（例如 rate=5 表示每 5 cycle 生成一个）
        :param max_length: 最大 request 初始长度
        """
        self.next_token_prob = next_token_prob
        self.rng = random.Random(seed)
        self.seed = seed
        self.agent_seed = agent_seed
        self.rate = rate
        # 同一个cycle以相同的概率多次尝试生成request
        self.num_per_cyc = num_per_cyc  
        self.max_length = max_length
        self.next_request_id = 0  # 全局自增 ID
        self.global_time = 0
        self.gen_tot = 0
        self.maximal_generation = maximal_generation

        self.alpha_L = alpha_L
        self.beta_L = beta_L
        
        self.basic_length = basic_length
        self.use_prediction_agent = use_prediction_agent
        
        # print("Basic Length, Maximal Generation: ",self.basic_length, self.maximal_generation)
        assert self.basic_length <= self.maximal_generation
        predictable_probability = 0.5
        self.agent_generator = PredictionAgent(p=predictable_probability, seed = agent_seed)
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
                if self.use_prediction_agent:
                    self.agent_generator.process_request(new_req, length)
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
                agent_seed: int = 42,
                type_seed: int = 1337,        # 决定 type 抽样的独立 RNG 种子
                rate: float = 1.0,
                max_length: int = 4096,       # type 1/2 的初始长度上限
                big_max_length: int = 32768,  # type 3/4 的初始长度上限
                num_per_cyc: int = 1,
                maximal_generation: int = 10000,
                basic_length: int = 0,
                type_probs=(0.7, 0.2, 0.02, 0.08),
                type2_target_length: int = 4096,
                type4_max_rounds: int = 20,
                use_prediction_agent: bool = False,
                alpha_L = 1.0,
                beta_L = 1.0,):
        assert len(type_probs) == 4
        assert all(p >= 0 for p in type_probs)
        assert sum(type_probs) > 0
        assert max_length <= big_max_length
        #对于长request而言, 其初始长度就有可能突破最长生成限制
        #在此情况下, 最短限制优先判断
        #assert basic_length <= maximal_generation

        self.next_token_prob = next_token_prob
        self.seed = seed
        self.agent_seed = agent_seed
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

        self.alpha_L = alpha_L
        self.beta_L = beta_L

        # 生成各类型计数, 仅用于事后查看分布
        self.type_counts = [0, 0, 0, 0]
        self.use_prediction_agent = use_prediction_agent
        predictable_probability = 0.5
        self.agent_generator = PredictionAgent(p=predictable_probability, seed = agent_seed)
    # ---------- 类型抽样 ----------
    def _pick_type(self) -> int:
        return self.type_rng.choices([1, 2, 3, 4], weights=self.type_probs, k=1)[0]

    # ---------- 各类型的 Request 构造 ----------
    def _build_request(self, req_type: int, arrival_time: int) -> Request:
        if req_type == 1:
            length = self.rng.randint(1, self.max_length)
            req = Request(rid=self.next_request_id, arrival_time=arrival_time,
                          length=length, max_possible_length=self.max_length,
                          next_token_prob=self.next_token_prob, seed=self.seed,
                          alpha_L=self.alpha_L, beta_L=self.beta_L)
        elif req_type == 2:
            length = self.rng.randint(1, self.max_length)
            req = Request(rid=self.next_request_id, arrival_time=arrival_time,
                          length=length, max_possible_length=self.max_length,
                          next_token_prob=self.next_token_prob, seed=self.seed,
                          use_max_length_limit=False, use_min_length_limit=True,
                          min_length_limit=self.type2_target_length,
                          alpha_L=self.alpha_L, beta_L=self.beta_L)
        elif req_type == 3:
            length = self.rng.randint(self.max_length, self.big_max_length)
            req = Request(rid=self.next_request_id, arrival_time=arrival_time,
                          length=length, max_possible_length=self.big_max_length,
                          next_token_prob=self.next_token_prob, seed=self.seed,
                          use_max_length_limit=False, use_min_length_limit=False,
                          alpha_L=self.alpha_L, beta_L=self.beta_L)
        elif req_type == 4:
            length = self.big_max_length
            req = Request(rid=self.next_request_id, arrival_time=arrival_time,
                          length=length, max_possible_length=self.big_max_length,
                          next_token_prob=self.next_token_prob, seed=self.seed,
                          use_max_length_limit=False, use_min_length_limit=False,
                          fixed_generation_round= True, fixed_generation_len=self.type4_max_rounds,
                          alpha_L=self.alpha_L, beta_L=self.beta_L)
        else:
            raise ValueError(f"Unknown req_type {req_type}")

        req.req_type = req_type            # 标在 request 上, 方便统计
        req.generated_time = arrival_time
        if self.use_prediction_agent:
                    self.agent_generator.process_request(req)

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
    

class Multitype_Agent_Generator:
    """
    先判定是否使用Agent输出, 如否则执行正常长度输出
    """

    def __init__(self, 
                next_token_prob: float = 0.99,
                seed=42, 
                agent_seed = 44,
                rate=1, 
                max_length=4096, 
                num_per_cyc = 20, 
                maximal_generation = 10000,
                basic_length = 0,
                use_prediction_agent = True,
                alpha_L = 1.0,
                beta_L = 1.0,
                is_MoE = False
                ):
        """
        :param rate: 每多少个 cycle 生成一个 request（例如 rate=5 表示每 5 cycle 生成一个）
        :param max_length: 最大 request 初始长度
        """
        self.next_token_prob = next_token_prob
        self.rng = random.Random(seed)
        self.seed = seed
        self.agent_seed = agent_seed
        self.rate = rate
        # 同一个cycle以相同的概率多次尝试生成request
        self.num_per_cyc = num_per_cyc  
        self.max_length = max_length
        self.next_request_id = 0  # 全局自增 ID
        self.global_time = 0
        self.gen_tot = 0
        self.maximal_generation = maximal_generation

        self.alpha_L = alpha_L
        self.beta_L = beta_L
        
        self.basic_length = basic_length
        self.use_prediction_agent = use_prediction_agent
        
        # print("Basic Length, Maximal Generation: ",self.basic_length, self.maximal_generation)
        assert self.basic_length <= self.maximal_generation
        predictable_probability = 0.5
        self.agent_generator = PredictionAgent(p=predictable_probability, seed = agent_seed)
        self.is_MoE = is_MoE

    def generate_length(self):
        """
        生成一个均匀分布的 request 长度
        """
        whether_gen_long = self.rng.random()
        if whether_gen_long < 0.9:
            larger_than_1024 = self.rng.random()
            if larger_than_1024 > 0.75:
                """Usually, self.max_length=4096"""
                return self.rng.randint(self.max_length//4, self.max_length)
            else:
                return self.rng.randint(1, self.max_length//4)
        else:
            return self.rng.randint(16384, 32768)

    def do_initial_generation(self):
        requests = []
        while self.gen_tot < self.basic_length:
                if self.gen_tot >= self.maximal_generation:
                    break
                
                
                length = self.generate_length()
                new_req = Request(rid=self.next_request_id, arrival_time=0, length=length, max_possible_length=self.max_length,  next_token_prob=self.next_token_prob, use_max_length_limit=False, seed=self.seed)
                new_req.generated_time = 0
                self.next_request_id += 1
                if self.use_prediction_agent:
                    if new_req.length < 16384:
                        self.agent_generator.process_request(new_req)
                if self.is_MoE:
                    new_req.is_MoE = True
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
                if self.is_MoE:
                    new_req.is_MoE = True
                requests.append(new_req)
                self.gen_tot += 1

        for _ in range(self.num_per_cyc):
                if self.gen_tot >= self.maximal_generation:
                    break
                
                if self.rng.random() > self.rate:
                    continue
                length = self.generate_length()
                new_req = Request(rid=self.next_request_id, arrival_time=global_time, length=length, 
                                  max_possible_length=self.max_length, use_max_length_limit=False,  
                                  next_token_prob=self.next_token_prob, seed=self.seed)
                new_req.generated_time = global_time
                if self.use_prediction_agent:
                    if new_req.length < 16384:
                        self.agent_generator.process_request(new_req)
                self.next_request_id += 1
                if self.is_MoE:
                    new_req.is_MoE = True
                requests.append(new_req)
                self.gen_tot += 1
                
        return requests