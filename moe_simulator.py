"""
独立 MoE 调度效率模拟器

只模拟 generator + MoE FFN 两层. 用于对比五种调度策略:
  fcfs      : 每个 expert 一个 FIFO 队列
  level     : 每个 expert 拆 5 个子队列, 按 remaining_experts 优先级
  level_c   : level + 拥塞方案 1 (按 expert 服务次数计阈值)
  level_t   : level + 拥塞方案 2 (按 cycle 计 time bound)
  sebf      : Smallest Effective Bottleneck First, bottleneck = 剩余 expert queue 总长度 max

每个 expert 类型有 10 个 worker, 共 8 种 expert => 80 workers.
单 task 处理时间 = 1 cycle. 每 request 随机 sample 4 种 distinct expert.

使用:
  python3 moe_sim.py --strategy=fcfs    --gen=uniform --rate=20 --cycles=10000
  python3 moe_sim.py --strategy=level
  python3 moe_sim.py --strategy=level_c --c2=20 --c3=50 --c4=100
  python3 moe_sim.py --strategy=level_t
  python3 moe_sim.py --strategy=sebf
"""

import random
import argparse
import os
from collections import OrderedDict, deque, defaultdict
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------
class Request:
    __slots__ = (
        'rid', 'generated_time', 'completion_time',
        'expert_ids', 'remaining_experts', 'extra_reduce',
        'dispatch_time', 'time_bound', 'task_locations',
    )

    def __init__(self, rid, generated_time, expert_ids):
        self.rid = rid
        self.generated_time = generated_time
        self.completion_time = -1
        self.expert_ids = expert_ids                # length 4, distinct
        self.remaining_experts = len(expert_ids)
        self.extra_reduce = 0
        self.dispatch_time = generated_time
        self.time_bound = 0
        # eid -> sub_q_level (level 系列用) 或 1 (sebf 用) 的反向索引
        # FCFS 不用; 表示 "req 在该 eid 上还有 task 未处理"
        self.task_locations = {}

    @property
    def priority_level(self):
        """remaining 减 extra_reduce, 下限 0."""
        return max(0, self.remaining_experts - self.extra_reduce)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class Generator:
    """三种分布选一种: fixed / uniform / poisson / geometric. 均值都是 rate."""

    def __init__(self, rate=20, num_experts=8, distribution='fixed', seed=42, gauss_sigma=None):
        self.rate = rate
        self.num_experts = num_experts
        self.distribution = distribution
        self.rng = random.Random(seed)
        self.next_rid = 0

        self.gauss_sigma = gauss_sigma if gauss_sigma is not None else rate / 3.0

    def _sample_count(self):
        if self.distribution == 'fixed':
            return self.rate
        elif self.distribution == 'uniform':
            # uniform [0, 2*rate], 期望 = rate
            return self.rng.randint(0, 2 * self.rate)
        elif self.distribution == 'poisson':
            # 用 Knuth 算法采样 (rate < 30 时可接受)
            L = pow(2.71828182845904523, -self.rate)
            k = 0
            p = 1.0
            while True:
                k += 1
                p *= self.rng.random()
                if p <= L:
                    return k - 1
        elif self.distribution == 'geometric':
            # 几何分布: 每个 cycle 概率 p 失败一次, 期望 = rate 个 request
            # 直接从 Geometric(p) 采样 rate 次然后求和? 不, 简化:
            # 几何分布 (在两次"事件"之间的间隔) 期望 = rate, p = 1/rate
            # 这里我们想生成 expectation=rate 个请求, 用泊松更自然.
            # 简化处理: 用 negative binomial 近似
            # 实际只是想要"非均匀波动", 用 geometric scaling: 50% 概率生成 rate 个, 50% 生成 0 个? 太粗暴.
            # 折中: 每个 cycle 生成 Geom(1/rate)*rate 数量个 — 期望 rate, 方差大.
            return min(int(self.rng.expovariate(1.0 / self.rate)), 10 * self.rate)
        elif self.distribution == 'gaussian':
            # 正态分布 N(rate, sigma^2), 取 round 并 clip 到 [0, 10*rate]
            x = self.rng.gauss(self.rate, self.gauss_sigma)
            return max(0, min(int(round(x)), 10 * self.rate))
        else:
            raise ValueError(f"unknown distribution: {self.distribution}")

    def step(self, current_time):
        n = self._sample_count()
        out = []
        for _ in range(n):
            expert_ids = self.rng.sample(range(self.num_experts), 4)
            req = Request(self.next_rid, current_time, expert_ids)
            self.next_rid += 1
            out.append(req)
        return out


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
class Stats:
    def __init__(self, num_experts, args):
        self.args = args
        self.num_experts = num_experts
        self.completed_count = 0
        self.total_latency = 0          # sum of (completion - generated)
        self.completed_latencies = []   # 每个完成 request 的 latency, 供 outlier 统计
        # expert 计数: 生成版 / 完成版
        self.expert_needed_at_gen = [0] * num_experts
        self.expert_needed_at_done = [0] * num_experts
        # 拥塞触发计数
        self.starve_c_total = 0
        self.starve_c_per_level = {2: 0, 3: 0, 4: 0}
        self.starve_t_total = 0
        self.strategy = args.strategy

    def on_generate(self, req):
        for eid in req.expert_ids:
            self.expert_needed_at_gen[eid] += 1

    def on_request_complete(self, req):
        self.completed_count += 1
        latency = req.completion_time - req.generated_time
        self.total_latency += latency
        self.completed_latencies.append(latency)
        for eid in req.expert_ids:
            self.expert_needed_at_done[eid] += 1

    def avg_latency(self):
        if self.completed_count == 0:
            return None
        return self.total_latency / self.completed_count

    def count_outliers(self, queues, current_time):
        """统计 latency >= 2 * avg 的 request 数 (已完成 + 未完成)."""
        if self.completed_count == 0:
            return None
        avg = self.total_latency / self.completed_count
        threshold = 2 * avg

        completed_outliers = sum(1 for L in self.completed_latencies if L >= threshold)

        """统计 latency >= 5 * avg 的 request 数 (已完成 + 未完成)."""
        
        long_threshold = 5 * avg
        dead_threashold = 20* avg

        completed_long_outliers = sum(1 for L in self.completed_latencies if L >= long_threshold)
        completed_dead_outliers = sum(1 for L in self.completed_latencies if L >= dead_threashold)
        seen = set()
        pending_outliers = 0
        pending_long_outliers = 0
        pending_dead_outliers = 0
        for q in queues:
            if isinstance(q, FCFSQueue):
                for (req, _eid) in q.q:
                    if req.rid in seen:
                        continue
                    seen.add(req.rid)
                    if current_time - req.generated_time >= threshold:
                        pending_outliers += 1
                    if current_time - req.generated_time >= long_threshold:
                        pending_long_outliers += 1
                    if current_time - req.generated_time >= dead_threashold:
                        pending_dead_outliers += 1
            elif isinstance(q, SEBFQueue):
                for (req, _eid, _eqt) in q.q:
                    if req.rid in seen:
                        continue
                    seen.add(req.rid)
                    if current_time - req.generated_time >= threshold:
                        pending_outliers += 1
                    if current_time - req.generated_time >= long_threshold:
                        pending_long_outliers += 1
                    if current_time - req.generated_time >= dead_threashold:
                        pending_dead_outliers += 1
            else:
                for sub in q.sub_q:
                    for (rid, _eid), (req, _e) in sub.items():
                        if rid in seen:
                            continue
                        seen.add(rid)
                        if current_time - req.generated_time >= threshold:
                            pending_outliers += 1
                        if current_time - req.generated_time >= long_threshold:
                            pending_long_outliers += 1
                        if current_time - req.generated_time >= dead_threashold:
                            pending_dead_outliers += 1

        return {
            'threshold': threshold,
            'long_threshold': long_threshold,
            'dead_threshold': dead_threashold,
            'completed_outliers': completed_outliers,
            'completed_long_outliers': completed_long_outliers,
            'completed_dead_outliers': completed_dead_outliers,
            'pending_outliers':   pending_outliers,
            'pending_long_outliers':   pending_long_outliers,
            'pending_dead_outliers':   pending_dead_outliers,
            'total_outliers':     completed_outliers + pending_outliers,
            'total_long_outliers':     completed_long_outliers + pending_long_outliers,
            'total_dead_outliers':     completed_dead_outliers + pending_dead_outliers,
        }

    def report(self, queues=None, current_time=None):
        avg = self.avg_latency()
        print("\n=== STATS ===")
        print(f"Completed requests: {self.completed_count}")
        print(f"Avg latency:        {avg:.3f}" if avg is not None else "Avg latency: N/A")
        print(f"Expert needed (at generation):")
        for eid, n in enumerate(self.expert_needed_at_gen):
            print(f"  expert {eid}: {n}")
        print(f"Expert needed (at completion):")
        for eid, n in enumerate(self.expert_needed_at_done):
            print(f"  expert {eid}: {n}")
        if self.starve_c_total > 0:
            print(f"Starve-C (方案 1) total: {self.starve_c_total}")
            print(f"  per-level: {dict(self.starve_c_per_level)}")
        if self.starve_t_total > 0:
            print(f"Starve-T (方案 2) total: {self.starve_t_total}")
        if queues is not None and current_time is not None:
            out = self.count_outliers(queues, current_time)
            if out is not None:
                print(f"Outliers (latency >= 2*avg = {out['threshold']:.2f}):")
                print(f"  completed (final latency >= threshold): {out['completed_outliers']}")
                print(f"  pending   (current wait >= threshold):  {out['pending_outliers']}")
                print(f"  total: {out['total_outliers']}")
            
                print(f"Long Outliers (latency >= 5*avg = {out['long_threshold']:.2f}):")
                print(f"  completed (final latency >= long_threshold): {out['completed_long_outliers']}")
                print(f"  pending   (current wait >= long_threshold):  {out['pending_long_outliers']}")
                print(f"  total: {out['total_long_outliers']}")

                print(f"Dead Outliers (latency >= 20*avg = {out['dead_threshold']:.2f}):")
                print(f"  completed (final latency >= dead_threshold): {out['completed_dead_outliers']}")
                print(f"  pending   (current wait >= dead_threshold):  {out['pending_dead_outliers']}")
                print(f"  total: {out['total_dead_outliers']}")

    def to_dict(self, queues=None, current_time=None, extra_info=None):
        """汇总成 dict, 供 json.dump 使用. extra_info 可包含 strategy/rate/cycles 等."""
        out = {
            "completed_requests": self.completed_count,
            "avg_latency": self.avg_latency(),
            "expert_needed_at_generation": list(self.expert_needed_at_gen),
            "expert_needed_at_completion": list(self.expert_needed_at_done),
            "starve_c_total": self.starve_c_total,
            "starve_c_per_level": {str(k): v for k, v in self.starve_c_per_level.items()},
            "starve_t_total": self.starve_t_total,
        }
        if queues is not None and current_time is not None:
            outlier = self.count_outliers(queues, current_time)
            out["outliers"] = outlier
        if extra_info is not None:
            out["config"] = extra_info
        return out

    def dump_json(self, path, queues=None, current_time=None, extra_info=None):
        import json
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = self.to_dict(queues=queues, current_time=current_time, extra_info=extra_info)
        #output_path = os.path.join(path, f"{self.strategy}_records.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Stats dumped to {path}")


# ---------------------------------------------------------------------------
# FCFS Queue (策略 1)
# ---------------------------------------------------------------------------
class FCFSQueue:
    """单 expert 的简单 FIFO 队列, 装 (req, eid) tuple."""

    def __init__(self, expert_id):
        self.expert_id = expert_id
        self.q = deque()

    def add_task(self, req, eid):
        self.q.append((req, eid))

    def pop_task(self):
        if not self.q:
            return None
        return self.q.popleft()

    def total_len(self):
        return len(self.q)


# ---------------------------------------------------------------------------
# SEBF Queue (策略 5: Smallest Effective Bottleneck First)
# ---------------------------------------------------------------------------
class SEBFQueue:
    """单 expert 的 SEBF 队列.

    优化策略: 每 cycle 主循环开头调一次 sort_for_cycle(all_queues), 按
    (bottleneck, enqueue_time) 排序; 本 cycle 内的 pop 直接按排序结果取.
    "实时" 解读为 "cycle 内静态 + cycle 间动态" - bottleneck 用 cycle 开始时各 queue
    长度算, cycle 内不变; cycle 之间 queue 长度变化都会反映到下次排序.

    bottleneck = max(req 剩余未完成 eid 对应 queue 的总长度).
    同 bottleneck 时按 enqueue_time FCFS.
    """

    def __init__(self, expert_id):
        self.expert_id = expert_id
        self.q = []                # list of (req, eid, enqueue_time)
        self._counter = 0
        self._sorted = []
        self._sorted_idx = 0

    def add_task(self, req, eid):
        self.q.append((req, eid, self._counter))
        self._counter += 1
        # 复用 task_locations 跟踪 "req 在此 eid 上有 task 未处理"
        # value 设 1 (SEBF 不需要 level, 只需要 eid 是否存在)
        req.task_locations[eid] = 1

    def sort_for_cycle(self, all_queues):
        """每 cycle 主循环开头调一次. 按 (bottleneck, enqueue_time) 排序."""
        def key(item):
            req, _eid, eqt = item
            remaining_eids = req.task_locations.keys()
            if not remaining_eids:
                return (float('inf'), eqt)
            bottleneck = max(all_queues[e].total_len() for e in remaining_eids)
            return (bottleneck, eqt)
        self._sorted = sorted(self.q, key=key)
        self._sorted_idx = 0

    def pop_task(self):
        """从 _sorted 取下一个. 已被取走的 task 自动跳过."""
        while self._sorted_idx < len(self._sorted):
            item = self._sorted[self._sorted_idx]
            self._sorted_idx += 1
            req, eid, _eqt = item
            # eid 不在 task_locations 说明已被前面 pop 消费过, 跳过
            if eid not in req.task_locations:
                continue
            try:
                self.q.remove(item)
            except ValueError:
                continue
            req.task_locations.pop(eid, None)
            return (req, eid)
        return None

    def total_len(self):
        return len(self.q)


# ---------------------------------------------------------------------------
# Priority Queue (策略 2/3/4)
# ---------------------------------------------------------------------------
class PriorityExpertQueue:
    """单 expert 的优先队列, 5 个子队列 sub_q[0..4].

    sub_q[0] = 救援队列 (方案 2 触发, 不在小 simulator 用; 留接口)
    sub_q[1..4] = priority_level 对应的 FCFS 队列, 1 最紧迫

    sub_q 实现: OrderedDict (FCFS + O(1) 任意删除).

    可选拥塞: c_threshold {2: c2, 3: c3, 4: c4} 即开方案 1.
    """

    NUM_LEVELS = 5

    def __init__(self, expert_id, c_threshold=None):
        self.expert_id = expert_id
        self.sub_q = [OrderedDict() for _ in range(self.NUM_LEVELS)]
        self.served_count = 0
        self.last_served = [0] * self.NUM_LEVELS
        self.c_threshold = c_threshold

    def add_task(self, req, eid, level=None):
        if level is None:
            level = req.priority_level
        key = (req.rid, eid)
        self.sub_q[level][key] = (req, eid)
        req.task_locations[eid] = level

    def move_task(self, req, eid, new_level):
        old_level = req.task_locations.get(eid)
        if old_level is None or old_level == new_level:
            return
        key = (req.rid, eid)
        item = self.sub_q[old_level].pop(key, None)
        if item is None:
            return
        self.sub_q[new_level][key] = item
        req.task_locations[eid] = new_level

    def pop_task(self):
        """返回 (task, starve_level_or_None).
        task = (req, eid) 或 None.
        starve_level_or_None: 触发方案 1 时是 k(2/3/4), 否则 None.
        """
        if self.sub_q[0]:
            return self._pop_from(0), None

        if self.c_threshold is not None:
            for k in (2, 3, 4):
                c_k = self.c_threshold.get(k)
                if c_k is None:
                    continue
                if self.sub_q[k] and (self.served_count - self.last_served[k]) >= c_k:
                    return self._pop_from(k), k

        for k in (1, 2, 3, 4):
            if self.sub_q[k]:
                return self._pop_from(k), None

        return None, None

    def _pop_from(self, k):
        key, (req, eid) = self.sub_q[k].popitem(last=False)
        req.task_locations.pop(eid, None)
        self.served_count += 1
        self.last_served[k] = self.served_count
        return (req, eid)

    def total_len(self):
        return sum(len(q) for q in self.sub_q)

    def per_level_lens(self):
        return [len(q) for q in self.sub_q]


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
class Worker:
    """单 worker. 处理一个 task 用 1 cycle (current_ending = load_time + 1)."""

    __slots__ = ('worker_id', 'expert_id', 'busy', 'ending', 'cur_req', 'cur_eid')

    def __init__(self, worker_id, expert_id):
        self.worker_id = worker_id
        self.expert_id = expert_id
        self.busy = False
        self.ending = -1
        self.cur_req = None
        self.cur_eid = -1

    def load(self, current_time, req, eid):
        self.busy = True
        self.cur_req = req
        self.cur_eid = eid
        self.ending = current_time   # 单 task 处理时间 = 1 cycle, 本 cycle 末完成

    def tick(self, current_time, on_done_callback):
        """完成时调 on_done_callback(req, eid). 返回是否本 cycle 完成了 task."""
        if not self.busy:
            return False
        # 完成判定: current_time >= ending (单 task = 1 cycle, load 那个 cycle 末就完成)
        if current_time < self.ending:
            return False
        req = self.cur_req
        eid = self.cur_eid
        self.busy = False
        self.cur_req = None
        self.cur_eid = -1
        on_done_callback(req, eid)
        return True


# ---------------------------------------------------------------------------
# Scheduler 主逻辑
# ---------------------------------------------------------------------------
class MoESimulator:
    def __init__(self, args):
        self.args = args
        self.num_experts = args.num_experts
        self.workers_per_expert = args.workers_per_expert
        self.strategy = args.strategy
        self.cycles = args.cycles

        # 生成器
        self.gen = Generator(rate=args.rate, num_experts=self.num_experts,
                             distribution=args.gen, seed=args.seed)

        # 统计
        self.stats = Stats(self.num_experts, args)

        # 队列
        if self.strategy == 'fcfs':
            self.queues = [FCFSQueue(eid) for eid in range(self.num_experts)]
        elif self.strategy == 'sebf':
            self.queues = [SEBFQueue(eid) for eid in range(self.num_experts)]
        elif self.strategy == 'level':
            self.queues = [PriorityExpertQueue(eid) for eid in range(self.num_experts)]
        elif self.strategy == 'level_c':
            ct = {2: args.c2, 3: args.c3, 4: args.c4}
            self.queues = [PriorityExpertQueue(eid, c_threshold=ct)
                           for eid in range(self.num_experts)]
        elif self.strategy == 'level_t':
            # 方案 2: 不开方案 1 的 c_threshold
            self.queues = [PriorityExpertQueue(eid) for eid in range(self.num_experts)]
        else:
            raise ValueError(f"unknown strategy: {self.strategy}")

        # workers
        self.workers = []
        self.workers_by_expert = defaultdict(list)
        wid = 0
        for eid in range(self.num_experts):
            for _ in range(self.workers_per_expert):
                w = Worker(wid, eid)
                self.workers.append(w)
                self.workers_by_expert[eid].append(w)
                wid += 1

        # 等待 dispatch 的新 request 队列 (本 cycle 生成的 req)
        self.pending_requests = []

    # -----------------------------------------------------------------
    # 任务回调: worker 完成一个 task
    # -----------------------------------------------------------------
    def _on_task_done(self, req, eid):
        req.remaining_experts -= 1
        # 主动迁移: 仅 level 系列做 (sebf 通过 cycle 头部排序自然处理, fcfs 不需要)
        if self.strategy in ('level', 'level_c', 'level_t'):
            new_level = req.priority_level
            if req.remaining_experts > 0:
                snapshot = dict(req.task_locations)
                for other_eid, _old in snapshot.items():
                    self.queues[other_eid].move_task(req, other_eid, new_level)
        if req.remaining_experts == 0:
            # request 完成
            req.completion_time = self.current_time
            self.stats.on_request_complete(req)

    # -----------------------------------------------------------------
    # 方案 2: 每 cycle 扫所有未完成 request, 检查 time bound
    # -----------------------------------------------------------------
    def _check_time_bounds(self):
        # "未完成"指仍在某个 queue 里有 task 的 request.
        # 直接遍历 queues 收集 distinct request.
        seen = set()
        reqs_to_check = []
        for q in self.queues:
            if isinstance(q, PriorityExpertQueue):
                for sub in q.sub_q:
                    for (rid, _eid), (req, _e) in sub.items():
                        if rid in seen:
                            continue
                        seen.add(rid)
                        reqs_to_check.append(req)
        for req in reqs_to_check:
            if req.priority_level == 0:
                continue
            if not req.task_locations:
                continue
            if self.current_time - req.dispatch_time >= req.time_bound:
                req.extra_reduce += 1
                req.dispatch_time = self.current_time   # 重置
                new_level = req.priority_level
                self.stats.starve_t_total += 1
                snapshot = dict(req.task_locations)
                for eid, _old in snapshot.items():
                    self.queues[eid].move_task(req, eid, new_level)

    # -----------------------------------------------------------------
    # 主循环
    # -----------------------------------------------------------------
    def run(self):
        for t in range(self.cycles):
            self.current_time = t

            # 1) Generator 生成新 request, 入队
            new_reqs = self.gen.step(t)
            for req in new_reqs:
                self.stats.on_generate(req)
                # 方案 2: 计算 time_bound (= 2 * 进入的 4 个 expert queue 总长度的 max)
                if self.strategy == 'level_t':
                    max_qlen = max(self.queues[eid].total_len() for eid in req.expert_ids)
                    req.time_bound = max(1, 2 * max_qlen) / self.workers_per_expert  # 至少 1 防 0
                # 入队
                for eid in req.expert_ids:
                    self.queues[eid].add_task(req, eid)

            # 2) 方案 2 触发检查 (在 worker tick 之前, 这样本 cycle 触发可以影响 worker 取任务)
            if self.strategy == 'level_t':
                self._check_time_bounds()

            # 2.5) SEBF: 本 cycle 排一次序, 给后续 pop 用
            if self.strategy == 'sebf':
                for q in self.queues:
                    q.sort_for_cycle(self.queues)

            # 3) Worker 取任务
            for eid in range(self.num_experts):
                q = self.queues[eid]
                for w in self.workers_by_expert[eid]:
                    if w.busy:
                        continue
                    if isinstance(q, FCFSQueue):
                        task = q.pop_task()
                        if task is not None:
                            req, e = task
                            w.load(t, req, e)
                    elif isinstance(q, SEBFQueue):
                        task = q.pop_task()
                        if task is not None:
                            req, e = task
                            w.load(t, req, e)
                    else:
                        task, starve_level = q.pop_task()
                        if task is not None:
                            req, e = task
                            w.load(t, req, e)
                            if starve_level is not None:
                                self.stats.starve_c_total += 1
                                self.stats.starve_c_per_level[starve_level] += 1

            # 4) Worker tick (完成的 task 触发 on_task_done)
            for w in self.workers:
                w.tick(t, self._on_task_done)
            # 诊断 (只 level 系列有 per_level_lens)
            if self.strategy in ('level', 'level_c', 'level_t'):
                if t % 1000 == 0:
                    print(f"[cycle {t}] expert queues:")
                    for eid in range(self.num_experts):
                        lens = self.queues[eid].per_level_lens()
                        if any(lens):
                            print(f"  expert {eid}: sub_q={lens}, "
                                f"served={self.queues[eid].served_count}")

            # 5) 定期诊断
            if t % max(1, self.cycles // 10) == 0:
                total_queued = sum(q.total_len() for q in self.queues)
                print(f"[cycle {t}] queued tasks = {total_queued}, "
                      f"completed = {self.stats.completed_count}")

        self.stats.report(queues=self.queues, current_time=self.cycles - 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--strategy', choices=['fcfs', 'sebf', 'level', 'level_c', 'level_t'],
                   default='fcfs')
    p.add_argument('--gen', choices=['fixed', 'uniform', 'poisson', 'geometric', 'gaussian'],
                   default='fixed', help='generator distribution (mean = --rate)')
    p.add_argument('--rate', type=int, default=20, help='expected requests / cycle')
    p.add_argument('--gauss_sigma', type=float, default=None,
                   help='高斯分布标准差 (仅 --gen=gaussian 用; 默认 rate/3)')
    p.add_argument('--cycles', type=int, default=10000)
    p.add_argument('--num_experts', type=int, default=8)
    p.add_argument('--workers_per_expert', type=int, default=10)
    # 方案 1 阈值
    p.add_argument('--c2', type=int, default=20)
    p.add_argument('--c3', type=int, default=50)
    p.add_argument('--c4', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', type=str, default=None,
                   help="输出 json 文件路径. 不传则自动生成 stats_<strategy>_<gen>_<rate>.json")
    return p.parse_args()


def main():
    args = parse_args()
    sim = MoESimulator(args)
    sim.run()

    # 输出 json
    if args.out is None:
        out_path = f"stats_{args.strategy}_{args.gen}_{args.rate}.json"
    else:
        out_path = f"result/stats_{args.strategy}_{args.out}"
    extra = {
        "strategy": args.strategy,
        "gen": args.gen,
        "rate": args.rate,
        "cycles": args.cycles,
        "num_experts": args.num_experts,
        "workers_per_expert": args.workers_per_expert,
        "c2": args.c2, "c3": args.c3, "c4": args.c4,
        "seed": args.seed,
    }
    sim.stats.dump_json(out_path,
                        queues=sim.queues,
                        current_time=args.cycles - 1,
                        extra_info=extra)


if __name__ == '__main__':
    main()