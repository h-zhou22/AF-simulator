import random
import heapq
import math
from request import Request
from batch import Batch
from attention import Server
from typing import List, Dict, Tuple
from collections import defaultdict, deque, OrderedDict
from FFN import FFN, dynamic_FFN, MoEFFN
from arranger import GlobalArranger, GreedyArranger, MultitypeArranger
from dataclasses import dataclass

class BasicScheduler:
# AF之间固定匹配不再更改
    def __init__(self, arranger, servers:List[Server], FFN_workers:List[FFN], stats, buffer, stored_batches:Dict[int, Batch], alpha_A, beta_A, alpha_T, beta_T, alpha_F, beta_F, initially_full=True):
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
        self.AF_match : Dict[int, int] = {} # server_id -> FFN_id

        self.initially_full = initially_full
        self.arranger = arranger
        if self.initially_full:
            self.do_initialize_filling()
        self.match_AF()

        

    def do_initialize_filling(self):
        multi_print = True
        if multi_print:
            self.arranger.print_multitype_queue_info()
        tot_batch_size = 0
        for batch in self.stored_batches.values():
            tot_batch_size += batch.batch_size
        if self.arranger.num_req_inque < tot_batch_size:
            print("Basic number:{}, actually needed:{}".format(self.arranger.num_req_inque, tot_batch_size))
            raise ValueError("Not enough requests in the buffer to fill all batches")
        for batch in self.stored_batches.values():
            self.arranger.do_initial_filling(batch)

    def match_AF(self):
        # match each server to a given FFN worker
        for i in range (self.num_servers):
            server = self.servers[i]
            FFN_id = i % self.num_FFN
            FFN_worker = self.FFN_workers[i % self.num_FFN]
            server.FFN_worker = FFN_worker
            self.AF_match[i] = FFN_id

    def do_cycle_work(self, current_time):
            for server_id in range(len(self.servers)):
            # 这里假定了server和FFN的ID是按照顺序排列的，后续Debug的时候需注意
                server = self.servers[server_id]
                FFN_server = self.FFN_workers[self.AF_match[server_id]]
                server.cycle_work(current_time, self.stats, FFN_server, self.alpha_T, self.beta_T)
                if server.compute_memory_usage() > server.memory_capacity + server.dynamic_space:
                    server.evict_requests(current_time)
                evicted_requests = server.evict_out_requests(current_time)
                if evicted_requests:
                    self.arranger.evict_all_requests(evicted_requests)
                    self.stats.record_eviction(len(evicted_requests)) 
            # available_batches : List[Tuple[int, int, int, int]] = []
            # for server in self.servers:
            #     extend_batches = server.find_available_batch()
            #     available_batches.extend(extend_batches)
            test_print_debug = False
            if current_time % 1000 == 0:
                test_print_debug = True
            if test_print_debug:
                print("Cycle: {}".format(current_time))
                # if current_time > 1:
                #     raise ValueError("Cycle work should only be called once")
                for server in self.servers:
                    server.print_debug_information()
                for batch in self.stored_batches.values():
                    batch.print_debug_information()

            self.arranger.arrange_requests(current_time)
            # while available_batches and self.buffer:
            #     request = self.buffer.pop()
            #     best_batch_info = min(available_batches)
            #     batch_id0 = best_batch_info[2]
            #     server_id0 = best_batch_info[3]
            #     best_batch = self.stored_batches[batch_id0]
            #     target_server = self.servers[server_id0]
            #     target_server.load_request_to_batch(current_time, best_batch_info[2], request)
            #     available_batches.remove(best_batch_info)
            #     if best_batch.has_free_slot(current_time):
            #         info0, info1 = best_batch.updated_info(current_time=current_time)
            #         new_info = (info0, info1, batch_id0, server_id0)
            #         available_batches.append(new_info)
            
            for server in self.servers:
                server.attention_work(current_time, self.alpha_A, self.beta_A)
            for FFN_worker in self.FFN_workers:
                FFN_worker.cycle_work(current_time, self.alpha_F, self.beta_F)

# 一组server和FFN worker之间的匹配
class AF_match:
    server_id: int
    server: Server
    FFN_id: int
    FFN_worker: FFN

@dataclass
class level_FFN:
    FFN_id: int
    FFN_worker: 'dynamic_FFN'
    # 应该为几个server服务
    marked_cnt: int
    # 服务的server位于第几层, 如跨层则选取其中较高的为标准
    level: int
    # 是否实际上为多个不同的server服务
    mixed_server_level: bool
    # 实际为几个server服务
    server_cnt: int

# @dataclass
# class unbalanced_server_info:
#     server_id: int
#     FFN_id: int
#     current_mapped_level: int
#     current_size_level: int

class PipelineScheduler:
# AF之间存在固定匹配，可能会被动态修改
    def __init__(self, arranger, servers:List[Server], FFN_workers:List[dynamic_FFN], stats, buffer, stored_batches:Dict[int, Batch], alpha_A, beta_A, alpha_T, beta_T, alpha_F, beta_F, initially_full=True):
        self.servers = servers
        self.arranger = arranger
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
        self.AF_match : Dict[int, int] = {} # server_id -> FFN_id
        self.AF_graph : Dict[int, List[int]] = defaultdict(list) # FFN_id, server_id
        
        # 这里的FFN count在每次floor值更新后都需要维护原有顺序
        self.Batch_FFN_unit_count : Dict[int, float] = {}
        self.server_FFN_unit_count : Dict[int, float] = {}
        self.server_FFN_unit_order : List[Tuple[int, float]] = []

        # 单个FFN worker至多同时为多少个Attention server提供服务, 在此调整
        self.max_AF_ratio = 5
        # 不同AF比的server都有多少
        self.AF_ratio_count : List[int] = [0] * (self.max_AF_ratio + 1)
        # 不同AF比的server的server_id
        self.AF_ratio_list : List[List[int]] = [[] for _ in range(self.max_AF_ratio + 1)]
        
        # self.unbalanced_servers : List[unbalanced_server_info] = []

        self.FFN_lower_bound = 0
        self.FFN_upper_bound = len(self.FFN_workers)
        # 0 For Proper. 1 For Too many FFN, 2 For Too many Attention
        self.currently_utilize_status = 0
        # FFN 不足 (status==2) 时退化为均匀分配 + AF比 swap, 此标志位记录
        self.ffn_insufficient = False

        self.FFN_table = [[] for _ in range(self.max_AF_ratio + 1)]

        self.initially_full = initially_full
        # 先把所有的Attention装满
        if self.initially_full:
            self.do_initialize_filling()
        # TODO: 初始化AF匹配策略以及后续AF匹配的动态调整

        self.do_initial_matching()
        self.print_AF_matching_situation()
        #raise NotImplementedError("Initial Matching testing")
        
    def print_AF_matching_situation(self):
        print("AF matching situation:")
        for server in self.servers:
            print("Server: {}, Mapped FFN (real): {}, Matched FFN: {}".format(server.server_id, server.mapped_FFN_id, self.AF_match[server.server_id]))
            print("Server Level: {}, Server weight: {}".format(server.FFN_level, server.weight))

    def do_initialize_filling(self):
        tot_batch_size = 0
        for batch in self.stored_batches.values():
            tot_batch_size += batch.batch_size
        if self.arranger.num_req_inque < tot_batch_size:
            print("Requests needed: {}".format(tot_batch_size))
            raise ValueError("Not enough requests in the buffer to fill all batches")
        for batch in self.stored_batches.values():
            self.arranger.do_initial_filling(batch)
    # def do_initialize_filling(self):
    #     print("Buffer size, Need to fill",len(self.buffer),self.num_batches*self.batch_size)
    #     if len(self.buffer) < self.num_batches*self.batch_size:
    #         raise ValueError("Buffer size is smaller than total batch capacity")
    #     for batch in self.stored_batches.values():
    #         while batch.num_req < batch.batch_size:
    #             request = self.buffer.pop()
    #             batch.append_request(0, request)
    #             # load_request加入之后会立刻开始处理, 所以初始化都用append
    #         batch.start_processing_from_empty(0)
        
    def do_initial_matching(self):
        # 先记录初始时AF用时比例分配初始资源
        self.do_initial_FFN_unit_recording()
        for server in self.servers:
            corresponding_level = math.floor(self.server_FFN_unit_count[server.server_id])
            if corresponding_level > self.max_AF_ratio:
                corresponding_level = self.max_AF_ratio
            self.AF_ratio_count[corresponding_level] += 1
            self.AF_ratio_list[corresponding_level].append(server.server_id)
        # 再计算恰好匹配的情况下需要的FFN worker的数量区间
        self.update_AF_ratio_bounds()
        # FFN 不足 (status==2): 原分配算法会触发 assert FFN_remaining >= 0,
        # 改走均匀分配 (af_ratio 大的优先进人多组), 运行时用 AF比 swap.
        if self.currently_utilize_status == 2:
            self.ffn_insufficient = True
            print("[PipelineScheduler] FFN insufficient (status==2), "
                  "fall back to uniform assignment + AF-ratio swap.")
            self._uniform_initial_assignment()
            return
        # 标记上一层最后多出来的几个Attention server
        level_remainder = 0
        level_remain_list = []
        FFN_remaining = self.num_FFN
        for level in reversed(range(self.max_AF_ratio + 1)):
            level_cnt  = self.AF_ratio_count[level]
            if level == self.max_AF_ratio:
                group_cnt = level
            else:
                group_cnt = level + 1
            while level_cnt + level_remainder >= group_cnt:
                FFN_id = self.num_FFN - FFN_remaining
                assert FFN_id >= 0
                FFN_remaining -= 1
                assert FFN_remaining >= 0
                server_to_fill = group_cnt - level_remainder
                flag = False
                if level_remainder > 0:
                    flag = True
                    for server_id in level_remain_list:
                        self.AF_match[server_id] = FFN_id
                        self.servers[server_id].map_to_FFN(FFN_id, level)
                        self.AF_graph[FFN_id].append(server_id)
                    
                    level_remainder = 0
                    level_remain_list.clear()

                for i in range(server_to_fill):
                    server_id = self.AF_ratio_list[level][level_cnt-i-1]
                    self.AF_match[server_id] = FFN_id
                    self.servers[server_id].map_to_FFN(FFN_id, level)
                    self.AF_graph[FFN_id].append(server_id)
                if flag:
                    new_level_FFN = level_FFN(FFN_id=FFN_id, FFN_worker=self.FFN_workers[FFN_id], marked_cnt=group_cnt, level=level, mixed_server_level=True, server_cnt=group_cnt)
                    self.FFN_table[level].append(new_level_FFN)
                else:
                    new_level_FFN = level_FFN(FFN_id=FFN_id, FFN_worker=self.FFN_workers[FFN_id], marked_cnt=group_cnt, level=level, mixed_server_level=False, server_cnt=group_cnt)
                    self.FFN_table[level].append(new_level_FFN)
                level_cnt -= server_to_fill
                
            if level_cnt > 0:
            # 余下的归入remainder, 与下一层一同考虑
                level_remainder += level_cnt
                level_remain_list.extend(self.AF_ratio_list[level][:level_cnt])
        
        if FFN_remaining > 0:
            # 构造反向映射: FFN_id -> 它当前服务的 server_id 列表
            # （后面要从某个 FFN "让出"一个 server 给新 FFN）
            # from collections import defaultdict
            # ffn_to_servers: Dict[int, List[int]] = defaultdict(list)
            # for sid, fid in self.AF_match.items():
            #     ffn_to_servers[fid].append(sid)

            # 从高 level 向低 level 依次重分配:
            # 每"轮"取最多 target_group_size 个候选 FFN, 每个让出 1 个 server,
            # 让出的 server 交给 1 个多余 FFN. 这一轮消耗 1 个 FFN_remaining.
            for level in reversed(range(self.max_AF_ratio + 1)):
                if FFN_remaining <= 0:
                    break

                # 本 level 当前的组容量 / 目标组容量
                if level == self.max_AF_ratio:
                    current_group_size = level          # 5
                else:
                    current_group_size = level + 1      # L+1
                target_group_size = current_group_size - 1
                if target_group_size <= 0:
                    continue   # level 0 已经每个 server 一个 FFN, 没法再减

                # 候选: 非 mixed 且当前 server_cnt 还是 current_group_size 的 FFN
                # (本轮新建的 FFN 或上一轮已经被减过的不再作为候选)
                candidates = [
                    lf for lf in self.FFN_table[level]
                    if (not lf.mixed_server_level) and lf.server_cnt == current_group_size
                ]

                idx = 0
                while idx < len(candidates) and FFN_remaining > 0:
                    chunk = candidates[idx: idx + target_group_size]
                    idx += len(chunk)
                    if not chunk:
                        break

                    # 让 chunk 里每个候选各让出 1 个 server
                    shed_servers: List[int] = []
                    for cand in chunk:
                        cand_servers = self.AF_graph[cand.FFN_id]
                        if not cand_servers:
                            continue   # 防御性
                        shed_sid = cand_servers.pop()      # 取最后一个 server 让出
                        shed_servers.append(shed_sid)
                        cand.marked_cnt -= 1
                        cand.server_cnt -= 1

                    if not shed_servers:
                        # chunk 里都没有 server 可让出, 跳过, 不消耗多余 FFN
                        continue

                    # 新建一个 FFN 接管让出的 server
                    new_ffn_id = self.num_FFN - FFN_remaining
                    FFN_remaining -= 1
                    for sid in shed_servers:
                        self.AF_match[sid] = new_ffn_id
                        self.AF_graph[new_ffn_id].append(sid)
                        self.servers[sid].map_to_FFN(new_ffn_id, level)

                    # 注意: 这里 marked_cnt / server_cnt 用实际接管数量
                    # (chunk 不足 target_group_size 时, 新 FFN 只有部分 server)
                    new_lf = level_FFN(
                        FFN_id=new_ffn_id,
                        FFN_worker=self.FFN_workers[new_ffn_id],
                        marked_cnt=len(shed_servers),
                        level=level,
                        mixed_server_level=False,
                        server_cnt=len(shed_servers),
                    )
                    self.FFN_table[level].append(new_lf)
        
        # 用多余的 FFN 反复减小已分配 FFN 的 group_size, 直到用完或无法继续.
        # 每轮 level 从低到高扫描:
        #   - 低 level 先减, 强化 "权重低 -> 服务的 server 少" 的性质
        #   - 即使 FFN_remaining 中途耗尽, 任何时刻状态都保持单调
        if FFN_remaining > 0:
            # 反向索引: FFN_id -> 它当前服务的 server_id 列表
            # 已经使用AF_graph即可

            while FFN_remaining > 0:
                progress = False
                for level in range(self.max_AF_ratio + 1):       # 低 -> 高
                    if FFN_remaining <= 0:
                        break
                    # 该 level 中还可以再减的 FFN (非 mixed, server_cnt >= 2)
                    eligible = [
                        lf for lf in self.FFN_table[level]
                        if (not lf.mixed_server_level) and lf.server_cnt >= 2
                    ]
                    if not eligible:
                        continue

                    # 本轮针对该 level 的"最大组": 把 max_size 减到 max_size - 1
                    max_size = max(lf.server_cnt for lf in eligible)
                    target = max_size - 1
                    if target <= 0:
                        continue
                    candidates = [lf for lf in eligible if lf.server_cnt == max_size]

                    idx = 0
                    while idx < len(candidates) and FFN_remaining > 0:
                        chunk = candidates[idx: idx + target]
                        idx += len(chunk)
                        if not chunk:
                            break

                        # chunk 中每个候选让出 1 个 server
                        shed: List[int] = []
                        for cand in chunk:
                            cand_servers = self.AF_graph[cand.FFN_id]
                            if not cand_servers:
                                continue
                            shed.append(cand_servers.pop())
                            cand.marked_cnt -= 1
                            cand.server_cnt -= 1

                        if not shed:
                            continue

                        # 新建一个 FFN 接管让出的 server
                        new_id = self.num_FFN - FFN_remaining
                        FFN_remaining -= 1
                        for sid in shed:
                            self.AF_match[sid] = new_id
                            self.AF_graph[new_id].append(sid)
                            self.servers[sid].map_to_FFN(new_id, level)

                        new_lf = level_FFN(
                            FFN_id=new_id,
                            FFN_worker=self.FFN_workers[new_id],
                            marked_cnt=len(shed),
                            level=level,
                            mixed_server_level=False,
                            server_cnt=len(shed),
                        )
                        self.FFN_table[level].append(new_lf)
                        progress = True

                if not progress:   # 所有 level 都减不动了, 提前退出
                    break
        
        # 根据最终的 AF_match 把所有 batch 接到对应 FFN 的流水线上
# 此时 batch.mapped_FFN_id 仍为 -1, construct_pipeline 会设它
        for server in self.servers:
            ffn_id = self.AF_match[server.server_id]
            ffn_worker = self.FFN_workers[ffn_id]
            for batch in server.batches.values():
                ffn_worker.construct_pipeline(0, batch)

        for batch in self.stored_batches.values():
            print("Batch ID: ", batch.batch_id)
            print("Matched FFN worker: ", batch.mapped_FFN_id)
            # print("Matched FFN level: ", batch.FFN_level)

    # ------------------------------------------------------------------
    # FFN 不足时的退化路径: 均匀分配 + 基于 AF 比 (t_A/t_F) 的 swap
    # ------------------------------------------------------------------
    def _server_af_ratio(self, server) -> float:
        """AF 比 = t_A / t_F. 越大 = attention 越主导 = 占 FFN 越少 (越友善)."""
        t_A = self.alpha_A * sum(b.length for b in server.batches.values()) + self.beta_A
        t_F = sum(self.alpha_F * max(b.num_req, 1) + self.beta_F
                  for b in server.batches.values())
        if t_F <= 0:
            return float('inf')
        return t_A / t_F

    def _uniform_initial_assignment(self):
        """均匀分配, af_ratio 大的进目标人数更多的组. 之后 construct_pipeline.

        base = num_servers // num_FFN; 前 (num_servers % num_FFN) 个 FFN 目标 base+1.
        af_ratio 降序填, 优先填目标人数多的 FFN.
        """
        base = self.num_servers // self.num_FFN
        extra = self.num_servers % self.num_FFN
        target_capacity = [base + 1 if f < extra else base for f in range(self.num_FFN)]
        ffn_order = sorted(range(self.num_FFN), key=lambda f: -target_capacity[f])

        ranked = sorted(self.servers, key=lambda s: -self._server_af_ratio(s))
        idx = 0
        for f in ffn_order:
            cap = target_capacity[f]
            for _ in range(cap):
                if idx >= len(ranked):
                    break
                server = ranked[idx]
                idx += 1
                self.AF_match[server.server_id] = f
                self.AF_graph[f].append(server.server_id)
                server.map_to_FFN(f, 0)
        # 接到对应 FFN 的流水线 (type=3 用 dynamic_FFN, 需 construct_pipeline)
        for server in self.servers:
            ffn_id = self.AF_match[server.server_id]
            ffn_worker = self.FFN_workers[ffn_id]
            for batch in server.batches.values():
                ffn_worker.construct_pipeline(0, batch)

    def _uniform_swap_balance(self, current_time):
        """FFN 不足时的运行时 swap: 人多组 af_ratio 最小 <-> 人少组 af_ratio 最大,
        差 >= 1 才换, 反复直到不满足. O(n) 每轮."""
        max_iter = 100
        for _ in range(max_iter):
            non_empty = [f for f in range(self.num_FFN) if self.AF_graph[f]]
            if len(non_empty) < 2:
                break
            f_big = max(non_empty, key=lambda f: len(self.AF_graph[f]))
            f_small = min(non_empty, key=lambda f: len(self.AF_graph[f]))
            if f_big == f_small or len(self.AF_graph[f_big]) - len(self.AF_graph[f_small]) <= 0:
                break
            a = min(self.AF_graph[f_big],
                    key=lambda sid: self._server_af_ratio(self.servers[sid]))
            b = max(self.AF_graph[f_small],
                    key=lambda sid: self._server_af_ratio(self.servers[sid]))
            af_a = self._server_af_ratio(self.servers[a])
            af_b = self._server_af_ratio(self.servers[b])
            if af_b - af_a >= 1.0:
                self._uniform_apply_swap(current_time, a, b, f_big, f_small)
                print(f"[pipeline uniform swap cycle {current_time}] swap server {a} "
                      f"(FFN {f_big}, af={af_a:.2f}) <-> server {b} "
                      f"(FFN {f_small}, af={af_b:.2f})")
            else:
                break

    def _uniform_apply_swap(self, current_time, sa, sb, f1, f2):
        """type=3 退化路径下交换两 server 的 FFN 归属.
        用 modify_pipeline + construct_pipeline (dynamic_FFN) 而非 replace_batch."""
        self.AF_match[sa] = f2
        self.AF_match[sb] = f1
        self.AF_graph[f1].remove(sa)
        self.AF_graph[f1].append(sb)
        self.AF_graph[f2].remove(sb)
        self.AF_graph[f2].append(sa)
        # 把 sa 的 batch 从 f1 流水线摘除, 接到 f2; sb 反之
        for batch in self.servers[sa].batches.values():
            self.FFN_workers[f1].modify_pipeline(current_time, batch.batch_id)
        for batch in self.servers[sb].batches.values():
            self.FFN_workers[f2].modify_pipeline(current_time, batch.batch_id)
        for batch in self.servers[sa].batches.values():
            self.FFN_workers[f2].construct_pipeline(current_time, batch)
        for batch in self.servers[sb].batches.values():
            self.FFN_workers[f1].construct_pipeline(current_time, batch)
        self.servers[sa].map_to_FFN(f2, 0)
        self.servers[sb].map_to_FFN(f1, 0)
        self.servers[sa].reactivate_batches(current_time, self.alpha_A, self.beta_A,
                                            self.alpha_T, self.beta_T)
        self.servers[sb].reactivate_batches(current_time, self.alpha_A, self.beta_A,
                                            self.alpha_T, self.beta_T)

    def update_AF_ratio_bounds(self):
        # 再计算恰好匹配的情况下需要的FFN worker的数量区间
        cnt_low = 0
        for i in range(self.max_AF_ratio):
            tot_server = self.AF_ratio_count[i]
            if i == self.max_AF_ratio -1:
                tot_server += self.AF_ratio_count[i+1]
            cnt_low += math.ceil(tot_server / (i+1)) 
        self.FFN_lower_bound = cnt_low
        cnt_high = 0
        for i in range(self.max_AF_ratio):
            tot_server = self.AF_ratio_count[i+1]
            if i == 0:
                tot_server += self.AF_ratio_count[0]
            cnt_high += math.ceil(tot_server / (i+1))
        self.FFN_upper_bound = cnt_high
        if self.num_FFN < self.FFN_lower_bound:
            self.currently_utilize_status =2
        elif self.num_FFN > self.FFN_upper_bound:
            self.currently_utilize_status = 1
        else:
            self.currently_utilize_status = 0

    def do_initial_FFN_unit_recording(self):
        for batch in self.stored_batches.values():
            self.Batch_FFN_unit_count[batch.batch_id] = batch.compute_num_F_unit_time(self.alpha_A, self.beta_A)
        
        self.Batch_FFN_unit_order = sorted(
        self.Batch_FFN_unit_count.items(), key=lambda x: x[1], reverse=True)

        for server in self.servers:
            server.update_FFN_level(self.alpha_A, self.beta_A, self.alpha_F, self.beta_F)
            self.server_FFN_unit_count[server.server_id] = server.weight
        self.server_FFN_unit_order = sorted(
            self.server_FFN_unit_count.items(), key=lambda x: x[1], reverse=True)
    

    def do_cycle_work(self, current_time):
            for server_id in range(len(self.servers)):
            # 这里假定了server和FFN的ID是按照顺序排列的，后续Debug的时候需注意
                server = self.servers[server_id]
                FFN_server = self.FFN_workers[self.AF_match[server_id]]
                server.cycle_work(current_time, self.stats, FFN_server, self.alpha_T, self.beta_T)
                if server.compute_memory_usage() > server.memory_capacity + server.dynamic_space:
                    server.evict_requests(current_time)
                evicted_requests = server.evict_out_requests(current_time)
                if evicted_requests:
                    self.arranger.evict_all_requests(evicted_requests)
                    self.stats.record_eviction(len(evicted_requests)) 
            # available_batches : List[Tuple[int, int, int, int]] = []
            # for server in self.servers:
            #     extend_batches = server.find_available_batch()
            #     available_batches.extend(extend_batches)

            self.arranger.arrange_requests(current_time)
            # while available_batches and self.buffer:
            #     request = self.buffer.pop()
            #     best_batch_info = min(available_batches)
            #     batch_id0 = best_batch_info[2]
            #     server_id0 = best_batch_info[3]
            #     best_batch = self.stored_batches[batch_id0]
            #     target_server = self.servers[server_id0]
            #     target_server.load_request_to_batch(current_time, best_batch_info[2], request)
            #     available_batches.remove(best_batch_info)
            #     if best_batch.has_free_slot(current_time):
            #         info0, info1 = best_batch.updated_info(current_time=current_time)
            #         new_info = (info0, info1, batch_id0, server_id0)
            #         available_batches.append(new_info)
            
            # 在free slot被填补之后，关注各个attention以及各个Batch的大小变化
            for server in self.servers:
                server.update_FFN_level(self.alpha_A, self.beta_A, self.alpha_F, self.beta_F)

            if self.ffn_insufficient:
                # FFN 不足: 走均匀分配 + AF比 swap, 跳过原 level-based swap/relocate
                self._uniform_swap_balance(current_time)
            else:
                exchange_pairs = self.find_swap_pairs()
                for pair in exchange_pairs:
                    self.apply_swap_pair(current_time, pair[0], pair[1])
                    # TODO 在server当中维护两个Batch的归属信息、状态信息
                    self.servers[pair[0]].reactivate_batches(current_time, self.alpha_A, self.beta_A, self.alpha_T, self.beta_T)
                    self.servers[pair[1]].reactivate_batches(current_time, self.alpha_A, self.beta_A, self.alpha_T, self.beta_T)

                self.relocate_unpaired_deviated(current_time)
                self.refresh_mixed_flags()

            for server in self.servers:
                server.attention_work(current_time, self.alpha_A, self.beta_A)
            for FFN_worker in self.FFN_workers:
                FFN_worker.cycle_work(current_time, self.alpha_F, self.beta_F)

    def apply_swap_pair(self, current_time: int, server_id_a: int, server_id_b: int):
        """交换两个 server 各自维护的两个 batch 所归属的 FFN.
        只负责交换,关于Batch交换后状态的维护需要额外的逻辑进行处理"""
        print("Swap {} and {}, Cycle: {}".format(server_id_a, server_id_b, current_time))
        #raise NotImplementedError("Not Swap Has been applied! ")
        server_a = self.servers[server_id_a]
        server_b = self.servers[server_id_b]
        ffn_id_a = self.AF_match[server_id_a]
        ffn_id_b = self.AF_match[server_id_b]
        if ffn_id_a == ffn_id_b:
            return   # 同一 FFN, 不需要 swap
        print("FFN IDa:{}, FFN IDb:{}".format(ffn_id_a, ffn_id_b))
        ffn_a = self.FFN_workers[ffn_id_a]
        ffn_b = self.FFN_workers[ffn_id_b]

        print(f"[swap@{current_time}] s{server_id_a}↔s{server_id_b}, "
          f"ffn_a={ffn_id_a}, ffn_b={ffn_id_b}")
        server_a = self.servers[server_id_a]
        server_b = self.servers[server_id_b]
        print(f"  s{server_id_a}: last={server_a.last_finished_batch_id}, "
            f"first={server_a.first_finished_batch_id}, "
            f"batches={list(server_a.batches.keys())}")
        print(f"  s{server_id_b}: last={server_b.last_finished_batch_id}, "
            f"first={server_b.first_finished_batch_id}, "
            f"batches={list(server_b.batches.keys())}")
        # 检查 server 拥有的 batch 当前 mapped_FFN_id 是否和 AF_match 一致
        for sid, server in [(server_id_a, server_a), (server_id_b, server_b)]:
            expected = self.AF_match[sid]
            for batch in server.batches.values():
                if batch.mapped_FFN_id != expected:
                    print(f"  ! INCONSISTENT: s{sid} expects ffn {expected}, "
                        f"but batch {batch.batch_id} matched to {batch.mapped_FFN_id}")

        # server_a 的两个 batch 全部从 ffn_a 搬到 ffn_b,
        # server_b 的两个 batch 全部从 ffn_b 搬到 ffn_a.
        # 用 swap_batches_between_ffns 一次处理一对 batch (a 的一个 + b 的一个).
        a_batches = list(server_a.batches.values())
        b_batches = list(server_b.batches.values())
        assert len(a_batches) == len(b_batches), \
            f"server batches mismatch: {len(a_batches)} vs {len(b_batches)}"

        

        # 更新 AF_match / AF_graph
        self.AF_match[server_id_a] = ffn_id_b
        self.AF_match[server_id_b] = ffn_id_a
        self.AF_graph[ffn_id_a].remove(server_id_a)
        self.AF_graph[ffn_id_a].append(server_id_b)
        self.AF_graph[ffn_id_b].remove(server_id_b)
        self.AF_graph[ffn_id_b].append(server_id_a)

        # 更新 server 一侧的归属标签
        new_level_a = self._ffn_level_of(ffn_id_b)
        new_level_b = self._ffn_level_of(ffn_id_a)
        server_a.map_to_FFN(ffn_id_b, new_level_a)
        server_b.map_to_FFN(ffn_id_a, new_level_b)

        batch_id_a_last = server_a.last_finished_batch_id
        batch_id_a_first = server_a.first_finished_batch_id
        batch_id_b_last = server_b.last_finished_batch_id
        batch_id_b_first = server_b.first_finished_batch_id
        if batch_id_a_last == -1 or batch_id_b_last == -1:
            batch_list_a = list(server_a.batches.values())
            batch_list_b = list(server_b.batches.values()) 
            batch_id_a_last = batch_list_a[-1].batch_id
            batch_id_b_last = batch_list_b[-1].batch_id
            batch_id_a_first = batch_list_a[0].batch_id
            batch_id_b_first = batch_list_b[0].batch_id
        # 取完 batch_id_a_last/first, batch_id_b_last/first 后
        # 检测退化: 如果 a 或 b 的 first==last, 说明 server 实际只有"一个 batch 跑过状态机"的快照,
        # 退化为按字典顺序取真正不同的两个 batch
        if batch_id_a_first == batch_id_a_last:
            batch_list = list(server_a.batches.values())
            if len(batch_list) >= 2:
                batch_id_a_first = (batch_list[0].batch_id
                                    if batch_list[0].batch_id != batch_id_a_last
                                    else batch_list[1].batch_id)
        if batch_id_b_first == batch_id_b_last:
            batch_list = list(server_b.batches.values())
            if len(batch_list) >= 2:
                batch_id_b_first = (batch_list[0].batch_id
                                    if batch_list[0].batch_id != batch_id_b_last
                                    else batch_list[1].batch_id)
        batch_a_last = self.stored_batches[batch_id_a_last]
        batch_a_first = self.stored_batches[batch_id_a_first]
        batch_b_last = self.stored_batches[batch_id_b_last]
        batch_b_first = self.stored_batches[batch_id_b_first]

        ffn_a.replace_batch(current_time, batch_id_a_last, batch_b_last)
        ffn_b.replace_batch(current_time, batch_id_b_last, batch_a_last)
        ffn_a.replace_batch(current_time, batch_id_a_first, batch_b_first)
        ffn_b.replace_batch(current_time, batch_id_b_first, batch_a_first)

    def relocate_unpaired_deviated(self, current_time):
        """对所有仍处偏离的 server, 找 level 匹配且有空缺的 FFN 单独搬过去.
        swap 之后调用. 找不到目标的 server 跳过."""
        
        for server in self.servers:
            cur_L = server.FFN_level
            tgt_L = self._server_target_level(server)
            if cur_L == tgt_L:
                continue
            target = self._find_relocation_target(tgt_L)
            if target is None:
                continue
            print("We change! ", current_time)
            print("Relocated Server ID", server.server_id)
            self._relocate_server(current_time, server.server_id, target.FFN_id)
    
    def refresh_mixed_flags(self):
        """根据当前 AF_graph 重算每个 FFN 的 mixed_server_level 标记.

        一个 FFN 是 mixed ⇔ 它服务的 server 的 current_mapped_level 不全相同.
        """
        for level_buckets in self.FFN_table:
            for lf in level_buckets:
                servers = self.AF_graph.get(lf.FFN_id, [])
                if not servers:
                    lf.mixed_server_level = False
                    lf.server_cnt = 0
                    continue
                target_levels = {self._server_target_level(self.servers[sid]) for sid in servers}
                lf.mixed_server_level = (target_levels != {lf.level})
                lf.server_cnt = len(servers)

    def _ffn_level_of(self, ffn_id: int) -> int:
        """从 FFN_table 反查 level. swap 不改 level 字段, 此处 ffn_id 一定能找到."""
        for level in range(self.max_AF_ratio + 1):
            for lf in self.FFN_table[level]:
                if lf.FFN_id == ffn_id:
                    return level
        raise ValueError(f"FFN {ffn_id} not found in FFN_table")   

    def _server_target_level(self, server) -> int:
        """该 server 按当前实时 attention/FFN 用时比'本应'所属的 level.

        重要: 不读 server.weight 这个缓存字段 (可能更新不及时), 每次实时算.
        分母用每个 batch 当前 num_req 算的 FFN 用时, 不再依赖 unit_FFN_time 常数.
        """
        server_weight = server.compute_total_unit_cost(
            self.alpha_A, self.beta_A, self.alpha_F, self.beta_F)
        return min(math.floor(server_weight), self.max_AF_ratio)

    def find_swap_pairs(self) -> List[Tuple[int, int]]:
        """返回 [(server_id_a, server_id_b), ...].

        匹配条件: a 当前在 La 层但应该去 Lb 层; b 当前在 Lb 层但应该去 La 层.
        每个 server 至多出现在一对中.
        """
        deviated_by_pair: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for server in self.servers:
            cur_L = server.FFN_level
            tgt_L = self._server_target_level(server)
            if cur_L == tgt_L:
                continue
            deviated_by_pair[(cur_L, tgt_L)].append(server.server_id)

        pairs: List[Tuple[int, int]] = []
        used: set = set()
        for (cur_L, tgt_L), bucket in deviated_by_pair.items():
            if cur_L >= tgt_L:
                continue   # 只从一边遍历, 避免重复
            opposite = deviated_by_pair.get((tgt_L, cur_L), [])
            i = j = 0
            while i < len(bucket) and j < len(opposite):
                sa = bucket[i]
                sb = opposite[j]
                if sa in used:
                    i += 1; continue
                if sb in used:
                    j += 1; continue
                pairs.append((sa, sb))
                used.add(sa); used.add(sb)
                i += 1; j += 1
        return pairs
    
    def _level_capacity(self, level: int, is_mixed: bool) -> int:
        """FFN 当前的可容纳 server 数. 与 do_initial_matching 里 group_cnt 一致."""
        if is_mixed:
            return level                         # 混合 level=L 容量 = L
        if level == self.max_AF_ratio:
            return level                         # 顶层纯 = max
        return level + 1                         # 其他纯 = L+1


    def _level_FFN_of(self, ffn_id: int):
        for level in range(self.max_AF_ratio + 1):
            for lf in self.FFN_table[level]:
                if lf.FFN_id == ffn_id:
                    return lf
        return None


    def _find_relocation_target(self, tgt_L: int):
        """在 level=tgt_L 找一个有空缺的 FFN. 优先纯, 其次空闲多."""
        if tgt_L < 0 or tgt_L > self.max_AF_ratio:
            return None
        best, best_score = None, None
        for lf in self.FFN_table[tgt_L]:
            cap = self._level_capacity(lf.level, lf.mixed_server_level)
            if lf.server_cnt >= cap:
                continue
            spare = cap - lf.server_cnt
            score = (0 if lf.mixed_server_level else 1, spare)   # 纯 > 混合; 空缺多 > 少
            if best is None or score > best_score:
                best, best_score = lf, score
        return best


    def _relocate_server(self, current_time, server_id, dest_ffn_id):
        """把单个 server 的两个 batch 从原 FFN 移到 dest_ffn_id."""
        server = self.servers[server_id]
        src_ffn_id = self.AF_match[server_id]
        if src_ffn_id == dest_ffn_id:
            return

        src_ffn = self.FFN_workers[src_ffn_id]
        dest_ffn = self.FFN_workers[dest_ffn_id]

        # 注意: 不能用 replace_batch (没有对偶 batch 拿来换),
        # 直接 modify_pipeline + construct_pipeline.
        # status==2 的 in-flight batch: 节点从 src 链表摘除后, src FFN 仍按
        #   self.current_ending 计时跑完, batch 自己走 F2A→A→A2F→load_batch(dest).
        # status==6 与 status==4 的 batch: load_ready 状态丢失, 由下面 reactivate_batches 重做 A2F.
        for batch in list(server.batches.values()):
            src_ffn.modify_pipeline(current_time, batch.batch_id)
            dest_ffn.construct_pipeline(current_time, batch)

        # 维护 AF_match / AF_graph
        self.AF_match[server_id] = dest_ffn_id
        self.AF_graph[src_ffn_id].remove(server_id)
        self.AF_graph[dest_ffn_id].append(server_id)

        new_level = self._ffn_level_of(dest_ffn_id)
        server.map_to_FFN(dest_ffn_id, new_level)

        # 维护 FFN_table 计数 (mixed 标记由 refresh_mixed_flags 统一更新)
        src_lf = self._level_FFN_of(src_ffn_id)
        dest_lf = self._level_FFN_of(dest_ffn_id)
        if src_lf is not None:
            src_lf.server_cnt -= 1
        if dest_lf is not None:
            dest_lf.server_cnt += 1

        # 触发 batch 状态修复 (status==6 重做 A2F; 其他状态无事发生)
        server.reactivate_batches(current_time, self.alpha_A, self.beta_A,
                                self.alpha_T, self.beta_T)

class BatchTypedQueue:
    # 随着FFN周期进行migrate的原队列方案，现已不用
    def __init__(self, num_queues=10):
        # 注意, 队列数量应保证比Batch的最大权重(即AF处理时间比)大
        self.num_queues = num_queues
        self.queues = [deque() for _ in range(num_queues)]
        #self.cycle = 0
        self.num_batch_inq = 0

    def add_batch(self, current_time: int, batch:Batch):
        """在当前cycle加入新Bintatch"""
        """根据Batch的weight决定加入哪个队列"""
        batch_weight = math.floor(batch.other_batch_FFN_unit_cost)
        target = (batch_weight + current_time) % self.num_queues
        self.queues[target].append(batch)
        self.num_batch_inq += 1

    #拿出一个Batch
    def worker_step(self, current_time: int, FFN_worker: FFN):
        """worker从 i%10 开始依次取"""
        # 队列中是否还有其他Batch
        flag = False
        
            
        for offset in range(self.num_queues):
                idx = (current_time + offset) % self.num_queues
                if self.queues[idx]:
                    batch = self.queues[idx].popleft()

                    self.num_batch_inq -= 1
                    FFN_worker.load_batch(current_time, batch)
                    #print(f"Worker {w} takes Batch {batch.id} from Q[{idx}]")
                    flag = True
                    break
        return flag
    
    # 清空时间队列里面剩余的Batch, 加入下一时间队列
    def migrate(self, current_time: int):
        """Q[i] → Q[i+1] 队首"""
        i = current_time % self.num_queues
        next_i = (i + 1) % self.num_queues

        remaining = list(self.queues[i])
        self.queues[i].clear()

        # 保持顺序插入队首
        for batch in reversed(remaining):
            self.queues[next_i].appendleft(batch)

    def print_status(self, current_time):
        print("Cycle {}: , num_batch_inq = {}".format(current_time, self.num_batch_inq))
        for i, q in enumerate(self.queues):
            print(f"Q[{i}]: {len(q)}")

class BatchQueue:
    """
    存放等待 FFN 处理的 batch. 用最小堆按 slack (紧迫度) 排序.

    slack = T_other_attention - T_self_FFN
      表示: 我可以再等多久才开始 FFN, 仍能和同 server 另一 batch 的 attention 完成时刻对齐.
      slack 越小 (或越负) 越紧迫.

    add_batch 时计算并入堆;
    worker_step 时弹出 slack 最小的 batch 给 FFN.
    """

    def __init__(self):
        self.heap = []                  # (slack, tiebreak, batch)
        self._counter = 0               # tiebreak: 相同 slack 时按入堆顺序 (FIFO)
        self.num_batch_inq = 0

    def add_batch(self, current_time: int, batch: Batch, alpha_F: float, beta_F: float):
        """加入新 batch. batch.other_batch_cost 已由 compute_other_batch_cost 写好."""
        own_ffn_cost = alpha_F * max(batch.num_req, 1) + beta_F
        slack = batch.other_batch_cost - own_ffn_cost
        begin_time = current_time + slack
        self._counter += 1
        heapq.heappush(self.heap, (begin_time, self._counter, batch))
        self.num_batch_inq += 1

    def worker_step(self, current_time: int, FFN_worker: FFN) -> bool:
        """worker 取一个最紧迫 (slack 最小) 的 batch 给 FFN. 返回是否成功取到."""
        if not self.heap:
            return False
        _slack, _tb, batch = heapq.heappop(self.heap)
        self.num_batch_inq -= 1
        FFN_worker.load_batch(current_time, batch)
        return True

    def print_status(self, current_time):
        print(f"Cycle {current_time}: num_batch_inq = {self.num_batch_inq}")
        if self.heap:
            slacks = sorted(s for s, _, _ in self.heap)
            print(f"  slack range: [{slacks[0]:.2f}, {slacks[-1]:.2f}], "
                  f"median: {slacks[len(slacks)//2]:.2f}")

class DynamicScheduler:
# AF之间固定匹配不再更改
    def __init__(self, arranger, servers:List[Server], FFN_workers:List[FFN], stats, buffer, stored_batches:Dict[int, Batch], alpha_A, beta_A, alpha_T, beta_T, alpha_F, beta_F, initially_full=True):
        self.servers = servers
        self.arranger = arranger
        self.buffer = buffer
        self.stored_batches = stored_batches
        self.num_servers = len(servers)
        self.num_batches = len(stored_batches)
        self.batch_size = servers[0].batch_size
        self.FFN_workers = FFN_workers
        self.num_FFN = len(FFN_workers)
        self.stats = stats
        self.alpha_A = alpha_A
        self.beta_A = beta_A
        self.alpha_T = alpha_T
        self.beta_T = beta_T
        self.alpha_F = alpha_F
        self.beta_F = beta_F
        #self.AF_match : Dict[int, int] = {} # server_id -> FFN_id

        self.initially_full = initially_full
        if self.initially_full:
            self.do_initialize_filling()
        
        self.batch_queue = BatchQueue()

    def do_initialize_filling(self):
        multi_print = True
        if multi_print:
            self.arranger.print_multitype_queue_info()
        tot_batch_size = 0
        for batch in self.stored_batches.values():
            tot_batch_size += batch.batch_size
        if self.arranger.num_req_inque < tot_batch_size:
            print("Requests needed: {}".format(tot_batch_size))
            raise ValueError("Not enough requests in the buffer to fill all batches")
        for batch in self.stored_batches.values():
            self.arranger.do_initial_filling(batch)
        
                # load_request加入之后会立刻开始处理, 所以初始化都用append

    def do_cycle_work(self, current_time):
            #print("Current Time: ", current_time)
            for server_id in range(len(self.servers)):
            # 这里假定了server和FFN的ID是按照顺序排列的，后续Debug的时候需注意
                server = self.servers[server_id]
                #FFN_server = self.FFN_workers[self.AF_match[server_id]]
                server.cycle_work(current_time, self.stats, alpha_T= self.alpha_T,beta_T= self.beta_T)
                if server.compute_memory_usage() > server.memory_capacity + server.dynamic_space:
                    server.evict_requests(current_time)
                evicted_requests = server.evict_out_requests(current_time)
                if evicted_requests:
                    self.arranger.evict_all_requests(evicted_requests)
                    self.stats.record_eviction(len(evicted_requests)) 
            # available_batches : List[Tuple[int, int, int, int]] = []
            # for server in self.servers:
            #     # 请注意, 这里假设Batch一般情况下都应该是全满的
            #     extend_batches = server.find_available_batch()
            #     available_batches.extend(extend_batches)
            """打印所有server和Batch的状态"""
            debug_print = False
            if debug_print:
                for server in self.servers:
                    server.print_debug_information()
                for batch in self.stored_batches.values():
                    batch.print_debug_information()
            self.arranger.arrange_requests(current_time)
            # while available_batches and self.buffer:
            #     request = self.buffer.pop()
            #     best_batch_info = min(available_batches)
            #     batch_id0 = best_batch_info[2]
            #     server_id0 = best_batch_info[3]
            #     best_batch = self.stored_batches[batch_id0]
            #     target_server = self.servers[server_id0]
            #     target_server.load_request_to_batch(current_time, best_batch_info[2], request)
            #     available_batches.remove(best_batch_info)
            #     if best_batch.has_free_slot(current_time):
            #         info0, info1 = best_batch.updated_info(current_time=current_time)
            #         new_info = (info0, info1, batch_id0, server_id0)
            #         available_batches.append(new_info)
            
            
            # 先处理刚刚完成传输的Batch
            # 这里假定scheduler靠近FFN节点，因此调度到FFN之间传输时间可以忽略
            for batch in self.stored_batches.values():
                if batch.status == 4:
                    if current_time < batch.current_ending:
                        continue
                    batch.status = 6
                    # 更新同组其他Batch的总开销作为自身的权重
                    server_id = batch.server_id
                    self.servers[server_id].compute_other_batch_cost(batch.batch_id, self.alpha_A, self.beta_A)
                    self.batch_queue.add_batch(current_time, batch, alpha_F=self.alpha_F, beta_F=self.beta_F)
                    # 分配进入合适的队列
            
            # FFN worker从队列中选取Batch
            free_FFN =0
            for FFN_worker in self.FFN_workers:
                if FFN_worker.current_busy:
                    continue
                else:
                    free_FFN += 1
            batch_inq = self.batch_queue.num_batch_inq
            if free_FFN and batch_inq:
                for FFN_worker in self.FFN_workers:
                    if FFN_worker.current_busy:
                        continue
                    else:
                        self.batch_queue.worker_step(current_time, FFN_worker)
                        free_FFN -= 1
                        if not free_FFN:
                            break
                        batch_inq -= 1
                        if not batch_inq:
                            print("There are still free FFN workers not used in some cycle.")   
                            break
            # 至此Batch已经load进入FFN， FFN会引导这个Batch完成状态的切换与后续的跟进                    
            for server in self.servers:
                server.attention_work(current_time, self.alpha_A, self.beta_A)
            for FFN_worker in self.FFN_workers:
                FFN_worker.cycle_work(current_time, self.alpha_F, self.beta_F)

            if current_time % 10000 == 0:
                self.arranger.print_multitype_queue_info()

class MoEScheduler:
    """
    MoE 架构调度器. 与 BasicScheduler/DynamicScheduler 的差别:
    - FFN 阶段不是把整个 batch 给一个 FFN worker, 而是把每个 (request, expert) 任务
      派发到对应的 expert queue, 由 MoEFFN worker 处理.
    - 8 个 expert queue, num_FFN 个 worker 均匀分到 8 类.
    """

    def __init__(self, arranger, servers, FFN_workers, stats, buffer, stored_batches,
                 alpha_A, beta_A, alpha_T, beta_T, alpha_F, beta_F,
                 num_experts=8, initially_full=False, costly_loading=False):
        self.arranger = arranger
        self.servers = servers
        self.FFN_workers = FFN_workers
        self.stats = stats
        self.buffer = buffer
        self.stored_batches = stored_batches
        self.alpha_A, self.beta_A = alpha_A, beta_A
        self.alpha_T, self.beta_T = alpha_T, beta_T
        self.alpha_F, self.beta_F = alpha_F, beta_F

        self.num_experts = num_experts
        # 8 个 expert queue: (request, batch)
        self.expert_queues = [deque() for _ in range(num_experts)]
        # worker 按 expert_id 分桶
        from collections import defaultdict
        self.workers_by_expert = defaultdict(list)
        for w in FFN_workers:
            assert hasattr(w, 'expert_id'), "MoEScheduler requires MoEFFN workers"
            self.workers_by_expert[w.expert_id].append(w)
        
        # 确保每种 expert 都有 worker
        for eid in range(num_experts):
            if not self.workers_by_expert[eid]:
                raise ValueError(f"No MoEFFN worker for expert {eid}")

        # 跟 BasicScheduler 一致: 初始填满 batch
        self.initially_full = initially_full
        if self.initially_full:
            self.do_initialize_filling()

    def do_initialize_filling(self):
        # 复用 BasicScheduler 的初始化
        tot_batch_size = sum(b.batch_size for b in self.stored_batches.values())
        if self.arranger.num_req_inque < tot_batch_size:
            print(f"Basic number:{self.arranger.num_req_inque}, "
                  f"actually needed:{tot_batch_size}")
        for batch in self.stored_batches.values():
            self.arranger.do_initial_filling(batch)

    def _dispatch_moe_batch(self, batch):
        """把一个 batch 拆成 (req, expert) 任务投入 expert queue."""
        batch.is_MoE_mode = True
        batch.prepare_moe_dispatch()
        batch.is_MoE_dispatched = True
        # batch.status 此时是 6 (A2F 完成, 待 FFN), 改成 2 (FFN 处理中)
        batch.status = 2
        total_tasks = 0
        for req in batch.requests:
            for eid in req.expert_ids:
                self.expert_queues[eid].append((req, batch))
                total_tasks += 1
        print(f"[dispatch] batch {batch.batch_id} dispatched, "
          f"requests={len(batch.requests)}, total_tasks={total_tasks}, "
          f"pending={batch.moe_pending_count}")

    def do_cycle_work(self, current_time):
        # Stage 1: server cycle (推进 batch 状态: F2A→A, A2F→6, attention→A2F, FFN→F2A)
        for server in self.servers:
            # 注意: 这里 FFN_worker 传 None, 因为 MoE 模式下 Server.cycle_work 的 status==4 分支
            # 不会调 FFN_worker.load_batch (有 is_MoE_mode 判断)
            server.cycle_work(current_time, self.stats, None, self.alpha_T, self.beta_T)

        # Stage 2: evict 检查
        for server in self.servers:
            if server.compute_memory_usage() > server.memory_capacity + server.dynamic_space:
                server.evict_requests(current_time)
            evicted = server.evict_out_requests(current_time)
            if evicted:
                self.arranger.evict_all_requests(evicted)
                self.stats.record_eviction(len(evicted))

        # Stage 3: arrange 新 request 进 batch
        self.arranger.arrange_requests(current_time)

        # Stage 4: 把刚到达 FFN 端 (status=6) 且未 dispatch 的 batch 派发到 expert queue
        for batch in self.stored_batches.values():
            if batch.status == 6 and not batch.is_MoE_dispatched:
                self._dispatch_moe_batch(batch)

        # Stage 5: 每个 expert 的空闲 worker 取 task
        for eid in range(self.num_experts):
            q = self.expert_queues[eid]
            for w in self.workers_by_expert[eid]:
                if not w.current_busy and q:
                    req, batch = q.popleft()
                    w.load_task(current_time, req, batch)

        # Stage 6: 推进所有 MoE worker
        for w in self.FFN_workers:
            w.cycle_work(current_time)

        # Stage 7: attention
        for server in self.servers:
            server.attention_work(current_time, self.alpha_A, self.beta_A)

        # 每 N cycle 打一次
        if current_time % 5000 == 0:
            qlens = [len(self.expert_queues[i]) for i in range(self.num_experts)]
            print(f"[cycle {current_time}] expert queue lens: {qlens}")  
            for w in self.FFN_workers:
                w.print_debug_information()
        test_print = False
        if test_print:
            if current_time % 1000 == 0:
                for batch in self.stored_batches.values():
                    batch.print_debug_information()



class PriorityExpertQueue:
    """
    单 expert 的优先队列, 内部 5 个 sub_q[0..4]:
      sub_q[0] = 救援队列 (方案 2/3 提升), 最高优先级
      sub_q[k] (k=1..4) = priority_level==k 的 (req, eid) 任务

    sub_q 实现: OrderedDict, key=(req.rid, eid), value=(req, eid).
      - 入队: __setitem__, O(1), 追加到末尾 (FCFS)
      - 出队首: 用 next(iter(...)) + pop, O(1)
      - 任意位置删除: pop(key), O(1)

    add_task / remove_task / move_task 全部 O(1).
    """

    NUM_LEVELS = 5    # 0..4

    def __init__(self, expert_id, c_threshold=None):
        """
        c_threshold: dict {k: c_k}, sub_q[k] 连续 c_k 个 task 处理后仍没被取过就强制取.
          缺省 {2: 20, 3: 50, 4: 100} (你给的值). None 表示不启用方案 1.
        """
        self.expert_id = expert_id
        self.sub_q = [OrderedDict() for _ in range(self.NUM_LEVELS)]

        # 方案 1 计数: served_count = 本 queue 至今处理过的 task 总数
        # last_served[k] = 上次从 sub_q[k] 取 task 时的 served_count
        self.served_count = 0
        self.last_served = [0] * self.NUM_LEVELS
        self.c_threshold = c_threshold      # dict 或 None

    def add_task(self, req, eid, batch, level=None):
        """入队. level 默认按 req.priority_level."""
        if level is None:
            level = req.priority_level
        key = (req.rid, eid)
        self.sub_q[level][key] = (req, eid, batch)
        req.task_locations[eid] = level     # 反向索引

    def remove_task(self, req, eid):
        """从所在 sub_q 删除. 调用方保证该 task 在队列中."""
        level = req.task_locations.pop(eid, None)
        if level is None:
            return False
        key = (req.rid, eid)
        self.sub_q[level].pop(key, None)
        return True

    def move_task(self, req, eid, new_level):
        """主动迁移单个 task 到 sub_q[new_level]. O(1)."""
        old_level = req.task_locations.get(eid)
        if old_level is None:
            return    # 该 task 已不在本 queue (可能刚被取走)
        if old_level == new_level:
            return
        key = (req.rid, eid)
        item = self.sub_q[old_level].pop(key, None)
        if item is None:
            return
        self.sub_q[new_level][key] = item
        req.task_locations[eid] = new_level

    def total_len(self):
        return sum(len(q) for q in self.sub_q)

    def per_level_lens(self):
        return [len(q) for q in self.sub_q]

    def pop_task(self):
        """
        取一个最高优先级的 task 给 worker. 返回 (req, eid) 或 None.
        优先级:
          1. sub_q[0] 总是优先
          2. 方案 1 starve 检查 (从低优先级 k=4..2 开始, 最先 starve 的先救)
          3. 否则按 k=1..4 顺序取
        """
        # Step 1: 救援队列
        if self.sub_q[0]:
            return self._pop_from(0)

        # Step 2: starve 检查 (k=4..2, 因为 c_k 你设为 c2<c3<c4, 但实际 k 越大越易 starve;
        # 这里我们按"哪个超阈值最久"找, 简化为按 k 顺序找第一个超阈值的)
        if self.c_threshold is not None:
            # 从 k=2 开始检查 (优先缓解高优先级队列的starvation问题)
            for k in (2, 3, 4):
                c_k = self.c_threshold.get(k)
                if c_k is None:
                    continue
                if not self.sub_q[k]:
                    """如果队列空了, 则重新开始计时"""
                    self.last_served[k] = self.served_count
                elif self.sub_q[k] and (self.served_count - self.last_served[k]) >= c_k:
                    return self._pop_from(k)

        # Step 3: 正常优先级
        for k in (1, 2, 3, 4):
            if self.sub_q[k]:
                return self._pop_from(k)

        return None

    def _pop_from(self, k):
        """从 sub_q[k] 取队首 (FCFS), 更新 served 计数."""
        # OrderedDict 的 popitem(last=False) 取队首 (FCFS)
        key, (req, eid, batch) = self.sub_q[k].popitem(last=False)
        req.task_locations.pop(eid, None)
        self.served_count += 1
        self.last_served[k] = self.served_count
        return (req, eid, batch)


class MoEPriorityScheduler:
    """
    MoE FFN_type=5: 带优先级队列 + 可选拥塞避免的 MoE 调度器.

    优先级核心: 每个 expert queue 内 5 个子队列 sub_q[0..4], 按 request 的 priority_level 分桶.
    迁移策略: 主动迁移 (worker 完成一个 task 后, 同 request 仍排队的 task 立刻挪到 sub_q[k-1]).
    """

    def __init__(self, arranger, servers, FFN_workers, stats, buffer, stored_batches,
                 alpha_A, beta_A, alpha_T, beta_T, alpha_F, beta_F,
                 num_experts=8,
                 starve_avoid_1=False,
                 starve_avoid_2=False,
                 starve_avoid_3=False,
                 c2=20, c3=50, c4=100,
                 starve3_threshold=40,
                 initially_full=False):
        self.arranger = arranger
        self.servers = servers
        self.FFN_workers = FFN_workers
        self.stats = stats
        self.buffer = buffer
        self.stored_batches = stored_batches
        self.alpha_A, self.beta_A = alpha_A, beta_A
        self.alpha_T, self.beta_T = alpha_T, beta_T
        self.alpha_F, self.beta_F = alpha_F, beta_F

        self.num_experts = num_experts

        # 三个拥塞避免开关
        self.starve_avoid_1 = starve_avoid_1
        self.starve_avoid_2 = starve_avoid_2
        self.starve_avoid_3 = starve_avoid_3
        self.starve3_threshold = starve3_threshold

        # PriorityExpertQueue 初始化
        # Threshold: 每个队列有多长时间没有被处理就会在下一轮强制被处理
        c_threshold = {2: c2, 3: c3, 4: c4} if starve_avoid_1 else None
        self.expert_queues = [
            PriorityExpertQueue(eid, c_threshold=c_threshold)
            for eid in range(num_experts)
        ]

        # worker 按 expert_id 分桶
        self.workers_by_expert = defaultdict(list)
        for w in FFN_workers:
            assert hasattr(w, 'expert_id'), "MoEPriorityScheduler requires MoEFFN workers"
            self.workers_by_expert[w.expert_id].append(w)
        for eid in range(num_experts):
            if not self.workers_by_expert[eid]:
                raise ValueError(f"No MoEFFN worker for expert {eid}")

        # 给每个 batch 注入 scheduler 引用 (供 batch.on_moe_expert_done 调用迁移函数)
        for batch in self.stored_batches.values():
            batch.scheduler = self
            batch.is_MoE_mode = True

        # 初始填充
        self.initially_full = initially_full
        if self.initially_full:
            self.do_initialize_filling()

    def do_initialize_filling(self):
        tot_batch_size = sum(b.batch_size for b in self.stored_batches.values())
        if self.arranger.num_req_inque < tot_batch_size:
            print(f"Basic number:{self.arranger.num_req_inque}, "
                  f"actually needed:{tot_batch_size}")
            raise ValueError("Not enough requests to fill all batches")
        for batch in self.stored_batches.values():
            self.arranger.do_initial_filling(batch)

    def _dispatch_moe_batch(self, batch, current_time):
        """把一个 batch 拆成 (req, expert) 任务投入 expert queue."""
        batch.is_MoE_mode = True
        batch.prepare_moe_dispatch()
        batch.is_MoE_dispatched = True
        batch.status = 2

        for req in batch.requests:
            # 方案 2: 记录 dispatch 时刻和 time_bound
            req.dispatch_time = current_time
            if self.starve_avoid_2:
                max_qlen = max(
                    self.expert_queues[eid].total_len() for eid in req.expert_ids
                )
                req.time_bound = 2 * max_qlen
            else:
                req.time_bound = 0    # 不启用就不算

            # task 入队 (按 priority_level = 4)
            for eid in req.expert_ids:
                self.expert_queues[eid].add_task(req, eid, batch)

    def migrate_request_tasks_after_done(self, request):
        """一个 task 完成后, 把该 request 仍在排队的 task 从 sub_q[k] 移到 sub_q[k-1].

        request.task_locations 给出 eid -> 当前 sub_q level.
        新的 priority_level 由 property 算 (remaining_experts - extra_reduce).
        遍历 task_locations 复制一份 (因为 move_task 会修改它).
        """
        new_level = request.priority_level
        locations_snapshot = dict(request.task_locations)
        for eid, _old_level in locations_snapshot.items():
            self.expert_queues[eid].move_task(request, eid, new_level)

    def promote_batch_to_zero(self, batch):
        """方案 3: 把 batch 内所有未完成 request 的所有 task 主动迁到 sub_q[0]."""
        for req in batch.requests:
            if req.priority_level == 0:
                continue
            if not req.task_locations:
                continue
            # 设 extra_reduce 让 priority_level 变 0
            req.extra_reduce = req.remaining_experts
            locations_snapshot = dict(req.task_locations)
            for eid, _old_level in locations_snapshot.items():
                self.expert_queues[eid].move_task(req, eid, 0)

    def check_time_bound_for_all(self, current_time):
        """方案 2: 每 cycle 扫所有 status==2 batch 内的 request, 检查 time_bound."""
        for batch in self.stored_batches.values():
            if batch.status != 2 or not batch.is_MoE_dispatched:
                continue
            for req in batch.requests:
                if req.remaining_experts <= 0:
                    continue
                if req.priority_level == 0:
                    continue    # 已经在最高优先级, 不再降
                if not req.task_locations:
                    continue
                if current_time - req.dispatch_time >= req.time_bound:
                    # 触发降级
                    req.extra_reduce += 1
                    req.dispatch_time = current_time    # 重置计时, 下个 bound 后再降
                    new_level = req.priority_level
                    locations_snapshot = dict(req.task_locations)
                    for eid, _old_level in locations_snapshot.items():
                        self.expert_queues[eid].move_task(req, eid, new_level)

    def do_cycle_work(self, current_time):
        # Stage 1: server cycle
        for server in self.servers:
            server.cycle_work(current_time, self.stats, None, self.alpha_T, self.beta_T)

        # Stage 2: evict
        for server in self.servers:
            if server.compute_memory_usage() > server.memory_capacity + server.dynamic_space:
                server.evict_requests(current_time)
            evicted = server.evict_out_requests(current_time)
            if evicted:
                self.arranger.evict_all_requests(evicted)
                self.stats.record_eviction(len(evicted))

        # Stage 3: arrange
        self.arranger.arrange_requests(current_time)

        # Stage 4: dispatch status==6 batches
        for batch in self.stored_batches.values():
            if batch.status == 6 and not batch.is_MoE_dispatched:
                self._dispatch_moe_batch(batch, current_time)

        # Stage 5: 方案 2 时间检查 (扫所有 status==2 的 request)
        if self.starve_avoid_2:
            self.check_time_bound_for_all(current_time)

        # Stage 6: worker 取 task
        for eid in range(self.num_experts):
            q = self.expert_queues[eid]
            for w in self.workers_by_expert[eid]:
                if not w.current_busy:
                    task = q.pop_task()
                    if task is not None:
                        req, _eid, batch = task
                        w.load_task(current_time, req, batch)

        # Stage 7: worker tick
        for w in self.FFN_workers:
            w.cycle_work(current_time)

        # Stage 8: attention
        for server in self.servers:
            server.attention_work(current_time, self.alpha_A, self.beta_A)

        # 诊断打印
        if current_time % 5000 == 0:
            print(f"[cycle {current_time}] expert queues:")
            for eid in range(self.num_experts):
                lens = self.expert_queues[eid].per_level_lens()
                if any(lens):
                    print(f"  expert {eid}: sub_q={lens}, "
                          f"served={self.expert_queues[eid].served_count}")


# ===========================================================================
# 方案 1: BalancedAttnFFNScheduler (FFN_type=6)
# ===========================================================================
class BalancedAttnFFNScheduler:
    """按 attention 节点 (server) 分配到 FFN, 但 FFN 内部用 FCFS (不维护流水线).
    每 cycle 实时算每个 server 的 w = t_F / (t_A + t_F), 并维护每个 FFN 的 sum_w.
    爆表 FFN (sum_w > 1) 内部 r < w 的 server 尝试与其它 FFN 的 server 交换.

    交换准则 (A3 + B1+禁忌):
      优先级 1: 双赢 (两 FFN 交换后都不爆表), 选交换后两边 sum 都最接近 1 的 (sum-1 平方和最小)
      优先级 2: 爆表 FFN 退出爆表, 选交换后 new_sum_F1 最小的
      允许多次交换, 已交换 pair 当 cycle 不再选
    """

    def __init__(self, arranger, servers, FFN_workers, stats, buffer, stored_batches,
                 alpha_A, beta_A, alpha_T, beta_T, alpha_F, beta_F, initially_full=True):
        self.arranger = arranger
        self.servers = servers
        self.FFN_workers = FFN_workers
        self.stats = stats
        self.buffer = buffer
        self.stored_batches = stored_batches
        self.alpha_A, self.beta_A = alpha_A, beta_A
        self.alpha_T, self.beta_T = alpha_T, beta_T
        self.alpha_F, self.beta_F = alpha_F, beta_F

        self.num_servers = len(servers)
        self.num_FFN = len(FFN_workers)

        # 普通 FFN (FCFS): server 必须走 status==4 → load_batch, 不能被 dynamic_matching 跳过
        for server in self.servers:
            server.dynamic_matching = False

        # server_id -> FFN_id  (核心: AF 匹配)
        self.AF_match: Dict[int, int] = {}
        # FFN_id -> [server_id, ...]
        self.AF_graph: Dict[int, List[int]] = defaultdict(list)

        self.initially_full = initially_full
        if self.initially_full:
            self.do_initialize_filling()

        # 初始分配: round-robin (按 w 降序排服务器, 依次喂给 sum_w 最小的 FFN)
        self._initial_assignment()

    def do_initialize_filling(self):
        tot_batch_size = sum(b.batch_size for b in self.stored_batches.values())
        if self.arranger.num_req_inque < tot_batch_size:
            print(f"Basic number:{self.arranger.num_req_inque}, actually needed:{tot_batch_size}")
            raise ValueError("Not enough requests to fill all batches")
        for batch in self.stored_batches.values():
            self.arranger.do_initial_filling(batch)

    # ------------------------------------------------------------------
    # 核心: 实时计算 w / sum_w / r
    # ------------------------------------------------------------------
    def _server_w(self, server) -> float:
        """w = t_F / (t_A + t_F).  t_F, t_A 都按 server 内当前所有 batch 实时算."""
        t_A = self.alpha_A * sum(b.length for b in server.batches.values()) + self.beta_A
        t_F = sum(self.alpha_F * max(b.num_req, 1) + self.beta_F
                  for b in server.batches.values())
        denom = t_A + t_F
        return t_F / denom if denom > 0 else 0

    def _ffn_sum_w(self, ffn_id) -> float:
        return sum(self._server_w(self.servers[sid]) for sid in self.AF_graph[ffn_id])

    def _initial_assignment(self):
        """初始按 w 降序的 best-fit-decreasing: 喂给当前 sum_w 最小的 FFN."""
        ws = sorted(
            [(self._server_w(s), s.server_id) for s in self.servers],
            key=lambda x: -x[0]
        )
        ffn_sum = [0.0] * self.num_FFN
        for w, sid in ws:
            target = min(range(self.num_FFN), key=lambda f: ffn_sum[f])
            self.AF_match[sid] = target
            self.AF_graph[target].append(sid)
            ffn_sum[target] += w
            self.servers[sid].mapped_FFN_id = target
        for ffn_id in range(self.num_FFN):
            for sid in self.AF_graph[ffn_id]:
                for batch in self.servers[sid].batches.values():
                    batch.mapped_FFN_id = ffn_id

    # ------------------------------------------------------------------
    # 交换平衡 (每 cycle 调一次)
    # ------------------------------------------------------------------
    def _try_balance(self, current_time):
        """每 cycle 一次. 多次交换直到没有更多有益交换."""
        attempted_pairs = set()    # 禁忌: 已尝试过的 (sa, sb)
        max_iter = 50              # 上限防极端
        for _ in range(max_iter):
            # 重新计算各 FFN 的 sum_w
            ffn_sum = {f: self._ffn_sum_w(f) for f in range(self.num_FFN)}
            overloaded = [f for f, s in ffn_sum.items() if s > 1.0]
            if not overloaded:
                break

            best_swap = None
            best_score = None   # 优先级 (tier, metric); tier=0 双赢, tier=1 退出爆表
            best_tier = 2       # 越小越好

            # 找需要换出的 server (爆表 FFN 里 r < w 的)
            for f1 in overloaded:
                sum_f1 = ffn_sum[f1]
                # 在爆表 FFN, w_a 的 server: r_a = w_a / sum_f1 < w_a 当 sum_f1 > 1
                # 即所有 server 都 r < w; 直接全部尝试交换
                for sa in self.AF_graph[f1]:
                    w_a = self._server_w(self.servers[sa])
                    # 找对端 server sb 在其它 FFN 上, 该 FFN 不爆表
                    for f2 in range(self.num_FFN):
                        if f2 == f1:
                            continue
                        for sb in self.AF_graph[f2]:
                            pair = tuple(sorted([sa, sb]))
                            if pair in attempted_pairs:
                                continue
                            w_b = self._server_w(self.servers[sb])
                            new_sum_f1 = sum_f1 - w_a + w_b
                            new_sum_f2 = ffn_sum[f2] - w_b + w_a
                            # 评估
                            if new_sum_f1 <= 1.0 and new_sum_f2 <= 1.0:
                                # 双赢
                                tier = 0
                                metric = (new_sum_f1 - 1.0)**2 + (new_sum_f2 - 1.0)**2
                            elif new_sum_f1 <= 1.0:
                                # f1 退出爆表 (但 f2 可能进入或仍未爆表)
                                tier = 1
                                metric = new_sum_f1
                            else:
                                # f1 仍爆表, 跳过
                                continue
                            score = (tier, metric)
                            if best_score is None or score < best_score:
                                best_swap = (sa, sb, f1, f2)
                                best_score = score
                                best_tier = tier

            if best_swap is None:
                break    # 没有更多有益交换

            sa, sb, f1, f2 = best_swap
            attempted_pairs.add(tuple(sorted([sa, sb])))
            self._apply_swap(current_time, sa, sb, f1, f2)
            # 打印
            print(f"[balance cycle {current_time}] swap server {sa} (FFN {f1}) "
                  f"<-> server {sb} (FFN {f2}), tier={best_tier}")

    def _apply_swap(self, current_time, sa, sb, f1, f2):
        """把 sa 的 FFN 改成 f2, sb 的 FFN 改成 f1. 更新 AF_match / AF_graph / batch.mapped_FFN_id."""
        self.AF_match[sa] = f2
        self.AF_match[sb] = f1
        self.AF_graph[f1].remove(sa)
        self.AF_graph[f1].append(sb)
        self.AF_graph[f2].remove(sb)
        self.AF_graph[f2].append(sa)
        self.servers[sa].mapped_FFN_id = f2
        self.servers[sb].mapped_FFN_id = f1
        for batch in self.servers[sa].batches.values():
            batch.mapped_FFN_id = f2
        for batch in self.servers[sb].batches.values():
            batch.mapped_FFN_id = f1
        # batch 重新激活 (status==6 / 4 需要重做 A2F 走新 FFN)
        self.servers[sa].reactivate_batches(current_time, self.alpha_A, self.beta_A,
                                             self.alpha_T, self.beta_T)
        self.servers[sb].reactivate_batches(current_time, self.alpha_A, self.beta_A,
                                             self.alpha_T, self.beta_T)

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def do_cycle_work(self, current_time):
        for sid in range(self.num_servers):
            server = self.servers[sid]
            ffn_id = self.AF_match[sid]
            FFN_worker = self.FFN_workers[ffn_id]
            server.cycle_work(current_time, self.stats, FFN_worker, self.alpha_T, self.beta_T)
            if server.compute_memory_usage() > server.memory_capacity + server.dynamic_space:
                server.evict_requests(current_time)
            evicted = server.evict_out_requests(current_time)
            if evicted:
                self.arranger.evict_all_requests(evicted)
                self.stats.record_eviction(len(evicted))

        self.arranger.arrange_requests(current_time)

        # 每 cycle 平衡一次
        self._try_balance(current_time)

        for server in self.servers:
            server.attention_work(current_time, self.alpha_A, self.beta_A)
        for FFN_worker in self.FFN_workers:
            FFN_worker.cycle_work(current_time, self.alpha_F, self.beta_F)


# ===========================================================================
# 方案 2: BalancedBatchFFNScheduler (FFN_type=7)
# ===========================================================================
class BalancedBatchFFNScheduler:
    """按 batch 分配到 FFN (同 server 两 batch 可在不同 FFN).
    每 batch 的 w = t_A(other_batch) / (2 * t_F(this_batch)).
    含义: 另一 batch 的 attention 期间能容纳的 (this batch FFN) 处理量.

    分组策略类似原 PipelineScheduler: 按 floor(w) 分桶, 高 w 的 batch 单挂少量 FFN,
    低 w 的 batch 多个挂同一 FFN.
    FFN 数量过少 → 优先让"挂的 batch 数少"的 FFN 超额;
    FFN 数量过多 → 优先分担"挂的 batch 数多"的 FFN 的负担.
    """

    def __init__(self, arranger, servers, FFN_workers, stats, buffer, stored_batches,
                 alpha_A, beta_A, alpha_T, beta_T, alpha_F, beta_F, initially_full=True):
        self.arranger = arranger
        self.servers = servers
        self.FFN_workers = FFN_workers
        self.stats = stats
        self.buffer = buffer
        self.stored_batches = stored_batches
        self.alpha_A, self.beta_A = alpha_A, beta_A
        self.alpha_T, self.beta_T = alpha_T, beta_T
        self.alpha_F, self.beta_F = alpha_F, beta_F

        self.num_servers = len(servers)
        self.num_FFN = len(FFN_workers)
        self.num_batches = len(stored_batches)

        # 普通 FFN (FCFS): 保证 status==4 不被 dynamic_matching 跳过
        for server in self.servers:
            server.dynamic_matching = False

        # batch_id -> FFN_id
        self.BF_match: Dict[int, int] = {}
        # FFN_id -> [batch_id, ...]
        self.BF_graph: Dict[int, List[int]] = defaultdict(list)

        # 单 FFN 至多能装的 batch 数 (与原 PipelineScheduler 一致)
        self.max_AF_ratio = 5

        self.initially_full = initially_full
        if self.initially_full:
            self.do_initialize_filling()

        self._initial_assignment()

    def do_initialize_filling(self):
        tot_batch_size = sum(b.batch_size for b in self.stored_batches.values())
        if self.arranger.num_req_inque < tot_batch_size:
            print(f"Basic number:{self.arranger.num_req_inque}, actually needed:{tot_batch_size}")
            raise ValueError("Not enough requests to fill all batches")
        for batch in self.stored_batches.values():
            self.arranger.do_initial_filling(batch)

    # ------------------------------------------------------------------
    # 核心: 实时算每个 batch 的 w
    # ------------------------------------------------------------------
    def _batch_t_A(self, batch) -> float:
        """batch 自己的 attention 阶段预估用时."""
        return self.alpha_A * batch.length + self.beta_A

    def _batch_t_F(self, batch) -> float:
        """batch 自己的 FFN 阶段预估用时."""
        return self.alpha_F * max(batch.num_req, 1) + self.beta_F

    def _batch_w_level(self, batch) -> float:
        """w_level = t_F(self) / t_A(other_batch).  仅用于 FFN level / 初始分组容量参考."""
        server = self.servers[batch.server_id]
        t_A_other = 0.0
        for b in server.batches.values():
            if b.batch_id != batch.batch_id:
                t_A_other += self.alpha_A * b.length + self.beta_A
        if t_A_other <= 0:
            return float('inf')
        return self._batch_t_F(batch) / t_A_other

    def _batch_w(self, batch) -> float:
        """w = t_F(self) / (2 * t_A(other_batch)).  '应得份额', 与 r 量纲一致 (都是 0~1 比例).
        r >= w 表示该 batch 在 FFN 内分到的资源 >= 它应得的份额 (不被阻塞).
        """
        server = self.servers[batch.server_id]
        t_A_other = 0.0
        for b in server.batches.values():
            if b.batch_id != batch.batch_id:
                t_A_other += self.alpha_A * b.length + self.beta_A
        if t_A_other <= 0:
            return float('inf')
        return self._batch_t_F(batch) / (2.0 * t_A_other)

    def _batch_weight(self, batch) -> float:
        """weight = t_F(self) / (t_A(b1) + t_A(b2)).
        用于计算 r (FFN 内资源占比) 和判断 FFN 爆表.
        分母是同 server 两个 batch 的 attention 用时之和.
        """
        server = self.servers[batch.server_id]
        t_A_sum = 0.0
        for b in server.batches.values():
            t_A_sum += self.alpha_A * b.length + self.beta_A
        if t_A_sum <= 0:
            return float('inf')
        return self._batch_t_F(batch) / t_A_sum

    def _ffn_sum_weight(self, ffn_id) -> float:
        """FFN 上所有 batch 的 weight 之和. > 2 即爆表."""
        return sum(self._batch_weight(self.stored_batches[bid])
                   for bid in self.BF_graph[ffn_id])

    def _ffn_overloaded(self, ffn_id) -> bool:
        """爆表判据: weight 之和 > 2 (分母用两个 batch attention 之和, 容量上限是 2)."""
        return self._ffn_sum_weight(ffn_id) > 2.0

    def _batch_r(self, batch, ffn_id=None) -> float:
        """r = weight(batch) / sum(weight of all batches in same FFN).
        FFN 内分到的资源占比. r >= w 表示分到的资源够用.
        """
        if ffn_id is None:
            ffn_id = self.BF_match[batch.batch_id]
        sum_w = self._ffn_sum_weight(ffn_id)
        if sum_w <= 0:
            return 0.0
        return self._batch_weight(batch) / sum_w

    def _ffn_batch_count(self, ffn_id):
        return len(self.BF_graph[ffn_id])

    def _initial_assignment(self):
        """初始分配: 按 weight 降序的 best-fit-decreasing.
        目标: 每个 FFN 的 sum_weight 接近但不超过 2.
        FFN 太少 → 优先让挂 batch 数少的 FFN 超额;
        FFN 太多 → 从挂 batch 多的 FFN 拆 batch 到空 FFN.
        """
        batches_sorted = sorted(
            self.stored_batches.values(),
            key=lambda b: -self._batch_weight(b)
        )

        ffn_sum = [0.0] * self.num_FFN
        ffn_count = [0] * self.num_FFN

        for batch in batches_sorted:
            wt = self._batch_weight(batch)
            # best-fit: 找塞进去后 sum 仍 <= 2 且剩余空间最小的 FFN
            candidates = [(2.0 - (ffn_sum[f] + wt), f)
                          for f in range(self.num_FFN)
                          if ffn_sum[f] + wt <= 2.0]
            if candidates:
                _, target = min(candidates)   # 剩余空间最小 = 最紧凑
            else:
                # 都会超额 → 优先挂 batch 数最少的 FFN (你的要求)
                target = min(range(self.num_FFN), key=lambda f: ffn_count[f])
            self.BF_match[batch.batch_id] = target
            self.BF_graph[target].append(batch.batch_id)
            ffn_sum[target] += wt
            ffn_count[target] += 1

        # FFN 太多 (有空 FFN): 从挂 batch 最多的 FFN 拆一个给空 FFN
        empty_ffn = [f for f in range(self.num_FFN) if ffn_count[f] == 0]
        while empty_ffn:
            f_empty = empty_ffn.pop(0)
            f_max = max(range(self.num_FFN), key=lambda f: ffn_count[f])
            if ffn_count[f_max] <= 1:
                break
            cands = sorted(self.BF_graph[f_max],
                           key=lambda bid: -self._batch_weight(self.stored_batches[bid]))
            mv_bid = cands[0]
            self.BF_graph[f_max].remove(mv_bid)
            self.BF_graph[f_empty].append(mv_bid)
            self.BF_match[mv_bid] = f_empty
            ffn_count[f_max] -= 1
            ffn_count[f_empty] += 1

        for bid, fid in self.BF_match.items():
            self.stored_batches[bid].mapped_FFN_id = fid

    # ------------------------------------------------------------------
    # 交换平衡 (类似方案 1)
    # ------------------------------------------------------------------
    def _try_balance(self, current_time):
        """按 batch 交换以缓解 FFN 爆表.

        发起前提: batch ba 在爆表 FFN f1 (sum_weight(f1) > 2) 且 r(ba) < w(ba) (被阻塞).
        与 batch bb (在 f2) 交换, 接受当且仅当满足以下之一:
          (a) 交换后 f1 和 f2 都不爆表 (sum_weight <= 2);
          (b) 交换后 r(ba) >= w(ba) 且 r(bb) >= w(bb) 都成立, 且 ba 的新 FFN (f2) 不爆表.
        交换后立刻重算受影响 FFN 内所有 batch 的 r (因为重算依赖 BF_graph, 自动反映).
        被换走的 batch 因换到 r>=w 的位置, 本回合不再满足发起前提, 不会连续交换.
        """
        max_iter = 50
        for _ in range(max_iter):
            # 重新算各 FFN 的 sum_weight (爆表判据)
            ffn_sum_weight = {f: self._ffn_sum_weight(f) for f in range(self.num_FFN)}
            overloaded = [f for f, s in ffn_sum_weight.items() if s > 2.0]
            if not overloaded:
                break

            applied = False
            for f1 in overloaded:
                sum_w_f1 = ffn_sum_weight[f1]
                for ba in list(self.BF_graph[f1]):
                    ba_batch = self.stored_batches[ba]
                    r_ba = self._batch_r(ba_batch, f1)
                    w_ba = self._batch_w(ba_batch)
                    # 发起前提: r < w (被阻塞)
                    if not (r_ba < w_ba):
                        continue
                    weight_a = self._batch_weight(ba_batch)

                    # 找对端 bb
                    found = None
                    for f2 in range(self.num_FFN):
                        if f2 == f1:
                            continue
                        sum_w_f2 = ffn_sum_weight[f2]
                        for bb in list(self.BF_graph[f2]):
                            bb_batch = self.stored_batches[bb]
                            weight_b = self._batch_weight(bb_batch)

                            new_sum_f1 = sum_w_f1 - weight_a + weight_b
                            new_sum_f2 = sum_w_f2 - weight_b + weight_a

                            # 条件 (a): 两边都不爆表
                            cond_a = (new_sum_f1 <= 2.0 and new_sum_f2 <= 2.0)

                            # 条件 (b): 交换后双方 r>=w 且 ba 新 FFN(f2) 不爆表
                            cond_b = False
                            if new_sum_f2 <= 2.0:
                                # 交换后 ba 在 f2, bb 在 f1; 算交换后的 r
                                # 交换后 f2 内 weight 和 = new_sum_f2, ba 的 weight 不变
                                r_ba_new = weight_a / new_sum_f2 if new_sum_f2 > 0 else 0.0
                                w_ba_new = self._batch_w(ba_batch)   # w 与 FFN 无关, 不变
                                # 交换后 bb 在 f1, f1 内 weight 和 = new_sum_f1
                                r_bb_new = weight_b / new_sum_f1 if new_sum_f1 > 0 else 0.0
                                w_bb_new = self._batch_w(bb_batch)
                                if r_ba_new >= w_ba_new and r_bb_new >= w_bb_new:
                                    cond_b = True

                            if cond_a or cond_b:
                                found = (ba, bb, f1, f2)
                                break
                        if found:
                            break

                    if found:
                        ba_, bb_, f1_, f2_ = found
                        self._apply_batch_swap(current_time, ba_, bb_, f1_, f2_)
                        print(f"[batch balance cycle {current_time}] swap batch {ba_} "
                              f"(FFN {f1_}) <-> batch {bb_} (FFN {f2_})")
                        applied = True
                        break   # 重新计算 sum_weight, 进入下一轮 iter
                if applied:
                    break

            if not applied:
                break   # 本轮没有可行交换, 停止

    def _apply_batch_swap(self, current_time, ba, bb, f1, f2):
        self.BF_match[ba] = f2
        self.BF_match[bb] = f1
        self.BF_graph[f1].remove(ba)
        self.BF_graph[f1].append(bb)
        self.BF_graph[f2].remove(bb)
        self.BF_graph[f2].append(ba)
        self.stored_batches[ba].mapped_FFN_id = f2
        self.stored_batches[bb].mapped_FFN_id = f1
        # 涉及的 server 需要 reactivate (因为 batch 已切换 FFN)
        sa = self.stored_batches[ba].server_id
        sb = self.stored_batches[bb].server_id
        affected = {sa, sb}
        for sid in affected:
            self.servers[sid].reactivate_batches(current_time, self.alpha_A, self.beta_A,
                                                  self.alpha_T, self.beta_T)

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def _server_cycle_for_batch_routing(self, current_time):
        """方案 2 的特殊 server cycle: status==4 时按 batch.mapped_FFN_id 选 FFN."""
        for server in self.servers:
            for batch_id, batch in server.batches.items():
                if batch.status == 3:
                    if batch.is_due(current_time):
                        batch.F2A_transmission_end(current_time)
                        batch.do_new_round(current_time, self.stats)
                elif batch.status == 4:
                    if batch.is_due(current_time):
                        batch.A2F_transmission_end(current_time)
                        # 按 batch 自己的 mapped_FFN_id 派发
                        ffn_id = batch.mapped_FFN_id
                        if ffn_id < 0:
                            ffn_id = self.BF_match.get(batch.batch_id, 0)
                        self.FFN_workers[ffn_id].load_batch(current_time, batch)
                elif batch.status == 1:
                    if batch.attention_now:
                        continue
                    if batch.is_due(current_time):
                        batch.A2F_transmission(current_time, self.alpha_T, self.beta_T)
                        server.current_busy = False
                elif batch.status == 2:
                    if batch.is_due(current_time):
                        batch.F2A_transmission(current_time, self.alpha_T, self.beta_T)
                        server.first_finished_batch_id = server.last_finished_batch_id
                        server.last_finished_batch_id = batch_id

    def do_cycle_work(self, current_time):
        self._server_cycle_for_batch_routing(current_time)

        for server in self.servers:
            if server.compute_memory_usage() > server.memory_capacity + server.dynamic_space:
                server.evict_requests(current_time)
            evicted = server.evict_out_requests(current_time)
            if evicted:
                self.arranger.evict_all_requests(evicted)
                self.stats.record_eviction(len(evicted))

        self.arranger.arrange_requests(current_time)

        # 每 cycle 平衡一次
        self._try_balance(current_time)

        for server in self.servers:
            server.attention_work(current_time, self.alpha_A, self.beta_A)
        for FFN_worker in self.FFN_workers:
            FFN_worker.cycle_work(current_time, self.alpha_F, self.beta_F)

# ===========================================================================
# FFN_type=8: UniformBalancedScheduler
# 均匀分组 + 基于 AF 比 (t_A/t_F) 的动态 swap
# ===========================================================================
class UniformBalancedScheduler:
    """均匀分配 server 到 FFN, 运行时按 AF 比 (t_A/t_F) 做动态平衡 swap.

    AF 比定义: af_ratio(server) = t_A / t_F
      t_A = alpha_A * sum(batch.length) + beta_A   (server 整体 attention 用时)
      t_F = sum_batches(alpha_F * num_req + beta_F) (server 整体 FFN 用时)
      af_ratio 越大 = attention 越主导 = 占 FFN 越少 = 越"友善" (适合挤在人多组).

    初始分配: 按 af_ratio 降序, af_ratio 大的优先塞到当前人数最多的组.

    运行时 swap (每 cycle, 反复直到不满足):
      取人最多组 F_big 里 af_ratio 最小的 server a;
      取人最少组 F_small 里 af_ratio 最大的 server b;
      若 af_ratio(b) - af_ratio(a) >= 1, 交换 a/b 的 FFN 归属; 否则停.
      O(n) 检查 (无需遍历所有 pair).
    """

    def __init__(self, arranger, servers, FFN_workers, stats, buffer, stored_batches,
                 alpha_A, beta_A, alpha_T, beta_T, alpha_F, beta_F, initially_full=True):
        self.arranger = arranger
        self.servers = servers
        self.FFN_workers = FFN_workers
        self.stats = stats
        self.buffer = buffer
        self.stored_batches = stored_batches
        self.alpha_A, self.beta_A = alpha_A, beta_A
        self.alpha_T, self.beta_T = alpha_T, beta_T
        self.alpha_F, self.beta_F = alpha_F, beta_F

        self.num_servers = len(servers)
        self.num_FFN = len(FFN_workers)

        # 用普通 FFN (FCFS), server 必须走 status==4 → A2F_end → load_batch 路径,
        # 不能被 dynamic_matching 跳过 (否则 batch 永远卡在 status==6)
        for server in self.servers:
            server.dynamic_matching = False

        self.AF_match: Dict[int, int] = {}            # server_id -> FFN_id
        self.AF_graph: Dict[int, List[int]] = defaultdict(list)  # FFN_id -> [server_id]

        self.initially_full = initially_full
        if self.initially_full:
            self.do_initialize_filling()

        self._initial_assignment()

    def do_initialize_filling(self):
        tot_batch_size = sum(b.batch_size for b in self.stored_batches.values())
        if self.arranger.num_req_inque < tot_batch_size:
            print(f"Basic number:{self.arranger.num_req_inque}, actually needed:{tot_batch_size}")
            raise ValueError("Not enough requests to fill all batches")
        for batch in self.stored_batches.values():
            self.arranger.do_initial_filling(batch)

    # ------------------------------------------------------------------
    # AF 比 (t_A / t_F)
    # ------------------------------------------------------------------
    def _server_af_ratio(self, server) -> float:
        t_A = self.alpha_A * sum(b.length for b in server.batches.values()) + self.beta_A
        t_F = sum(self.alpha_F * max(b.num_req, 1) + self.beta_F
                  for b in server.batches.values())
        if t_F <= 0:
            return float('inf')
        return t_A / t_F

    # ------------------------------------------------------------------
    # 初始分配: af_ratio 大的优先进人数最多的组
    # ------------------------------------------------------------------
    def _initial_assignment(self):
        """均匀分配, 且 af_ratio 大的 server 进"目标人数更多"的组.

        - base = num_servers // num_FFN; 前 (num_servers % num_FFN) 个 FFN 目标人数 base+1, 其余 base.
        - 按 af_ratio 降序排 server, 优先填"目标人数多 (base+1)"的 FFN, 填满再填 base 的.
        - 每个 FFN 填到各自目标人数就不再接收, 保证人数差 <= 1 (均匀).
        """
        base = self.num_servers // self.num_FFN
        extra = self.num_servers % self.num_FFN     # 前 extra 个 FFN 多 1 人
        # 目标人数: FFN 0..extra-1 是 base+1, 其余是 base
        target_capacity = [base + 1 if f < extra else base for f in range(self.num_FFN)]
        # FFN 填充顺序: 目标人数多的优先 (这样 af 大的 server 落到人多组)
        ffn_order = sorted(range(self.num_FFN), key=lambda f: -target_capacity[f])

        ranked = sorted(self.servers, key=lambda s: -self._server_af_ratio(s))
        idx = 0
        for f in ffn_order:
            cap = target_capacity[f]
            for _ in range(cap):
                if idx >= len(ranked):
                    break
                server = ranked[idx]
                idx += 1
                self.AF_match[server.server_id] = f
                self.AF_graph[f].append(server.server_id)
                server.mapped_FFN_id = f
                for batch in server.batches.values():
                    batch.mapped_FFN_id = f

    # ------------------------------------------------------------------
    # 运行时 swap
    # ------------------------------------------------------------------
    def _try_balance(self, current_time):
        max_iter = 100
        for _ in range(max_iter):
            # 找人最多 / 人最少的非空组
            non_empty = [f for f in range(self.num_FFN) if self.AF_graph[f]]
            if len(non_empty) < 2:
                break
            f_big = max(non_empty, key=lambda f: len(self.AF_graph[f]))
            f_small = min(non_empty, key=lambda f: len(self.AF_graph[f]))
            if f_big == f_small or len(self.AF_graph[f_big]) - len(self.AF_graph[f_small]) <= 0:
                break

            # f_big 里 af_ratio 最小的 server a
            a = min(self.AF_graph[f_big],
                    key=lambda sid: self._server_af_ratio(self.servers[sid]))
            # f_small 里 af_ratio 最大的 server b
            b = max(self.AF_graph[f_small],
                    key=lambda sid: self._server_af_ratio(self.servers[sid]))

            af_a = self._server_af_ratio(self.servers[a])
            af_b = self._server_af_ratio(self.servers[b])

            if af_b - af_a >= 1.0:
                self._apply_swap(current_time, a, b, f_big, f_small)
                print(f"[uniform balance cycle {current_time}] swap server {a} "
                      f"(FFN {f_big}, af={af_a:.2f}) <-> server {b} "
                      f"(FFN {f_small}, af={af_b:.2f})")
            else:
                break   # 最极端的一对都不满足, 停

    def _apply_swap(self, current_time, sa, sb, f1, f2):
        self.AF_match[sa] = f2
        self.AF_match[sb] = f1
        self.AF_graph[f1].remove(sa)
        self.AF_graph[f1].append(sb)
        self.AF_graph[f2].remove(sb)
        self.AF_graph[f2].append(sa)
        self.servers[sa].mapped_FFN_id = f2
        self.servers[sb].mapped_FFN_id = f1
        for batch in self.servers[sa].batches.values():
            batch.mapped_FFN_id = f2
        for batch in self.servers[sb].batches.values():
            batch.mapped_FFN_id = f1
        self.servers[sa].reactivate_batches(current_time, self.alpha_A, self.beta_A,
                                            self.alpha_T, self.beta_T)
        self.servers[sb].reactivate_batches(current_time, self.alpha_A, self.beta_A,
                                            self.alpha_T, self.beta_T)

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def do_cycle_work(self, current_time):
        for sid in range(self.num_servers):
            server = self.servers[sid]
            FFN_server = self.FFN_workers[self.AF_match[sid]]
            server.cycle_work(current_time, self.stats, FFN_server, self.alpha_T, self.beta_T)
            if server.compute_memory_usage() > server.memory_capacity + server.dynamic_space:
                server.evict_requests(current_time)
            evicted = server.evict_out_requests(current_time)
            if evicted:
                self.arranger.evict_all_requests(evicted)
                self.stats.record_eviction(len(evicted))

        self.arranger.arrange_requests(current_time)

        self._try_balance(current_time)

        for server in self.servers:
            server.attention_work(current_time, self.alpha_A, self.beta_A)
        for FFN_worker in self.FFN_workers:
            FFN_worker.cycle_work(current_time, self.alpha_F, self.beta_F)