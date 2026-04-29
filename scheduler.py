import random
import math
from request import Request
from batch import Batch
from attention import Server
from typing import List, Dict, Tuple
from collections import defaultdict
from FFN import FFN, dynamic_FFN

class BasicScheduler:
# AF之间固定匹配不再更改
    def __init__(self, servers:List[Server], FFN_workers:List[FFN], stats, buffer, stored_batches:Dict[int, Batch], alpha_A, beta_A, alpha_T, beta_T, alpha_F, beta_F, initially_full=False):
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
        self.AF_match : Dict[int, int] # server_id -> FFN_id

        self.initially_full = initially_full
        if self.initially_full:
            self.do_initialize_filling()
        self.match_AF()

    def do_initialize_filling(self):
        if self.buffer().size() < self.num_batches*self.batch_size:
            raise ValueError("Buffer size is smaller than total batch capacity")
        for batch in self.stored_batches.values():
            if batch.num_req < batch.batch_size:
                request = self.buffer.pop()
                batch.append_request(0, request)
                # load_request加入之后会立刻开始处理, 所以初始化都用append

    def match_AF(self):
        # match each server to a given FFN worker
        for i in range (self.num_servers):
            server = self.servers[i]
            FFN_worker = self.FFN_workers[i % self.num_FFN]
            server.FFN_worker = FFN_worker
            self.AF_match[server.server_id] = FFN_worker.worker_id

    def do_cycle_work(self, current_time):
            for server_id in len(self.servers):
            # 这里假定了server和FFN的ID是按照顺序排列的，后续Debug的时候需注意
                server = self.servers[server_id]
                FFN_server = self.FFN_workers[self.AF_match[server_id]]
                server.cycle_work(current_time, self.stats, FFN_server, self.alpha_T, self.beta_T)

            available_batches : List[Tuple[int, int, int, int]] = []
            for server in self.servers:
                extend_batches = server.find_available_batch()
                available_batches.extend(extend_batches)

            while available_batches and self.buffer:
                request = self.buffer.pop()
                best_batch_info = min(available_batches)
                batch_id0 = best_batch_info[2]
                server_id0 = best_batch_info[3]
                best_batch = self.stored_batches[batch_id0]
                target_server = self.servers[server_id0]
                target_server.load_request_to_batch(current_time, best_batch_info[2], request)
                available_batches.remove(best_batch_info)
                if best_batch.has_free_slot(current_time):
                    info0, info1 = best_batch.updated_info(current_time=current_time)
                    new_info = (info0, info1, batch_id0, server_id0)
                    available_batches.append(new_info)
            
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

class level_FFN:
    FFN_id: int
    FFN_worker: FFN
    # 应该为几个server服务
    marked_cnt: int
    # 服务的server位于第几层, 如跨层则选取其中较高的为标准
    level: int
    # 是否实际上为多个不同的server服务
    mixed_server_level: bool
    # 实际为几个server服务
    server_cnt: int

class unbalanced_server_info:
    server_id: int
    FFN_id: int
    current_mapped_level: int
    current_size_level: int

class PipelineScheduler:
# AF之间存在固定匹配，可能会被动态修改
    def __init__(self, servers:List[Server], FFN_workers:List[dynamic_FFN], stats, buffer, stored_batches:Dict[int, Batch], alpha_A, beta_A, alpha_T, beta_T, alpha_F, beta_F, initially_full=False):
        self.servers = servers
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
        
        self.unbalanced_servers : List[unbalanced_server_info] = []

        self.FFN_lower_bound = 0
        self.FFN_upper_bound = len(self.FFN_workers)
        # 0 For Proper. 1 For Too many FFN, 2 For Too many Attention
        self.currently_utilize_status = 0

        self.FFN_table = [[] for _ in range(self.max_AF_ratio + 1)]

        self.initially_full = initially_full
        # 先把所有的Attention装满
        if self.initially_full:
            self.do_initialize_filling()
        # TODO: 初始化AF匹配策略以及后续AF匹配的动态调整

        self.do_initial_matching()

    def do_initialize_filling(self):
        if self.buffer().size() < self.num_batches*self.batch_size:
            raise ValueError("Buffer size is smaller than total batch capacity")
        for batch in self.stored_batches.values():
            while batch.num_req < batch.batch_size:
                request = self.buffer.pop()
                batch.append_request(0, request)
                # load_request加入之后会立刻开始处理, 所以初始化都用append
            batch.start_processing_from_empty(0)

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
                        self.AF_graph[FFN_id].append(server_id)
                    
                    level_remainder = 0
                    level_remain_list.clear()

                for i in range(server_to_fill):
                    server_id = self.AF_ratio_list[level][level_cnt-i-1]
                    self.AF_match[server_id] = FFN_id
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
        
        # 下面使用多余的FFN为部分前面的FFN分担overhead的server
        # if FFN_remaining > 0:
        #     for level in reversed(range(self.max_AF_ratio + 1)):
        #         modified_server_cnt = 0
        #         if len(self.FFN_table[level]) <= level:
        #             continue 
        #         for level_iter_FFN in self.FFN_table[level]:
        #             if FFN_remaining <= 0:
        #                 break
        #             additional_worker_id = self.num_FFN - FFN_remaining
        #             additional_FFN = self.FFN_workers[additional_worker_id] 
        #             if level_iter_FFN.mixed_server_level:
        #                 continue
                    
        #             if modified_server_cnt >= level:
        #                 additional_level_FFN = level_FFN(FFN_id= self.num_FFN - FFN_remaining, FFN_worker = self.FFN_workers[self.num_FFN - FFN_remaining], marked_cnt=level, level=level, mixed_server_level=False, server_cnt=level)
        #             FFN_remaining -= 1
        #         if FFN_remaining <= 0:
        #             break
        # 下面用多余的 FFN 为部分前面的 FFN 分担 overhead
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


    def update_AF_ratio_bounds(self):
        # 再计算恰好匹配的情况下需要的FFN worker的数量区间
        cnt_low = 0
        for i in range(self.max_AF_ratio + 1):
            tot_server = self.AF_ratio_count[i]
            if i == self.max_AF_ratio:
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
            self.server_FFN_unit_count[server.server_id] = server.compute_total_unit_cost(
                self.alpha_A, self.beta_A)
        self.server_FFN_unit_order = sorted(
            self.server_FFN_unit_count.items(), key=lambda x: x[1], reverse=True)
    

    def do_cycle_work(self, current_time):
            for server_id in len(self.servers):
            # 这里假定了server和FFN的ID是按照顺序排列的，后续Debug的时候需注意
                server = self.servers[server_id]
                FFN_server = self.FFN_workers[self.AF_match[server_id]]
                server.cycle_work(current_time, self.stats, FFN_server, self.alpha_T, self.beta_T)

            available_batches : List[Tuple[int, int, int, int]] = []
            for server in self.servers:
                extend_batches = server.find_available_batch()
                available_batches.extend(extend_batches)

            while available_batches and self.buffer:
                request = self.buffer.pop()
                best_batch_info = min(available_batches)
                batch_id0 = best_batch_info[2]
                server_id0 = best_batch_info[3]
                best_batch = self.stored_batches[batch_id0]
                target_server = self.servers[server_id0]
                target_server.load_request_to_batch(current_time, best_batch_info[2], request)
                available_batches.remove(best_batch_info)
                if best_batch.has_free_slot(current_time):
                    info0, info1 = best_batch.updated_info(current_time=current_time)
                    new_info = (info0, info1, batch_id0, server_id0)
                    available_batches.append(new_info)
            
            # 在free slot被填补之后，关注各个attention以及各个Batch的大小变化
            for server in self.servers:
                server.update_FFN_level(self.alpha_A, self.beta_A)
            exchange_pairs = self.find_exchange_pairs()
            for pair in exchange_pairs:
                self.apply_swap_pair(current_time, pair[0], pair[1])
                # TODO 在server当中维护两个Batch的归属信息、状态信息
                self.servers[pair[0]]

            for server in self.servers:
                server.attention_work(current_time, self.alpha_A, self.beta_A)
            for FFN_worker in self.FFN_workers:
                FFN_worker.cycle_work(current_time, self.alpha_F, self.beta_F)

    def apply_swap_pair(self, current_time: int, server_id_a: int, server_id_b: int):
        """交换两个 server 各自维护的两个 batch 所归属的 FFN.
        只负责交换,关于Batch交换后状态的维护需要额外的逻辑进行处理"""
        server_a = self.servers[server_id_a]
        server_b = self.servers[server_id_b]
        ffn_id_a = self.AF_match[server_id_a]
        ffn_id_b = self.AF_match[server_id_b]
        if ffn_id_a == ffn_id_b:
            return   # 同一 FFN, 不需要 swap

        ffn_a = self.FFN_workers[ffn_id_a]
        ffn_b = self.FFN_workers[ffn_id_b]

        # server_a 的两个 batch 全部从 ffn_a 搬到 ffn_b,
        # server_b 的两个 batch 全部从 ffn_b 搬到 ffn_a.
        # 用 swap_batches_between_ffns 一次处理一对 batch (a 的一个 + b 的一个).
        a_batches = list(server_a.batches)
        b_batches = list(server_b.batches)
        assert len(a_batches) == len(b_batches), \
            f"server batches mismatch: {len(a_batches)} vs {len(b_batches)}"

        for batch_a, batch_b in zip(a_batches, b_batches):
            self.swap_batches_between_ffns(ffn_a, batch_a, ffn_b, batch_b)

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


    def _ffn_level_of(self, ffn_id: int) -> int:
        """从 FFN_table 反查 level. swap 不改 level 字段, 此处 ffn_id 一定能找到."""
        for level in range(self.max_AF_ratio + 1):
            for lf in self.FFN_table[level]:
                if lf.FFN_id == ffn_id:
                    return level
        raise ValueError(f"FFN {ffn_id} not found in FFN_table")   

    def _server_target_level(self, server) -> int:
        """该 server 按当前权值'本应'所属的 level."""
        return min(math.floor(server.current_weight), self.max_AF_ratio)

    def find_swap_pairs(self) -> List[Tuple[int, int]]:
        """返回 [(server_id_a, server_id_b), ...].

        匹配条件: a 当前在 La 层但应该去 Lb 层; b 当前在 Lb 层但应该去 La 层.
        每个 server 至多出现在一对中.
        """
        deviated_by_pair: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for server in self.servers:
            cur_L = server.current_mapped_level
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


class DynamicScheduler:
    def __init__(self, servers:List[Server], FFN_workers:List[dynamic_FFN], stats, alpha_A, beta_A, alpha_T, beta_T, alpha_F, beta_F):
        self.servers = servers
        self.FFN_workers = FFN_workers
        self.stats = stats
        self.alpha_A = alpha_A
        self.beta_A = beta_A
        self.alpha_T = alpha_T
        self.beta_T = beta_T
        self.alpha_F = alpha_F
        self.beta_F = beta_F