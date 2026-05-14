import random
from request import Request
from batch import Batch
from typing import List, Dict, Tuple
from collections import deque
import math

class FFN:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.current_busy = False
        self.current_ending = -1
        self.buffer = deque()
        # 当前队列长度
        self.current_queue_length = 0

        self.ceiling = True
    
    def load_batch(self, current_time, batch:Batch):
        self.buffer.append(batch)
        
    def cycle_work(self, current_time, alpha_F, beta_F):
        if self.current_busy:
            if self.ceiling:
                if current_time < self.current_ending:
                    return
            else:
                if current_time < self.current_ending:
                    return
            self.current_busy = False
        if self.buffer:
            batch = self.buffer.popleft()
            self.current_ending = batch.FFN_processing(current_time, alpha_F, beta_F, current_ending=self.current_ending)
            self.current_busy = True

class BatchNode:
    def __init__(self, batch_id, batch: Batch):
        self.batch_id = batch_id
        self.batch = batch
        self.load_ready = False # 是否完成了此轮的前序工作做好了FFN的准备
        self.net :BatchNode = None
        self.prev :BatchNode = None

class BatchList:
    def __init__(self):
        self.head:BatchNode = None
        self.tail:BatchNode = None
        self.current:BatchNode = None
        self.batch_count = 0
        self.map = {}
    
    def print_all_batches(self):
    # For debug usage, print all the batches loaded
        print("Total batch count: ",self.batch_count)
        
        head_id = self.head.batch_id
        
        print("HeadBatch ID: ", head_id)
        iter_node = self.head.net
        #iter_node = iter_node.net
        while iter_node.batch_id != head_id:
            print("Batch ID: ", iter_node.batch_id)
            iter_node = iter_node.net

    def add_batch(self, batch: Batch):
        node = BatchNode(batch.batch_id, batch)
        
        if self.head is None:
            self.head = node
            self.tail = node
            node.net = node
            node.prev = node
        else:
            node.net = self.head
            self.head.prev = node
            self.tail.net = node
            node.prev = self.tail
            self.tail = node

        self.map[batch.batch_id] = node

        if self.current is None:
            self.current = self.head

        self.batch_count += 1

    def remove_batch(self, batch_id):
        #raise NotImplementedError("remove_batch is not implemented yet")
        node = self.map.get(batch_id)
        if node is None:
            raise ValueError(f"Batch {batch_id} not found in BatchList, cannot remove")

        if node == self.head and node == self.tail:
            self.head = None
            self.tail = None
            self.current = None
        else:
            node.prev.net = node.net
            node.net.prev = node.prev

            if node == self.head:
                self.head = node.net
            if node == self.tail:
                self.tail = node.prev
            if node == self.current:
                self.current = node.net

        del self.map[batch_id]
        self.batch_count -= 1

    # FFN worker在Load时直接将状态调成成ready
    # 完成FFN工作后将ready调整成False
    def finish_current_work(self):
        self.current.load_ready = False
        self.current = self.current.net

    def exchange_current_with_next(self):
        if self.current is None or self.current.net is None or self.current.net is self.current:
            raise ValueError("Current node or next node is None/self, cannot exchange")

        prev_node    = self.current.prev
        current_node = self.current
        next_node    = current_node.net
        nn_node      = next_node.net  # 新增: 必须保留 next.net, 否则会断链

        # 二节点环 (A == next.net, prev_node 其实就是 next_node) 的退化情况
        if nn_node is current_node:
            # 双向环只有 current 和 next, 拓扑上 swap 后等价于原图,
            # 仅 head/tail/current 标记需要调整
            if self.head is current_node:
                self.head, self.tail = next_node, current_node
            else:
                self.head, self.tail = current_node, next_node
            self.current = next_node
            return

        # 通用情况: A <-> B <-> C <-> D  =>  A <-> C <-> B <-> D
        prev_node.net   = next_node
        next_node.prev   = prev_node
        next_node.net   = current_node
        current_node.prev = next_node
        current_node.net = nn_node      # 新增
        nn_node.prev      = current_node # 新增

        if self.head is current_node:
            self.head = next_node
        elif self.head is next_node:
            self.head = current_node

        if self.tail is current_node:
            self.tail = next_node
        elif self.tail is next_node:
            self.tail = current_node

        self.current = next_node

    def exchange_current_with_prev(self):
        if self.current is None or self.current.prev is None or self.current.prev is self.current:
            raise ValueError("Current node or previous node is None/self, cannot exchange")

        prev_node    = self.current.prev
        current_node = self.current
        next_node    = current_node.net
        pp_node      = prev_node.prev   # 新增: 保留 prev.prev

        if pp_node is current_node:
            # 二节点环退化情况
            if self.head is current_node:
                self.head, self.tail = prev_node, current_node
            else:
                self.head, self.tail = current_node, prev_node
            self.current = prev_node
            return

        # X <-> P <-> B <-> N  =>  X <-> B <-> P <-> N
        pp_node.net      = current_node
        current_node.prev = pp_node
        current_node.net = prev_node
        prev_node.prev    = current_node
        prev_node.net    = next_node     # 新增
        next_node.prev    = prev_node     # 新增

        if self.head is current_node:
            self.head = prev_node
        elif self.head is prev_node:
            self.head = current_node

        if self.tail is current_node:
            self.tail = prev_node
        elif self.tail is prev_node:
            self.tail = current_node

        self.current = prev_node

    # 还需增添与前序节点交换的方法
    def exchange_current_with_given(self, batch_id):
    # 交换当前节点和指定batch_id的节点
        if self.current is None:
            raise ValueError("Current node is None, cannot exchange")
        

        target_node = self.map.get(batch_id)
        if target_node is None:
            raise ValueError(f"Batch {batch_id} not found in BatchList, cannot exchange")

        prev_node = self.current.prev
        next_node = self.current.net
        if next_node == target_node:
            self.exchange_current_with_next()
            return
        if prev_node == target_node:
            self.current = prev_node
            return

        
        if target_node == self.current:
            return
        #raise NotImplementedError("Exchange current with given node is not implemented")

        current_node = self.current

        # 1) 把 target 从原位置摘出
        t_prev = target_node.prev
        t_next = target_node.net
        t_prev.net = t_next
        t_next.prev = t_prev
        if self.head is target_node:
            self.head = t_next
        if self.tail is target_node:
            self.tail = t_prev

        # 2) 插入到 current 前面: ... <-> c_prev <-> target <-> current <-> ...
        c_prev = current_node.prev
        c_prev.net = target_node
        target_node.prev = c_prev
        target_node.net = current_node
        current_node.prev = target_node

        # 3) 若 current 原本是 head, 现在 target 排到了它前面, target 成为新 head
        if self.head is current_node:
            self.head = target_node

        self.current = target_node

    def replace_batch_with(self, old_batch_id, new_batch: Batch,
                        inherit_load_ready: bool = False) -> 'BatchNode':
        """用 new_batch 原位替换 old_batch_id 所在的节点. 返回旧节点 (可能是 ghost).

        mark_old_ephemeral=False: 旧节点 prev/next 清空, 直接交给 GC.
        mark_old_ephemeral=True : 旧节点 ephemeral=True, 保留 next 指向其原后继
                                (单向悬挂, 双向链表里已经看不到它了);
                                调用方负责后续 dispose_ephemeral.
        inherit_load_ready=True : 新节点继承旧节点的 load_ready (一般用于 NEW 已就绪的情形).
        """
        old_node = self.map.get(old_batch_id)
        if old_node is None:
            raise ValueError(f"Batch {old_batch_id} not found in BatchList, cannot replace")
        if new_batch.batch_id in self.map:
            existing_node = self.map[new_batch.batch_id]
            existing = existing_node.batch
            raise ValueError(
                f"Batch {new_batch.batch_id} already in BatchList; cannot replace.\n"
                f"  trying to insert: id={new_batch.batch_id}, "
                f"mapped_FFN_id={new_batch.mapped_FFN_id}\n"
                f"  existing entry  : id={existing.batch_id}, "
                f"mapped_FFN_id={existing.mapped_FFN_id}\n"
                f"  existing is new_batch? {existing is new_batch}\n"
                f"  self (BatchList of FFN)... old_batch_id passed in = {old_batch_id}"
            )
        new_node = BatchNode(new_batch.batch_id, new_batch)
        new_node.load_ready = old_node.load_ready if inherit_load_ready else False

        # 单节点环
        if old_node is self.head and old_node is self.tail:
            new_node.net = new_node
            new_node.prev = new_node
            self.head = new_node
            self.tail = new_node
        else:
            prev_node = old_node.prev
            next_node = old_node.net
            new_node.prev = prev_node
            new_node.net = next_node
            prev_node.net = new_node
            next_node.prev = new_node
            if self.head is old_node:
                self.head = new_node
            if self.tail is old_node:
                self.tail = new_node
        
        if self.current is old_node:
            self.current = new_node

        del self.map[old_batch_id]
        self.map[new_batch.batch_id] = new_node
        # batch_count 不变 (1-to-1)

        return old_node
    
# FFN只维护自身流水线的状态，BatchNode可以与实际的Batch无关
# Batchlist只有顺序是重要的。如果当前Node被交换, 那么current node的Batch可能并非实际处理的Batch
class dynamic_FFN:
# 可以改变流水线顺序的FFN worker, 可以寻找下一个准备就绪的Batch而非严格按照顺序执行
    def __init__(self, worker_id, should_serve_num_batches, allow_exchange=False):
        self.worker_id = worker_id
        self.current_busy = False
        self.current_ending = -1

        self.processing_batch_id = -1     # 当前正在被 FFN 处理的节点
        

        self.allow_exchange = allow_exchange
        
        self.buffer = BatchList()
        self.served_num_batches = 0
        self.should_serve_batches = should_serve_num_batches

        self.ceiling = True
    
    def construct_pipeline(self, current_time, batch:Batch):
    # 向流水线中添加一个Batch
        # if self.worker_id == 6:
        #     print("Appended Batch {} at time {}".format(batch.batch_id, current_time))
        if batch.mapped_FFN_id != -1:
            print("Current cycle: ", current_time)
            raise ValueError(f"Batch {batch.batch_id} has already been matched to FFN {batch.mapped_FFN_id}, cannot add to FFN {self.worker_id}")
        batch.mapped_FFN_id = self.worker_id
        self.buffer.add_batch(batch)
        print("Batch ID: {}, Mapped to FFN ID: {}".format(batch.batch_id, self.worker_id))
        self.served_num_batches += 1

    def modify_pipeline(self, current_time, batch_id):
        removed_node = self.buffer.map.get(batch_id)
        removed_batch = removed_node.batch
        removed_batch.mapped_FFN_id = -1

        self.buffer.remove_batch(batch_id)
        self.served_num_batches -= 1
    
    def debug_print_pipeline(self):
        print("FFN ID: ", self.worker_id)
        self.buffer.print_all_batches()
    def load_batch(self, current_time, batch:Batch):
        node = self.buffer.map.get(batch.batch_id)
        if node is None:
            print("FFN ID: ", self.worker_id)
            self.buffer.print_all_batches()
            print("Current cycle: ", current_time)
            raise ValueError(f"Batch {batch.batch_id} not found in FFN pipeline")

        node.load_ready = True
        
    def cycle_work(self, current_time, alpha_F, beta_F):
        if self.current_busy:
            if current_time < self.current_ending:
                return
            self.current_busy = False
            
            self.processing_batch_id = -1

        # 此时current_busy=False, 寻找下一个处理的BatchNode    
        if self.buffer.batch_count > 0:
            # TODO, 请检查此处修改后的逻辑正确情况
            # TODO, 是否允许交换流水线顺序的逻辑
            if not self.allow_exchange:
                if not self.buffer.current.load_ready:
                    return
                else:
                    self.current_busy = True
                    self.processing_batch_id = self.buffer.current.batch_id
                    self.current_ending = self.buffer.current.batch.FFN_processing(current_time, alpha_F, beta_F, current_ending=self.current_ending)
                    # current_node事实上被切换到了下一个节点
                    self.buffer.finish_current_work()
                    
            else:
            # 寻找到下一个就绪的Node并在流水线上与当前节点交换
                if self.buffer.current.load_ready:
                    self.current_busy = True
                    self.processing_batch_id = self.buffer.current.batch_id
                    self.current_ending = self.buffer.current.batch.FFN_processing(current_time, alpha_F, beta_F, current_ending=self.current_ending)
                    self.buffer.finish_current_work()
                    
                else:
                    
                    # 寻找下一个就绪的Node
                    next_node = self.buffer.current.net
                    find_ready = False
                    while next_node != self.buffer.current:
                        if next_node.load_ready:
                            # 交换当前节点和下一个就绪的节点
                            self.buffer.exchange_current_with_given(next_node.batch_id)
                            find_ready = True
                            break
                        next_node = next_node.net
                    if find_ready:
                        #raise NotImplementedError("Exchange logic not implemented yet")
                        self.processing_batch_id = self.buffer.current.batch_id
                        self.current_busy = True
                        self.current_ending = self.buffer.current.batch.FFN_processing(current_time, alpha_F, beta_F, current_ending=self.current_ending)
                        # 在开始处理时就更改Buffer的状态
                        self.buffer.finish_current_work()

    def replace_batch(self, current_time, old_batch_id, new_batch: Batch,
                  inherit_load_ready: bool = False):
        """用 new_batch 替换 pipeline 里 old_batch_id 对应的 batch, 保持其在流水线中的位置.
        允许替换任意状态的 batch:
- status==2 (in-flight): old_batch 通过 current_ending 时间戳完成当前轮 FFN 工作,
  new_batch 占据其 pipeline 槽位等待下一轮. 其实也没有太大影响,正常load即可
- status==6 (已 load 但未开始): 由 Attention 侧负责丢弃当前轮 FFN 工作并重做 A2F.
- status==4 (已传输但尚未到达): 由 Attention 侧负责丢弃当前轮 FFN 工作并重做 A2F.
- 其他状态: 仅做节点身份替换.

调用方约定: 通常仅由 swap_batches_between_ffns 调用, 后者保证两边对换原子完成.
        """
        old_node = self.buffer.map.get(old_batch_id)
        if old_node is None:
            raise ValueError(
                f"Batch {old_batch_id} not found in FFN {self.worker_id} pipeline, cannot replace")

        # 此处有待商榷, 已经匹配的Batch仍可以被重新分配 
        # if new_batch.mapped_FFN_id != -1:
        #     raise ValueError(
        #         f"Batch {new_batch.batch_id} already matched to FFN {new_batch.mapped_FFN_id}, "
        #         f"cannot place into FFN {self.worker_id}")

        # 不能取消id，否则先交换的batch的FFN_id会被覆盖
        # old_node.batch.mapped_FFN_id = -1

        # 原位替换
        self.buffer.replace_batch_with(old_batch_id, new_batch,
                                    inherit_load_ready=inherit_load_ready)

        # 绑定新 batch
        new_batch.mapped_FFN_id = self.worker_id
        # served_num_batches / should_serve_batches 不变 (1-to-1 swap)

class MoEFFN:
    """单 expert MoE worker. 每次处理一个 (request, expert_id) 任务."""

    def __init__(self, worker_id, expert_id, alpha_F, beta_F):
        self.worker_id = worker_id
        self.expert_id = expert_id
        self.alpha_F = alpha_F
        self.beta_F = beta_F

        self.current_busy = False
        self.current_ending = -1
        self.current_request = None
        self.current_batch = None

    def load_task(self, current_time, request, batch):
        assert not self.current_busy
        self.current_busy = True
        self.current_request = request
        self.current_batch = batch
        cost = self.alpha_F * 1 + self.beta_F
        self.current_ending = current_time + cost

    def cycle_work(self, current_time, *args, **kwargs):
        """*args, **kwargs 是为了和 FFN.cycle_work / dynamic_FFN.cycle_work 签名兼容
        (它们要 alpha_F, beta_F), 但 MoEFFN 在构造时已经 bake 进去, 这里忽略."""
        if not self.current_busy:
            return
        if current_time + 1 <= self.current_ending:
            return
        # 完成
        req = self.current_request
        batch = self.current_batch
        batch.on_moe_expert_done(current_time, req)
        self.current_busy = False
        self.current_request = None
        self.current_batch = None