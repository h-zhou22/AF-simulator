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
    
    def load_batch(self, current_time, batch:Batch):
        self.buffer.append(batch)
        
    def cycle_work(self, current_time, alpha_F, beta_F):
        if self.current_busy:
            if current_time < self.current_ending:
                return
            self.current_busy = False
        if self.buffer:
            batch = self.buffer.pop()
            self.current_ending = batch.FFN_processing(current_time, alpha_F, beta_F)
            self.current_busy = True

class BatchNode:
    def __init__(self, batch_id, batch: Batch):
        self.batch_id = batch_id
        self.batch = batch
        self.load_ready = False # 是否完成了此轮的前序工作做好了FFN的准备
        self.next = None
        self.prev = None

class BatchList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.current = None
        self.batch_count = 0
        self.map = {}

    def add_batch(self, batch: Batch):
        node = BatchNode(batch.batch_id, batch)

        if self.head is None:
            self.head = node
            self.tail = node
            node.next = node
            node.prev = node
        else:
            node.next = self.head
            self.tail.next = node
            node.prev = self.tail
            self.tail = node

        self.map[batch.batch_id] = node

        if self.current is None:
            self.current = self.head

        self.batch_count += 1

    def remove_batch(self, batch_id):
        node = self.map.get(batch_id)
        if node is None:
            raise ValueError(f"Batch {batch_id} not found in BatchList, cannot remove")

        if node == self.head and node == self.tail:
            self.head = None
            self.tail = None
            self.current = None
        else:
            node.prev.next = node.next
            node.next.prev = node.prev

            if node == self.head:
                self.head = node.next
            if node == self.tail:
                self.tail = node.prev
            if node == self.current:
                self.current = node.next

        del self.map[batch_id]
        self.batch_count -= 1

    # FFN worker在Load时直接将状态调成成ready
    # 完成FFN工作后将ready调整成False
    def finish_current_work(self):
        self.current.load_ready = False
        self.current = self.current.next

    def exchange_current_with_next(self):
    # 动态维护流水线执行顺序
        if self.current is None or self.current.next is None:
            raise ValueError("Current node or next node is None, cannot exchange")
        
        next_node = self.current.next
        prev_node = self.current.prev
        current_node = self.current

        # 交换当前节点和下一个节点
        prev_node.next = next_node
        next_node.prev = prev_node
        next_node.next = current_node
        current_node.prev = next_node

        self.current = next_node

        

        # 更新头尾指针
        if self.head == current_node:
            self.head = next_node
        elif self.head == next_node:
            self.head = current_node

        if self.tail == current_node:
            self.tail = next_node
        elif self.tail == next_node:
            self.tail = current_node

    def exchange_current_with_prev(self):
    # 交换当前节点和前一个节点
        if self.current is None or self.current.prev is None:
            raise ValueError("Current node or previous node is None, cannot exchange")
        
        prev_node = self.current.prev
        next_node = self.current.next
        current_node = self.current

        # 交换当前节点和前一个节点
        prev_node.prev.next = current_node
        current_node.prev = prev_node.prev
        current_node.next = prev_node
        prev_node.prev = current_node

        self.current = prev_node

        # 更新头尾指针
        if self.head == current_node:
            self.head = prev_node
        elif self.head == prev_node:
            self.head = current_node

        if self.tail == current_node:
            self.tail = prev_node
        elif self.tail == prev_node:
            self.tail = current_node

    # 还需增添与前序节点交换的方法
    def exchange_current_with_given(self, batch_id):
    # 交换当前节点和指定batch_id的节点
        if self.current is None:
            raise ValueError("Current node is None, cannot exchange")
        

        target_node = self.map.get(batch_id)
        if target_node is None:
            raise ValueError(f"Batch {batch_id} not found in BatchList, cannot exchange")

        prev_node = self.current.prev
        next_node = self.current.next
        if next_node == target_node:
            self.exchange_current_with_next()
            return
        if prev_node == target_node:
            self.exchange_current_with_prev()
            return

        
        if target_node == self.current:
            return

        current_node = self.current

        target_prev = target_node.prev
        target_next = target_node.next
        

        # 交换当前节点和目标节点
        prev_node.next = target_node
        target_node.prev = prev_node

        next_node.prev = target_node
        target_node.next = next_node

        target_prev.next = current_node
        current_node.prev = target_prev

        target_next.prev = current_node
        current_node.next = target_next

        self.current = target_node

        # 更新头尾指针
        if self.head == self.current:
            self.head = target_node
        elif self.head == target_node:
            self.head = self.current

        if self.tail == self.current:
            self.tail = target_node
        elif self.tail == target_node:
            self.tail = self.current

class dynamic_FFN:
    def __init__(self, worker_id, should_serve_num_batches, allow_exchange=False):
        self.worker_id = worker_id
        self.current_busy = False
        self.current_ending = -1
        self.current_pointer_id = 0

        self.allow_exchange = allow_exchange
        
        self.buffer = BatchList()
        self.served_num_batches = 0
        self.should_serve_batches = should_serve_num_batches
    
    def construct_pipeline(self, current_time, batch:Batch):
    # 向流水线中添加一个Batch
        self.buffer.add_batch(batch)
        self.served_num_batches += 1

    def modify_pipeline(self, current_time, batch_id):
        self.buffer.remove_batch(batch_id)
        self.served_num_batches -= 1

    def load_batch(self, current_time, batch:Batch):
        node = self.buffer.map.get(batch.batch_id)
        if node is None:
                raise ValueError(f"Batch {batch.batch_id} not found in FFN pipeline")

        node.load_ready = True
        
    def cycle_work(self, current_time, alpha_F, beta_F):
        if self.current_busy:
            if current_time < self.current_ending:
                return
            self.current_busy = False
        if self.buffer.batch_count > 0:
            # TODO, 请检查此处修改后的逻辑正确情况
            # TODO, 是否允许交换流水线顺序的逻辑
            if not self.allow_exchange:
                if not self.buffer.current.load_ready:
                    return
                else:
                    self.current_busy = True
                    self.current_ending = self.buffer.current.batch.FFN_processing(current_time, alpha_F, beta_F)
                    # 在开始处理时就更改Buffer的状态
                    self.buffer.finish_current_work()
            else:
            # 寻找到下一个就绪的Node并在流水线上与当前节点交换
                if self.buffer.current.load_ready:
                    self.current_busy = True
                    self.current_ending = self.buffer.current.batch.FFN_processing(current_time, alpha_F, beta_F)
                    # 在开始处理时就更改Buffer的状态
                    self.buffer.finish_current_work()
                else:
                    # 寻找下一个就绪的Node
                    next_node = self.buffer.current.next
                    find_ready = False
                    while next_node != self.buffer.current:
                        if next_node.load_ready:
                            # 交换当前节点和下一个就绪的节点
                            self.buffer.exchange_current_with_given(next_node.batch_id)
                            find_ready = True
                            break
                        next_node = next_node.next
                    if find_ready:
                        self.current_busy = True
                        self.current_ending = self.buffer.current.batch.FFN_processing(current_time, alpha_F, beta_F)
                        # 在开始处理时就更改Buffer的状态
                        self.buffer.finish_current_work()
            # batch = self.buffer.pop()
            # self.current_ending = batch.FFN_processing(current_time, alpha_F, beta_F)
            # self.current_busy = True

