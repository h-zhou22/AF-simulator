import json
import os
from request import Request
from batch import Batch
from collections import defaultdict

class StatsCollector:
    def __init__(self, prefix: str=""):
        self.records = []  
        self.batch_info = []

        self.finished_request = 0   
        self.tot_increase_length = 0
        self.total_generated_tokens = 0
        
        self.total_rounds = 0
        self.total_final_length = 0
        self.total_avg_round_time = 0
        self.count_avg_round = 0   
        
        self.prefix = prefix
        self.output_dir = "result"
        self.length_distribution = {
            "1-256": 0,
            "257-512": 0,
            "513-1024": 0,
            "1025-2048": 0,
            "2049-4096": 0,
            ">4096": 0
        }
        os.makedirs(self.output_dir, exist_ok=True)
        

    def record(self, req: Request):
        """
        Record statistics for a completed request, with precise semantics:
        - lifecycle time
        - eviction statistics
        - pause time due to eviction
        - processing time per round
        """

        # -------- 基本完成计数 --------
        #print("Next token probability:", req.next_token_prob)
        self.finished_request += 1
        if req.original_len <= 256:
            self.length_distribution["1-256"]+=1
        elif 257 <= req.original_len <= 512:
            self.length_distribution["257-512"]+=1
        elif 513 <= req.original_len <= 1024:
            self.length_distribution["513-1024"]+=1   
        elif 1025 <= req.original_len <= 2048:
            self.length_distribution["1025-2048"]+=1
        elif 2049 <= req.original_len <= 4096:
            self.length_distribution["2049-4096"]+=1
        else:
            self.length_distribution[">4096"]+=1
        

        # -------- 生命周期时间 --------
        increase_length = req.length - req.original_len
        self.tot_increase_length += increase_length
        if req.arrival is not None and req.completion_time is not None:
            total_time = req.completion_time - req.start_processing_time
        else:
            total_time = None

        if req.rounds > 0 and total_time is not None:
            avg_time_per_round = total_time / req.rounds
        else:
            avg_time_per_round = None

        # -------- 累计全局统计 --------
        if req.rounds > 0:
            self.total_generated_tokens += req.rounds
            self.total_rounds += req.rounds

        if req.length is not None:
            self.total_final_length += req.length

        # -------- 记录单 request 数据 --------
        self.records.append({
            "rid": req.rid,

            # lifecycle
            "startal_time": req.start_processing_time,
            "completion_time": req.completion_time,
            "total_time": total_time,
            "avg_time_per_round": avg_time_per_round,

            # rounds & processing
            "rounds": req.rounds,
            
            # lengths
            "initial_length": req.original_len,
            "final_length": req.length,

        })

        
    def record_batch(self, batch: Batch):
        rounds = 0
        tot_cost = 0
        if len(batch.round_cost)>0:
            for round_cost in batch.round_cost:
                rounds += 1
                tot_cost += round_cost
        avg_cost = tot_cost/rounds

        self.batch_info.append(
            {
                "batch_id": batch.bids,
                "served_requests": batch.ever_served_request,
                "Acost": batch.Acost,
                "Fcost": batch.Fcost,
                "Round_cost": batch.round_cost,
                "Avg_Round_cost": avg_cost
            }
        )

    def summary(self):
        """
        Global summary statistics
        """
        if not self.records:
            return {}

        total_time_sum = 0
        total_cycle_time_sum = 0
        cycle_time_count = 0

        # length buckets
        buckets = {
            "1-256": [],
            "257-512": [],
            "513-1024": [],
            "1025-2048": [],
            "2049-4096": [],
            ">4096": []
        }

        for r in self.records:
            arrival = r["startal_time"]
            completion = r["completion_time"]
            rounds = r["rounds"]
            init_len = r["initial_length"]

            if completion is None:
                continue

            total_time = completion - arrival
            total_time_sum += total_time

            # avg per-cycle time (only if rounds > 0)
            if rounds > 0:
                total_cycle_time_sum += total_time / rounds
                cycle_time_count += 1

            # bucket classification
            if 1 <= init_len <= 256:
                buckets["1-256"].append(total_time)
            elif 257 <= init_len <= 512:
                buckets["257-512"].append(total_time)
            elif 513 <= init_len <= 1024:
                buckets["513-1024"].append(total_time)
            elif 1025 <= init_len <= 2048:
                buckets["1025-2048"].append(total_time)
            elif 2049 <= init_len <= 4096:
                buckets["2049-4096"].append(total_time)
            else:
                buckets[">4096"].append(total_time)

        avg_total_time = total_time_sum / self.finished_request if self.finished_request > 0 else None
        avg_cycle_time = (
            total_cycle_time_sum / cycle_time_count
            if cycle_time_count > 0 else None
        )

        bucket_avg_time = {
            k: (sum(v) / len(v) if v else None)
            for k, v in buckets.items()
        }

        total_batch = 0
        batch_round_cost = 0
        for b in self.batch_info:
            total_batch += 1
            batch_round_cost += b["Avg_Round_cost"]
        batch_round_cost /= total_batch

        return {
            "finished_requests": self.finished_request,
            #"vip_requests": len(buckets["vip"]),
            "avg_total_time": avg_total_time,
            "avg_time_per_cycle_per_request": avg_cycle_time,
            "avg_total_time_by_initial_length": bucket_avg_time,
            "finished count": self.length_distribution,

            "num_batches": total_batch,
            "avg_batch_cost": batch_round_cost
        }

    def dump_batch_info_to_json(self):
        """将所有 request 记录输出到 JSON 文件"""
        path = os.path.join(
            self.output_dir,
            f"{self.prefix}_batch_info.json"
        )
        with open(path, "w") as f:
            json.dump(self.batch_info, f, indent=2)

    def dump_records_to_json(self):
        """将所有 request 记录输出到 JSON 文件"""
        path = os.path.join(
            self.output_dir,
            f"{self.prefix}_records.json"
        )
        with open(path, "w") as f:
            json.dump(self.records, f, indent=2)

    def dump_summary_to_json(self):
        """将 summary 统计输出到 JSON 文件"""
        summary_data = self.summary()
        filename = os.path.join(
            self.output_dir,
            f"{self.prefix}_summary.json"
        )
        with open(filename, "w") as f:
            json.dump(summary_data, f, indent=2)

