import json
import os
from request import Request
from batch import Batch
from collections import defaultdict


# ---- 长度桶 (按 original_len) ----
LENGTH_BUCKETS = ("1-1024", "1025-4096", "4097-8192", ">8192")

def _length_bucket(length):
    if length <= 1024:
        return "1-1024"
    elif length <= 4096:
        return "1025-4096"
    elif length <= 8192:
        return "4097-8192"
    else:
        return ">8192"


class StatsCollector:
    def __init__(self, prefix: str = ""):
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
        self.length_distribution = {b: 0 for b in LENGTH_BUCKETS}
        os.makedirs(self.output_dir, exist_ok=True)

        self.batch_avg_attention = []

        # ---- per-actual_type / per-agent / per-agent_type ----
        # actual_type ∈ {0,1,2,3,4}, 用 -2 兜底未知 (一般不会发生)
        self._at_count           = defaultdict(int)
        self._at_proc_time_sum   = defaultdict(float)   # processing time, 不含排队
        self._at_total_time_sum  = defaultdict(float)   # total time, 含排队 + evict
        # agent_id (-1 表示无 agent)
        self._aid_count          = defaultdict(int)
        self._aid_proc_time_sum  = defaultdict(float)
        self._aid_total_time_sum = defaultdict(float)
        # agent_type (-1 表示无 agent_type)
        self._atype_count          = defaultdict(int)
        self._atype_proc_time_sum  = defaultdict(float)
        self._atype_total_time_sum = defaultdict(float)
        self._atype_agent_ids      = defaultdict(set)   # 该 type 出现过的 distinct agent_id 集合

        # ---- evict 计数 ----
        self.total_evict_count = 0     # request-level: 每被 evict 一次 +1
        # 同一 request 多次被 evict 都各自计入

    # ---- 由 scheduler 在 evict 后调用 ----
    def record_eviction(self, n: int = 1):
        self.total_evict_count += n

    def record(self, req: Request):
        record_print = True
        if record_print:
            print("Finished request: {}, Actual type:{}, Predicted Type:{}, Cycle:{}".format(
                self.finished_request, req.actual_type, req.predicted_type, req.completion_time))

        self.finished_request += 1
        bucket = _length_bucket(req.original_len)
        self.length_distribution[bucket] += 1

        increase_length = req.length - req.original_len
        self.tot_increase_length += increase_length

        # processing time: 不含排队 (从被 batch 接纳到完成)
        if req.completion_time is not None and req.start_processing_time is not None:
            proc_time = req.completion_time - req.start_processing_time
        else:
            proc_time = None

        # total time with queue: 含排队 + evict 间隔 (从生成到完成)
        gen_t = getattr(req, "generated_time", None)
        if gen_t is None:
            gen_t = req.arrival
        if req.completion_time is not None and gen_t is not None:
            total_time_q = req.completion_time - gen_t
        else:
            total_time_q = None

        avg_time_per_round = (proc_time / req.rounds) if (req.rounds > 0 and proc_time is not None) else None

        if req.rounds > 0:
            self.total_generated_tokens += req.rounds
            self.total_rounds += req.rounds
        if req.length is not None:
            self.total_final_length += req.length

        # ---- per-actual_type 累积 ----
        at = req.actual_type if req.actual_type is not None else -2
        self._at_count[at] += 1
        if proc_time is not None:
            self._at_proc_time_sum[at] += proc_time
        if total_time_q is not None:
            self._at_total_time_sum[at] += total_time_q

        # ---- per-agent_id ----
        aid = getattr(req, "agent_id", None)
        if aid is None:
            aid = -1
        self._aid_count[aid] += 1
        if proc_time is not None:
            self._aid_proc_time_sum[aid] += proc_time
        if total_time_q is not None:
            self._aid_total_time_sum[aid] += total_time_q

        # ---- per-agent_type ----
        atype = getattr(req, "agent_belong", -1)
        if atype is None:
            atype = -1
        self._atype_count[atype] += 1
        if proc_time is not None:
            self._atype_proc_time_sum[atype] += proc_time
        if total_time_q is not None:
            self._atype_total_time_sum[atype] += total_time_q
        if aid >= 0:
            self._atype_agent_ids[atype].add(aid)

        self.records.append({
            "rid": req.rid,
            "generated_time": gen_t,
            "startal_time": req.start_processing_time,
            "completion_time": req.completion_time,
            "processing_time": proc_time,            # 不含排队
            "total_time_with_queue": total_time_q,   # 含排队+evict
            "avg_time_per_round": avg_time_per_round,
            "rounds": req.rounds,
            "initial_length": req.original_len,
            "final_length": req.length,
            "actual_type": req.actual_type,
            "predicted_type": req.predicted_type,
            "agent_id": aid,
            "agent_type": atype,
        })

    def record_batch(self, batch: Batch):
        rounds = len(batch.round_cost)
        if rounds == 0:
            batch.print_info()
            return

        tot_cost = sum(batch.round_cost)
        avg_cost = tot_cost / rounds
        attention_avg_cost = sum(batch.Acost) / len(batch.Acost) if batch.Acost else 0
        attention_weight = attention_avg_cost / batch.FFN_unit_cost if batch.FFN_unit_cost else 0
        self.batch_avg_attention.append(attention_weight)
        self.batch_info.append({
            "batch_id": batch.batch_id,
            "served_type": getattr(batch, "served_type", None),
            "rounds_run": rounds,
            "served_requests": batch.ever_served_request,
            "Attention_avg_cost": attention_avg_cost,
            "Avg_Round_cost": avg_cost,
        })

    # ---- summary helpers ----
    def _build_per_actual_type(self):
        out = {}
        for at in sorted(self._at_count.keys()):
            cnt = self._at_count[at]
            out[str(at)] = {
                "count": cnt,
                "avg_processing_time":     (self._at_proc_time_sum[at] / cnt) if cnt > 0 else None,
                "avg_total_time_with_queue": (self._at_total_time_sum[at] / cnt) if cnt > 0 else None,
            }
        return out

    def _build_per_agent_type(self):
        out = {}
        for atype in sorted(self._atype_count.keys()):
            cnt = self._atype_count[atype]
            n_agents = len(self._atype_agent_ids[atype])
            out[str(atype)] = {
                "count": cnt,
                "avg_processing_time":       (self._atype_proc_time_sum[atype] / cnt) if cnt > 0 else None,
                "avg_total_time_with_queue": (self._atype_total_time_sum[atype] / cnt) if cnt > 0 else None,
                "distinct_agents":   n_agents,
                "avg_count_per_agent": (cnt / n_agents) if n_agents > 0 else None,
            }
        return out

    def _build_per_agent_id(self):
        # 写入独立文件, 不进 summary
        out = {}
        for aid in sorted(self._aid_count.keys()):
            cnt = self._aid_count[aid]
            out[str(aid)] = {
                "count": cnt,
                "avg_processing_time":       (self._aid_proc_time_sum[aid] / cnt) if cnt > 0 else None,
                "avg_total_time_with_queue": (self._aid_total_time_sum[aid] / cnt) if cnt > 0 else None,
            }
        return out

    def summary(self):
        if not self.records:
            return {}

        proc_time_sum = 0
        total_q_time_sum = 0
        per_round_sum = 0
        per_round_count = 0
        buckets_proc = {b: [] for b in LENGTH_BUCKETS}
        buckets_total_q = {b: [] for b in LENGTH_BUCKETS}

        for r in self.records:
            proc = r["processing_time"]
            total_q = r["total_time_with_queue"]
            rounds = r["rounds"]
            init_len = r["initial_length"]
            bk = _length_bucket(init_len)
            if proc is not None:
                proc_time_sum += proc
                buckets_proc[bk].append(proc)
                if rounds > 0:
                    per_round_sum += proc / rounds
                    per_round_count += 1
            if total_q is not None:
                total_q_time_sum += total_q
                buckets_total_q[bk].append(total_q)

        n = self.finished_request
        avg_proc = proc_time_sum / n if n > 0 else None
        avg_total_q = total_q_time_sum / n if n > 0 else None
        avg_per_round = per_round_sum / per_round_count if per_round_count > 0 else None

        bucket_avg_proc = {k: (sum(v)/len(v) if v else None) for k, v in buckets_proc.items()}
        bucket_avg_total_q = {k: (sum(v)/len(v) if v else None) for k, v in buckets_total_q.items()}

        if self.batch_info:
            total_batch = len(self.batch_info)
            batch_round_cost = sum(b["Avg_Round_cost"] for b in self.batch_info) / total_batch
        else:
            total_batch = 0
            batch_round_cost = None
        batch_attention_cost = (sum(self.batch_avg_attention) / len(self.batch_avg_attention)
                                if self.batch_avg_attention else None)

        return {
            "finished_requests": self.finished_request,
            "avg_processing_time": avg_proc,                  # 不含排队
            "avg_total_time_with_queue": avg_total_q,         # 含排队+evict
            "avg_time_per_cycle_per_request": avg_per_round,
            "avg_processing_time_by_initial_length": bucket_avg_proc,
            "avg_total_time_with_queue_by_initial_length": bucket_avg_total_q,
            "finished count": self.length_distribution,
            "batch_attention_avg": batch_attention_cost,
            "num_batches": total_batch,
            "avg_batch_cost": batch_round_cost,

            "total_evict_count": self.total_evict_count,
            "per_actual_type":  self._build_per_actual_type(),
            "per_agent_type":   self._build_per_agent_type(),
        }

    # ---- dumps ----
    def dump_batch_info_to_json(self):
        path = os.path.join(self.output_dir, f"{self.prefix}_batch_info.json")
        with open(path, "w") as f:
            json.dump(self.batch_info, f, indent=2)

    def dump_records_to_json(self):
        path = os.path.join(self.output_dir, f"{self.prefix}_records.json")
        with open(path, "w") as f:
            json.dump(self.records, f, indent=2)

    def dump_per_agent_to_json(self):
        path = os.path.join(self.output_dir, f"{self.prefix}_per_agent.json")
        with open(path, "w") as f:
            json.dump(self._build_per_agent_id(), f, indent=2)

    def dump_summary_to_json(self):
        summary_data = self.summary()
        filename = os.path.join(self.output_dir, f"{self.prefix}_summary.json")
        with open(filename, "w") as f:
            json.dump(summary_data, f, indent=2)