# AF-Simulator

A discrete-cycle simulator for LLM serving systems. It models the interaction
between **Attention servers** and **FFN workers**, and provides several
scheduling strategies for matching attention computation to FFN computation.
A second, standalone simulator (`moe_simulator.py`) focuses purely on
**MoE expert-queue scheduling** efficiency.

---

## Part 1 — Usage

### 1.1 `main.py` — the AF (Attention–FFN) simulator

```bash
python3 main.py [options]
```

A typical invocation:

```bash
python3 main.py \
    --arranger=2 --generator=3 --FFN_type=3 \
    --num_server=20 --num_FFN=16 --num_batch=2 --batch_size=64 \
    --basic_num=10000 --total_request=2000 --maximal_generation=10000 \
    --gen_req_per_cyc=5 --gen_prob=1.0 \
    --alpha_A=0.001 --beta_A=0 \
    --alpha_F=1.5  --beta_F=10 \
    --alpha_T=0    --beta_T=1 \
    --alpha_L=0.01 --beta_L=0 \
    --out_prefix="my_experiment"
```

Results are written to `result/{out_prefix}_summary.json`,
`result/{out_prefix}_records.json`, `result/{out_prefix}_batch_info.json`,
and `result/{out_prefix}_per_agent.json`.

#### Topology / scheduling options

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--arranger` | int | 0 | Request-to-batch arranger. `0` Global, `1` Greedy, `2` Multitype. |
| `--generator` | int | 1 | Request generator. `0` Uniform, `1` Uniform-Random, `2` Multitype-Random, `3` Multitype-Agent (used with `--arranger=2`). |
| `--FFN_type` | int | 0 | FFN/scheduler strategy — see table below. |
| `--num_server` | int | 1 | Number of attention servers. |
| `--num_FFN` | int | 1 | Number of FFN workers. |
| `--num_batch` | int | 2 | Number of batches kept inside each server. |
| `--batch_size` | int | 16 | Number of requests per batch. |
| `--num_experts` | int | 8 | Number of MoE experts (only used when `--is_MoE`). |

#### `--FFN_type` values

| Value | Scheduler | Description |
|---|---|---|
| `0` | single FFN | One FFN worker, no scheduling. |
| `1` | `BasicScheduler` | Baseline; fixed server→FFN mapping. |
| `2` | `MoEScheduler` | Simple MoE expert routing. |
| `3` | `PipelineScheduler` | Dynamic pipeline matching by AF time ratio. Falls back to uniform assignment + AF-ratio swap when FFNs are insufficient. |
| `4` | `DynamicScheduler` | Dynamic batching between batches and FFN. |
| `5` | `MoEPriorityScheduler` | MoE routing with priority / congestion-avoidance queues. |
| `6` | `BalancedAttnFFNScheduler` | Per-server FCFS matching; balances FFN load by `w = t_F/(t_A+t_F)`, swapping servers off overloaded FFNs. |
| `7` | `BalancedBatchFFNScheduler` | Per-batch FCFS matching; balances FFN load at batch granularity. |
| `8` | `UniformBalancedScheduler` | Uniform server distribution + dynamic AF-ratio (`t_A/t_F`) swap. |

#### Generation / workload options

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--basic_num` | int | 2000 | Requests pre-generated at the first cycle (used to fill batches). Must be ≥ `num_batch × batch_size × num_server`. |
| `--total_request` | int | 80 | Total finished requests required before the run stops. |
| `--maximal_generation` | int | 10000 | Max generated tokens / generation budget. |
| `--gen_req_per_cyc` | int | 1 | New requests generated each cycle (Uniform-Random generator). |
| `--gen_prob` | float | 0.001 | Probability of generating the next token (Uniform-Random generator). |
| `--rate` | int | 1 | Generation rate parameter. |
| `--next_token_prob` | float | 0.995 | Probability of continuing a sequence during the pipeline. |
| `--max_prompt_len` | int | 4096 | Maximum prompt length for generated requests. |
| `--use_length_limit` | flag | off | Enforce a maximum token length per batch. |
| `--batch_max_length` | int | 65536 | Max tokens per batch when `--use_length_limit` is set. |
| `--max_cycles` | int | 2^31-1 | Hard cap on simulation cycles. |

#### Cost-model coefficients

Each phase is modelled as a linear cost `alpha * size + beta`:

| Option | Type | Default | Phase |
|---|---|---|---|
| `--alpha_A` / `--beta_A` | float | 0.1 / 512 | **Attention** time = `alpha_A * (token length) + beta_A`. |
| `--alpha_F` / `--beta_F` | float | 0.1 / 512 | **FFN** time = `alpha_F * (num requests) + beta_F`. |
| `--alpha_T` / `--beta_T` | float | 0.001 / 16 | **Transmission** (A<->F) time. |
| `--alpha_L` / `--beta_L` | float | 1 / 1 | **Loading** time. |
| `--alpha_F_moe` / `--beta_F_moe` | float | 0.5 / 2.5 | FFN cost specific to MoE experts. |

#### MoE-specific options (require `--is_MoE`)

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--is_MoE` | flag | off | Enable MoE mode. |
| `--moe_starve_1` | flag | off | Congestion-avoidance scheme 1 (per-expert service-count threshold). |
| `--moe_starve_2` | flag | off | Congestion-avoidance scheme 2 (time-bound demotion). |
| `--moe_starve_3` | flag | off | Congestion-avoidance scheme 3 (batch tail-completion promotion). |
| `--moe_c2` / `--moe_c3` / `--moe_c4` | int | 20 / 50 / 100 | Scheme-1 thresholds for priority levels 2/3/4. |
| `--moe_starve3_threshold` | int | 40 | Scheme-3 trigger: promote when `batch.moe_pending_count` is below this. |

#### Misc options

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--allow_exchange` | flag | off | Allow batches inside a pipeline FFN to exchange order when earlier ones are ready. |
| `--server_capacity` | int | 655360 | Max memory capacity per attention server. |
| `--prediction_agent` | flag | off | Use agents that predict generation length. |
| `--out_prefix` | str | "" | Prefix for the output JSON files. |

---

### 1.2 `moe_simulator.py` — the standalone MoE queue simulator

This simulator models only a **request generator** and **MoE expert workers**
(no attention/transmission layer). There are `num_experts` expert types, each
served by `workers_per_expert` workers (default 8 x 10 = 80 workers). Each
request samples 4 distinct experts and is complete once all 4 expert-tasks
finish; one task takes exactly one cycle.

```bash
python3 moe_simulator.py [options]
```

Examples:

```bash
python3 moe_simulator.py --strategy=sebf   --gen=poisson  --rate=22 --cycles=10000
python3 moe_simulator.py --strategy=level_c --c2=20 --c3=50 --c4=100
python3 moe_simulator.py --strategy=fcfs   --gen=gaussian --rate=20 --gauss_sigma=8
```

| Option | Type | Default | Meaning |
|---|---|---|---|
| `--strategy` | choice | `fcfs` | Scheduling strategy: `fcfs`, `level`, `level_c`, `level_t`, `sebf` (see below). |
| `--gen` | choice | `fixed` | Per-cycle arrival distribution: `fixed`, `uniform`, `poisson`, `geometric`, `gaussian` — all with mean `--rate`. |
| `--rate` | int | 20 | Expected number of requests generated per cycle. |
| `--gauss_sigma` | float | `rate/3` | Standard deviation for the gaussian generator only. |
| `--cycles` | int | 10000 | Number of simulation cycles (fixed-length run). |
| `--num_experts` | int | 8 | Number of expert types. |
| `--workers_per_expert` | int | 10 | Workers per expert type. |
| `--c2` / `--c3` / `--c4` | int | 20 / 50 / 100 | Scheme-1 (`level_c`) starvation thresholds for priority levels 2/3/4. |
| `--seed` | int | 42 | RNG seed. |
| `--out` | str | auto | Output file name; written under `result/`. Auto-named `stats_<strategy>_<gen>_<rate>.json` if omitted. |

#### Strategies

| Strategy | Description |
|---|---|
| `fcfs` | One FIFO queue per expert. |
| `level` | Five priority sub-queues per expert, ordered by remaining experts. |
| `level_c` | `level` + congestion scheme 1 (service-count thresholds `c2/c3/c4`). |
| `level_t` | `level` + congestion scheme 2 (cycle-based time bound). |
| `sebf` | Smallest Effective Bottleneck First; serves the request whose most-congested remaining expert queue is shortest, ties broken by arrival order. |

#### Reported statistics (printed and dumped to JSON)

- Completed request count and average latency (`completion - generated`).
- Per-expert "needed" counts, both at generation and at completion.
- Congestion-trigger counters (for `level_c` / `level_t`).
- Outlier counts at 2x, 5x, and 20x the average latency, split into
  *completed* and still-*pending* requests.

---

## Part 2 — File & code structure

| File | Role |
|---|---|
| `main.py` | Entry point for the AF simulator. Parses CLI args, builds servers / FFN workers / generator / arranger / scheduler, and runs the main per-cycle loop for each `FFN_type`. |
| `moe_simulator.py` | Self-contained MoE queue simulator (generator + experts + stats); independent of the rest of the codebase. |
| `attention.py` | The `Server` class — an attention node that holds a fixed set of batches, runs the attention phase, computes its AF time ratio, and drives each batch through its status transitions. |
| `FFN.py` | FFN worker implementations: `FFN` (FCFS), `dynamic_FFN` (pipeline with `BatchList`/`BatchNode`), and `MoEFFN` (expert-task processing). |
| `batch.py` | The `Batch` class — a group of requests moving together through Attention -> transmission -> FFN, with status bookkeeping and re-activation after a match switch. |
| `request.py` | The `Request` class — a single sequence with its length, type, and timing fields. |
| `generator.py` | Workload generators: `UniformGenerator`, `UniformRandomGenerator`, `MultitypeRandomGenerator`, `Multitype_Agent_Generator`, plus a `PredictionAgent` for length prediction. |
| `arranger.py` | Request-to-batch arrangers: `GlobalArranger`, `GreedyArranger`, `MultitypeArranger`. They decide which queued requests fill which batch slots. |
| `scheduler.py` | All AF schedulers (see below). Each owns the server<->FFN matching logic and a `do_cycle_work()` driver. |
| `stats.py` | `StatsCollector` — accumulates per-request and per-batch metrics and writes the result JSON files. |

### Scheduler classes (`scheduler.py`)

| Class | `FFN_type` | Notes |
|---|---|---|
| `BasicScheduler` | 1 | Fixed mapping baseline. |
| `PipelineScheduler` | 3 | AF-ratio level matching with a dynamic pipeline; includes the uniform-fallback path for insufficient FFNs. Helper classes: `AF_match`, `level_FFN`. |
| `DynamicScheduler` | 4 | Dynamic batching; helper queues `BatchTypedQueue`, `BatchQueue`. |
| `MoEScheduler` | 2 | Basic MoE routing. |
| `MoEPriorityScheduler` | 5 | MoE routing with `PriorityExpertQueue` and congestion-avoidance schemes. |
| `BalancedAttnFFNScheduler` | 6 | Per-server FCFS; balances FFN load via `w = t_F/(t_A+t_F)`. |
| `BalancedBatchFFNScheduler` | 7 | Per-batch FCFS; balances at batch granularity (batches of one server may live on different FFNs). |
| `UniformBalancedScheduler` | 8 | Uniform initial distribution + AF-ratio swap (`t_A/t_F`). |

### Per-cycle execution model

Every scheduler exposes a `do_cycle_work(current_time)` method that, each cycle:

1. Advances every server one step (attention progress, transmission, batch
   status transitions, eviction checks).
2. Lets the arranger refill freed batch slots from the request queue.
3. Optionally rebalances the server/batch <-> FFN matching (swap logic).
4. Runs the attention phase on each server and the FFN phase on each FFN worker.

`main.py` repeats this until `total_request` requests have finished or
`max_cycles` is reached, then asks `StatsCollector` to dump the JSON results.

### Batch status codes (`batch.py`)

| Status | Meaning |
|---|---|
| 0 | empty |
| 1 | attention phase |
| 2 | FFN phase |
| 3 | FFN -> Attention transmission |
| 4 | Attention -> FFN transmission |
| 5 | waiting for attention allocation |
| 6 | waiting for FFN allocation |