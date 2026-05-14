import math
import time
import argparse
from generator import UniformGenerator, UniformRandomGenerator, MultitypeRandomGenerator, Multitype_Agent_Generator
from attention import Server
from typing import Dict, List, Tuple
from stats import StatsCollector
from request import Request
from FFN import FFN, dynamic_FFN, MoEFFN
from batch import Batch
from scheduler import BasicScheduler, DynamicScheduler, PipelineScheduler, MoEScheduler
from collections import deque
from arranger import GlobalArranger, GreedyArranger, MultitypeArranger

# main() 内, 在 global_time = 0 之后, 进入 if args.FFN_type == 0/1/3/4 之前
cycle_times = []                # 每个 cycle 的 wall-clock (秒)
last_report_cycle = 0
last_report_wall = time.perf_counter()
total_wall_start = last_report_wall
REPORT_EVERY = 1000             # 每多少 cycle 打印一次, 按需调

def _record_cycle_and_maybe_report(global_time, finished_requests):
    """每个 cycle 末尾调一次. 返回值忽略."""
    nonlocal_dummy = None  # main 里直接闭包不行, 用全局变量替代见下

def parse_args():
    parser = argparse.ArgumentParser(description="Simulation Experiment Controller")

    parser.add_argument("--arranger", type=int, default=0,
                        help="0 global arranger, 1 Greedy arranger, 2 Multi-type arranger")
    parser.add_argument("--generator", type=int, default=1,
                        help="0 uniform generator, 1 random-uniform generator, 2 geometry generator, 3 poisson generator")
    parser.add_argument("--num_server", type=int, default=1,
                        help="number of servers to create")
    parser.add_argument("--FFN_type", type = int, default=0,
                        help="0: single FFN worker, 1: FFN server maintaining a given order, 2: MoE, 3: Baseline, 4: FFN with dynamic batching between Batches and FFN, 5 MoE FFN with queuing optimization")
    
    parser.add_argument("--num_batch", type=int, default=2,
                        help="number of batches inside each server")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="number of requests inside each batch")
    parser.add_argument("--use_length_limit", action="store_true",
                        help="use max-length limitation for batches")
    parser.add_argument("--batch_max_length", type=int, default=65536,
                        help="maximal allowed tokens inneach batch")
    
    parser.add_argument("--next_token_prob", type=float, default=0.995,
                        help="Probability for next token during pipeline.")

    parser.add_argument("--gen_prob", type=float, default=0.001,
                        help="Probability to generate next token (UR generator)")
    parser.add_argument("--rate", type=int, default=1,
                        help="Frequency of cycles to generate a token (in Uniformgenerator)")
    parser.add_argument("--basic_num", type=int, default=2000,
                        help="Number of requests generated at the first cycle")
    parser.add_argument("--gen_req_per_cyc", type=int, default=1,
                        help="requests generated in each cycle (UniformRandomGenerator)")
    parser.add_argument("--total_request", type=int, default=80,
                        help="total number of requests to generate before stopping experiment")
    
    parser.add_argument("--max_prompt_len", type=int, default=4096,
                        help="maximum prompt length for generated requests")
    parser.add_argument("--maximal_generation", type=int, default=10000)
    
    parser.add_argument("--num_FFN", type=int, default=1,
                        help="number of FFN workers to create")

    parser.add_argument("--alpha_A", type=float, default=0.1)
    parser.add_argument("--alpha_T", type=float, default=0.001)
    parser.add_argument("--alpha_F", type=float, default=0.1)
    parser.add_argument("--alpha_L", type=float, default=1)
    parser.add_argument("--beta_A", type=float, default=512.0)
    parser.add_argument("--beta_T", type=float, default=16.0)
    parser.add_argument("--beta_F", type=float, default=512.0)
    parser.add_argument("--beta_L", type=float, default=1.0)

    parser.add_argument(
        "--out_prefix",
        type=str,
        default="",
        help="output file prefix for statistics json files"
    )
    
    parser.add_argument("--allow_exchange", action="store_true",
                        help="Allow batches inside pipeline FFN to exchange when former ones are ready")
    
    parser.add_argument("--server_capacity", type=int, default=655360,
        help="Maximal memory capacity of Attention server")
    parser.add_argument("--prediction_agent", action="store_true",
                        help="Use different agents with ability to predict the generation length.")

    parser.add_argument("--is_MoE", action="store_true",
                        help="Use MoE.")

    return parser.parse_args()

def main():
    args = parse_args()
    stats = StatsCollector(args.out_prefix)

    alpha_A = args.alpha_A
    alpha_F = args.alpha_F
    alpha_T = args.alpha_T
    beta_A = args.beta_A
    beta_F = args.beta_F
    beta_T = args.beta_T
    batch_size = args.batch_size
    unit_FFN_time = alpha_F * batch_size + beta_F
    
    num_servers = args.num_server
    servers = []

    batch_id = 0
    num_batch = args.num_batch
    batch_size = args.batch_size
    use_length_limit = args.use_length_limit
    stored_batches: Dict[int, Batch] = {}

    memory_capacity = args.server_capacity
    use_prediction_agent = args.prediction_agent
    arranger_type = args.arranger
    generator_type = args.generator

    if arranger_type < 2:
        for idx in range(num_servers):
            batches: Dict[int, Batch] = {}
            for i in range(num_batch):
                new_batch = Batch(batch_id, batch_size, unit_FFN_time, use_length_limit, args.batch_max_length)
                batches[batch_id] =  new_batch
                new_batch.server_id = idx
                stored_batches[batch_id] = new_batch
                batch_id += 1
            server = Server(idx, args.num_batch, batch_size, unit_FFN_time, batches, memory_capacity=memory_capacity)
            if args.FFN_type == 4:
                server.dynamic_matching = True
            servers.append(server)
    else:
        """配置multi-type server和batch的长度分布情况"""
        if generator_type >= 2:
            short_round_server_cnt = math.ceil(num_servers/64)
            short_server_cnt = num_servers//8
            middle_server_cnt = num_servers//4
            long_server_cnt = num_servers//4
            long_mix_server_cnt = num_servers//32
            longest_server_cnt = num_servers//4
            normal_server_cnt = num_servers - short_round_server_cnt - middle_server_cnt - short_server_cnt - long_server_cnt - longest_server_cnt - long_mix_server_cnt
            assert normal_server_cnt >= 0
            server_id = 0
            for _ in range(short_round_server_cnt):
                    batches: Dict[int, Batch] = {}
                    new_batch_A = Batch(batch_id, batch_size, unit_FFN_time, use_length_limit, args.batch_max_length)
                    batches[batch_id] =  new_batch_A
                    new_batch_A.served_type = 4
                    new_batch_A.batch_size = 64
                    new_batch_A.server_id = server_id
                    stored_batches[batch_id] = new_batch_A
                    batch_id += 1

                    new_batch_B = Batch(batch_id, batch_size, unit_FFN_time, use_length_limit, args.batch_max_length)
                    batches[batch_id] =  new_batch_B
                    new_batch_B.served_type = 5
                    new_batch_B.batch_size = 64
                    new_batch_B.server_id = server_id
                    stored_batches[batch_id] = new_batch_B
                    batch_id += 1

                    server = Server(server_id, args.num_batch, batch_size, unit_FFN_time, batches, memory_capacity=memory_capacity)
                    if args.FFN_type == 4:
                        server.dynamic_matching = True
                    servers.append(server)
                    server_id += 1
                    server.typed_server = True

            for _ in range(short_server_cnt):
                batches: Dict[int, Batch] = {}
                for j in range(num_batch):
                    new_batch = Batch(batch_id, batch_size, unit_FFN_time, use_length_limit, args.batch_max_length)
                    batches[batch_id] =  new_batch
                    new_batch.served_type = 0
                    new_batch.batch_size = 256
                    new_batch.server_id = server_id
                    stored_batches[batch_id] = new_batch
                    batch_id += 1
                
                server = Server(server_id, args.num_batch, batch_size, unit_FFN_time, batches, memory_capacity=memory_capacity)
                if args.FFN_type == 4:
                        server.dynamic_matching = True
                servers.append(server)
                server_id += 1
                server.typed_server = True

            for _ in range(middle_server_cnt):
                batches: Dict[int, Batch] = {}
                for j in range(num_batch):
                    new_batch = Batch(batch_id, batch_size, unit_FFN_time, use_length_limit, args.batch_max_length)
                    batches[batch_id] =  new_batch
                    new_batch.served_type = 1
                    new_batch.batch_size = 128
                    new_batch.server_id = server_id
                    stored_batches[batch_id] = new_batch
                    batch_id += 1
                
                server = Server(server_id, args.num_batch, batch_size, unit_FFN_time, batches, memory_capacity=memory_capacity)
                if args.FFN_type == 4:
                        server.dynamic_matching = True
                servers.append(server)
                server_id += 1
                server.typed_server = True

            for _ in range(longest_server_cnt):
                    batches: Dict[int, Batch] = {}
                    new_batch_A = Batch(batch_id, batch_size, unit_FFN_time, use_length_limit, args.batch_max_length)
                    batches[batch_id] =  new_batch_A
                    new_batch_A.served_type = 0
                    new_batch_A.batch_size = 256
                    new_batch_A.server_id = server_id
                    stored_batches[batch_id] = new_batch_A
                    batch_id += 1

                    new_batch_B = Batch(batch_id, batch_size, unit_FFN_time, use_length_limit, args.batch_max_length)
                    batches[batch_id] =  new_batch_B
                    new_batch_B.served_type = 3
                    new_batch_B.batch_size = 20
                    new_batch_B.server_id = server_id
                    stored_batches[batch_id] = new_batch_B
                    batch_id += 1

                    server = Server(server_id, args.num_batch, batch_size, unit_FFN_time, batches, memory_capacity=memory_capacity)
                    if args.FFN_type == 4:
                            server.dynamic_matching = True
                    servers.append(server)
                    server_id += 1
                    server.typed_server = True

            for _ in range(long_server_cnt):
                    batches: Dict[int, Batch] = {}
                    new_batch_A = Batch(batch_id, batch_size, unit_FFN_time, use_length_limit, args.batch_max_length)
                    batches[batch_id] =  new_batch_A
                    new_batch_A.served_type = 2
                    new_batch_A.batch_size = 64
                    new_batch_A.server_id = server_id
                    stored_batches[batch_id] = new_batch_A
                    batch_id += 1

                    new_batch_B = Batch(batch_id, batch_size, unit_FFN_time, use_length_limit, args.batch_max_length)
                    batches[batch_id] =  new_batch_B
                    new_batch_B.served_type = 7
                    new_batch_B.batch_size = 64
                    new_batch_B.server_id = server_id
                    stored_batches[batch_id] = new_batch_B
                    batch_id += 1

                    server = Server(server_id, args.num_batch, batch_size, unit_FFN_time, batches, memory_capacity=memory_capacity)
                    if args.FFN_type == 4:
                            server.dynamic_matching = True
                    servers.append(server)
                    server_id += 1
                    server.typed_server = True

            for _ in range(long_mix_server_cnt):
                    batches: Dict[int, Batch] = {}
                    new_batch_A = Batch(batch_id, batch_size, unit_FFN_time, use_length_limit, args.batch_max_length)
                    batches[batch_id] =  new_batch_A
                    new_batch_A.served_type = 2
                    new_batch_A.batch_size = 64
                    new_batch_A.server_id = server_id
                    stored_batches[batch_id] = new_batch_A
                    batch_id += 1

                    new_batch_B = Batch(batch_id, batch_size, unit_FFN_time, use_length_limit, args.batch_max_length)
                    batches[batch_id] =  new_batch_B
                    new_batch_B.served_type = 6
                    new_batch_B.batch_size = 128
                    new_batch_B.server_id = server_id
                    stored_batches[batch_id] = new_batch_B
                    batch_id += 1

                    server = Server(server_id, args.num_batch, batch_size, unit_FFN_time, batches, memory_capacity=memory_capacity)
                    if args.FFN_type == 4:
                            server.dynamic_matching = True
                    servers.append(server)
                    server_id += 1
                    server.typed_server = True

            for _ in range(normal_server_cnt):
                batches: Dict[int, Batch] = {}
                for j in range(num_batch):
                    new_batch = Batch(batch_id, batch_size, unit_FFN_time, use_length_limit, args.batch_max_length)
                    batches[batch_id] =  new_batch
                    new_batch.served_type = -1
                    new_batch.batch_size = 64
                    new_batch.server_id = server_id
                    stored_batches[batch_id] = new_batch
                    batch_id += 1
                server = Server(server_id, args.num_batch, batch_size, unit_FFN_time, batches, memory_capacity=memory_capacity)
                if args.FFN_type == 4:
                        server.dynamic_matching = True
                servers.append(server)
                server_id += 1
                server.typed_server = True
        else:
            short_round_server_cnt = math.ceil(num_servers/64)
            short_server_cnt = num_servers//8
            middle_server_cnt = num_servers//4
            long_server_cnt = num_servers//4
            long_mix_server_cnt = num_servers//32
            normal_server_cnt = num_servers - short_round_server_cnt - middle_server_cnt - short_server_cnt - long_server_cnt - long_mix_server_cnt
            assert normal_server_cnt >= 0
            server_id = 0
            for _ in range(short_round_server_cnt):
                    batches: Dict[int, Batch] = {}
                    new_batch_A = Batch(batch_id, batch_size, unit_FFN_time, use_length_limit, args.batch_max_length)
                    batches[batch_id] =  new_batch_A
                    new_batch_A.served_type = 4
                    new_batch_A.batch_size = 64
                    new_batch_A.server_id = server_id
                    stored_batches[batch_id] = new_batch
                    batch_id += 1

                    new_batch_B = Batch(batch_id, batch_size, unit_FFN_time, use_length_limit, args.batch_max_length)
                    batches[batch_id] =  new_batch_B
                    new_batch_B.served_type = 5
                    new_batch_B.batch_size = 64
                    new_batch_B.server_id = server_id
                    stored_batches[batch_id] = new_batch
                    batch_id += 1

                    server = Server(server_id, args.num_batch, batch_size, unit_FFN_time, batches, memory_capacity=memory_capacity)
                    if args.FFN_type == 4:
                        server.dynamic_matching = True
                    servers.append(server)
                    server_id += 1
                    server.typed_server = True

            for _ in range(short_server_cnt):
                batches: Dict[int, Batch] = {}
                for j in range(num_batch):
                    new_batch = Batch(batch_id, batch_size, unit_FFN_time, use_length_limit, args.batch_max_length)
                    batches[batch_id] =  new_batch
                    new_batch.served_type = 0
                    new_batch.batch_size = 256
                    new_batch.server_id = server_id
                    stored_batches[batch_id] = new_batch
                    batch_id += 1
                
                server = Server(server_id, args.num_batch, batch_size, unit_FFN_time, batches, memory_capacity=memory_capacity)
                if args.FFN_type == 4:
                        server.dynamic_matching = True
                servers.append(server)
                server_id += 1
                server.typed_server = True

            for _ in range(middle_server_cnt):
                batches: Dict[int, Batch] = {}
                for j in range(num_batch):
                    new_batch = Batch(batch_id, batch_size, unit_FFN_time, use_length_limit, args.batch_max_length)
                    batches[batch_id] =  new_batch
                    new_batch.served_type = 1
                    new_batch.batch_size = 128
                    new_batch.server_id = server_id
                    stored_batches[batch_id] = new_batch
                    batch_id += 1
                
                server = Server(server_id, args.num_batch, batch_size, unit_FFN_time, batches, memory_capacity=memory_capacity)
                if args.FFN_type == 4:
                        server.dynamic_matching = True
                servers.append(server)
                server_id += 1
                server.typed_server = True

            for _ in range(long_server_cnt):
                    batches: Dict[int, Batch] = {}
                    new_batch_A = Batch(batch_id, batch_size, unit_FFN_time, use_length_limit, args.batch_max_length)
                    batches[batch_id] =  new_batch_A
                    new_batch_A.served_type = 2
                    new_batch_A.batch_size = 64
                    new_batch_A.server_id = server_id
                    stored_batches[batch_id] = new_batch
                    batch_id += 1

                    new_batch_B = Batch(batch_id, batch_size, unit_FFN_time, use_length_limit, args.batch_max_length)
                    batches[batch_id] =  new_batch_B
                    new_batch_B.served_type = 7
                    new_batch_B.batch_size = 64
                    new_batch_B.server_id = server_id
                    stored_batches[batch_id] = new_batch
                    batch_id += 1

                    server = Server(server_id, args.num_batch, batch_size, unit_FFN_time, batches, memory_capacity=memory_capacity)
                    if args.FFN_type == 4:
                            server.dynamic_matching = True
                    servers.append(server)
                    server_id += 1
                    server.typed_server = True

            for _ in range(long_mix_server_cnt):
                    batches: Dict[int, Batch] = {}
                    new_batch_A = Batch(batch_id, batch_size, unit_FFN_time, use_length_limit, args.batch_max_length)
                    batches[batch_id] =  new_batch_A
                    new_batch_A.served_type = 2
                    new_batch_A.batch_size = 64
                    new_batch_A.server_id = server_id
                    stored_batches[batch_id] = new_batch
                    batch_id += 1

                    new_batch_B = Batch(batch_id, batch_size, unit_FFN_time, use_length_limit, args.batch_max_length)
                    batches[batch_id] =  new_batch_B
                    new_batch_B.served_type = 6
                    new_batch_B.batch_size = 128
                    new_batch_B.server_id = server_id
                    stored_batches[batch_id] = new_batch
                    batch_id += 1

                    server = Server(server_id, args.num_batch, batch_size, unit_FFN_time, batches, memory_capacity=memory_capacity)
                    if args.FFN_type == 4:
                            server.dynamic_matching = True
                    servers.append(server)
                    server_id += 1
                    server.typed_server = True

            for _ in range(normal_server_cnt):
                batches: Dict[int, Batch] = {}
                for j in range(num_batch):
                    new_batch = Batch(batch_id, batch_size, unit_FFN_time, use_length_limit, args.batch_max_length)
                    batches[batch_id] =  new_batch
                    new_batch.served_type = -1
                    new_batch.batch_size = 64
                    new_batch.server_id = server_id
                    stored_batches[batch_id] = new_batch
                    batch_id += 1
                server = Server(server_id, args.num_batch, batch_size, unit_FFN_time, batches, memory_capacity=memory_capacity)
                if args.FFN_type == 4:
                        server.dynamic_matching = True
                servers.append(server)
                server_id += 1
                server.typed_server = True


    generator_seed = 4
    agent_seed = 42
    if args.generator == 0:
        generator = UniformGenerator(
            #arranger=arranger,
            next_token_prob= args.next_token_prob,
            seed= generator_seed,
            agent_seed = agent_seed,
            rate=args.rate,
            max_length=args.max_prompt_len,
            #next_token_prob=0.7,
            num_per_cyc= args.gen_req_per_cyc,
            maximal_generation = args.maximal_generation,
            alpha_L= args.alpha_L,
            beta_L= args.beta_L
        )  
    elif args.generator == 1:
        generator = UniformRandomGenerator(
            next_token_prob= args.next_token_prob,
            seed= generator_seed,
            agent_seed= agent_seed,
            rate=args.gen_prob,
            max_length=args.max_prompt_len,
            num_per_cyc= args.gen_req_per_cyc,
            maximal_generation = args.maximal_generation,
            basic_length=args.basic_num,
            alpha_L= args.alpha_L,
            beta_L= args.beta_L
        ) 
    elif args.generator == 2:
        generator = MultitypeRandomGenerator(
            next_token_prob= args.next_token_prob,
            seed= generator_seed,
            agent_seed= agent_seed,
            rate=args.gen_prob,
            max_length=args.max_prompt_len,
            num_per_cyc= args.gen_req_per_cyc,
            maximal_generation = args.maximal_generation,
            basic_length=args.basic_num,
            alpha_L= args.alpha_L,
            beta_L= args.beta_L
        )
    elif args.generator == 3:
         generator = Multitype_Agent_Generator(
            next_token_prob= args.next_token_prob,
            seed= generator_seed,
            agent_seed= agent_seed,
            rate=args.gen_prob,
            max_length=args.max_prompt_len,
            num_per_cyc= args.gen_req_per_cyc,
            maximal_generation = args.maximal_generation,
            basic_length=args.basic_num,
            alpha_L= args.alpha_L,
            beta_L= args.beta_L,
            is_MoE= args.is_MoE
        ) 
    else:
        raise NotImplementedError("Not Implemented Yet in generator.py")

    # 这里在修改FFN逻辑之后需要修改
    FFN_workers = []
    num_FFN = args.num_FFN
    for FFN_id in range(num_FFN):
        if args.FFN_type == 0 :
            FFN_worker = FFN(FFN_id)
        elif args.FFN_type == 1:
            FFN_worker =FFN(FFN_id)
        elif args.FFN_type == 3:
            FFN_worker = dynamic_FFN(FFN_id, should_serve_num_batches= 0, allow_exchange=args.allow_exchange)
        elif args.FFN_type == 4:
            FFN_worker = FFN(FFN_id)
        FFN_workers.append(FFN_worker)

    global_time = 0
    finished_requests = 0
    test_print = False

    costly_loading = False
    if args.arranger == 0:
        arranger = GlobalArranger(servers, stored_batches, costly_loading=costly_loading)
    elif args.arranger == 1:
        arranger = GreedyArranger(servers, costly_loading=costly_loading)
    elif args.arranger == 2:
        arranger = MultitypeArranger(servers, costly_loading=costly_loading)
    
    for batch in stored_batches.values():
        batch.arranger = arranger

    buffer = deque()
    req_inq = 0


    if args.FFN_type == 0:
    # Use single FFN worker for current experiment
        FFN_server = FFN_workers[0]
        # TODO: Main Loop
        while finished_requests < args.total_request:
            newly_generated_reqs = generator.step(global_time)
            for req in newly_generated_reqs:
                buffer.append(req)
                req_inq += 1
            for server in servers:
                server.cycle_work(global_time, stats, FFN_server, alpha_T, beta_T)

            available_batches : List[Tuple[int, int, int, int]] = []
            for server in servers:
                extend_batches = server.find_available_batch()
                available_batches.extend(extend_batches)
                # [num_req, batch_len, bid, sid]
            if test_print:
                print("Cycle: ", global_time)
                print("Finished req count: ",finished_requests)
                print("Available batches:{}, Buffer size: {}".format(len(available_batches), len(buffer)))
            while available_batches and buffer:
                
                if test_print:
                    for batch1 in available_batches:
                        print("Batch info ",batch1)
                    print("Buffer size: ",len(buffer))
                request = buffer.pop()
                best_batch_info = min(available_batches)
                batch_id0 = best_batch_info[2]
                server_id0 = best_batch_info[3]
                best_batch = stored_batches[batch_id0]
                target_server = servers[server_id0]
                target_server.load_request_to_batch(global_time, best_batch_info[2], request)
                available_batches.remove(best_batch_info)
                if best_batch.has_free_slot(global_time):
                    info0, info1 = best_batch.updated_info(current_time=global_time)
                    new_info = (info0, info1, batch_id0, server_id0)
                    available_batches.append(new_info)

            for server in servers:
                if test_print:
                    print("Server ID: ",server.server_id)
                    
                server.attention_work(global_time, alpha_A, beta_A)

            FFN_server.cycle_work(global_time, alpha_F, beta_F)

            finished_requests = stats.finished_request
            global_time += 1

            if test_print:
                print("Global Time: ", global_time)
                print("Finished requests: ", finished_requests)
                print("Total requests: ", len(buffer))
                print("Generated req: ", req_inq)
                print("Batch count: ",len(stored_batches))
                
                for j in range(num_batch):
                    print("BID: ",j)
                    batch_info = stored_batches[j].update_info(global_time)
                    print("Batch ino: ",batch_info)
                    print("Batch status: ",stored_batches[j].status)
                    print("Batch current ending: ", stored_batches[j].current_ending)
    # Loop End For Single FFN cases
    elif args.FFN_type == 2:
        # MoE
        if args.num_FFN < args.num_experts:
            raise ValueError(
                f"num_FFN ({args.num_FFN}) must >= num_experts ({args.num_experts}) for MoE")
        if not args.is_MoE:
            raise ValueError("FFN_type=5 (MoE) requires --is_MoE")
        from FFN import MoEFFN
        FFN_workers = []
        for wid in range(args.num_FFN):
            eid = wid % args.num_experts
            FFN_workers.append(MoEFFN(wid, eid, args.alpha_F_moe, args.beta_F_moe))

        scheduler = MoEScheduler(
            arranger, servers, FFN_workers, stats, buffer, stored_batches,
            alpha_A, beta_A, alpha_T, beta_T,
            args.alpha_F_moe, args.beta_F_moe,
            num_experts=args.num_experts,
            initially_full=True,
        )

        while finished_requests < args.total_request and (args.max_cycles <= 0 or global_time < args.max_cycles):
            newly_generated_reqs = generator.step(global_time)
            for req in newly_generated_reqs:
                arranger.inqueue_request(req)
            scheduler.do_cycle_work(global_time)
            finished_requests = stats.finished_request
            global_time += 1
    elif args.FFN_type == 1:  
        cyc_t0 = time.perf_counter()
        least_num_to_fill = args.num_batch * args.batch_size 
        if args.basic_num < least_num_to_fill:
            print("Basic num:{}, least to fill:{}".format(args.basic_num, args.num_batch * args.batch_size ))
            raise ValueError("Basic number of requests should be larger than the total number of requests in the batch")
        initial_reqs = generator.do_initial_generation()
        for req in initial_reqs:
            arranger.inqueue_request(req)

        scheduler = BasicScheduler(arranger, servers, FFN_workers, stats, buffer, stored_batches, alpha_A, beta_A, alpha_T, beta_T, alpha_F, beta_F)
        scheduler.match_AF()
        while finished_requests < args.total_request:
            time_print = False
            if global_time % 1000 == 0 and time_print:
                print("Global Time: ", global_time)
                print("Finished requests: ", finished_requests)
                # for batch in stored_batches.values():
                #     print("Batch {}, Status: {} ".format(batch.batch_id, batch.status))
            newly_generated_reqs = generator.step(global_time)
            for req in newly_generated_reqs:
                arranger.inqueue_request(req)

            scheduler.do_cycle_work(global_time)
            cycle_times.append(time.perf_counter() - cyc_t0)
            finished_requests = stats.finished_request
            global_time += 1

    elif args.FFN_type == 3:
        cyc_t0 = time.perf_counter()
        least_num_to_fill = args.num_batch * args.batch_size * args.num_server
        if args.basic_num < least_num_to_fill:
            raise ValueError("Basic number of requests should be larger than the total number of requests in the batch")
        initial_reqs = generator.do_initial_generation()
        for req in initial_reqs:
            arranger.inqueue_request(req)

        # 一开始要填满所有Batch， 至少需要生成这些request才能满足要求
        scheduler = PipelineScheduler(arranger, servers, FFN_workers, stats, buffer, stored_batches, alpha_A, beta_A, alpha_T, beta_T, alpha_F, beta_F, initially_full=True)
        # PipelineScheduler的初始匹配在构造函数中完成
        while finished_requests < args.total_request:
            newly_generated_reqs = generator.step(global_time)
            for req in newly_generated_reqs:
                arranger.inqueue_request(req)

            scheduler.do_cycle_work(global_time)
            cycle_times.append(time.perf_counter() - cyc_t0)
            finished_requests = stats.finished_request
            global_time += 1

            if global_time % 10000 == 0:
                print("Global Time: ", global_time)
                print("Finished requests: ", finished_requests)
                for batch in stored_batches.values():
                    batch.print_debug_information()

    elif args.FFN_type == 4:
        cyc_t0 = time.perf_counter()
        least_num_to_fill = args.num_batch * args.batch_size * args.num_server
        if args.basic_num < least_num_to_fill:
            raise ValueError("Basic number of requests should be larger than the total number of requests in the batch")
        initial_reqs = generator.do_initial_generation()
        for req in initial_reqs:
            arranger.inqueue_request(req)

        scheduler = DynamicScheduler(arranger, servers, FFN_workers, stats, buffer, stored_batches, alpha_A, beta_A, alpha_T, beta_T, alpha_F, beta_F, initially_full=True)
        
        while finished_requests < args.total_request:
            newly_generated_reqs = generator.step(global_time)
            for req in newly_generated_reqs:
                arranger.inqueue_request(req)
                
            scheduler.do_cycle_work(global_time)
            cycle_times.append(time.perf_counter() - cyc_t0)
            finished_requests = stats.finished_request
            global_time += 1 

            if global_time % 10000 == 0:
                print("Global Time: ", global_time)
                print("Generated requests: ", generator.gen_tot)
                print("Finished requests: ", finished_requests)
                for batch in stored_batches.values():
                    batch.print_debug_information()

    if cycle_times:
        total_wall = time.perf_counter() - total_wall_start
        n = len(cycle_times)
        avg = sum(cycle_times) / n
        sorted_t = sorted(cycle_times)
        p50 = sorted_t[n // 2]
        p95 = sorted_t[int(n * 0.95)]
        p99 = sorted_t[int(n * 0.99)]
        print(f"\n=== Cycle timing summary ===")
        print(f"Total cycles:     {n}")
        print(f"Total wall:       {total_wall:.2f}s")
        print(f"Avg per cycle:    {1000*avg:.3f}ms")
        print(f"p50 / p95 / p99:  {1000*p50:.3f} / {1000*p95:.3f} / {1000*p99:.3f}ms")
        print(f"Max:              {1000*max(cycle_times):.3f}ms")

        # 按 1000 cycle 一段平均, 看是否随实验进行变慢
        print(f"\n=== Per-10000-cycle averages (look for trends) ===")
        for i in range(0, n, 10000):
            chunk = cycle_times[i:i + 10000]
            print(f"  cycles {i}-{i+len(chunk)-1}: avg={10000*sum(chunk)/len(chunk):.3f}ms")
    main_print = False
    for batch_id in range(len(stored_batches)):
        if main_print:
            problem_FFN_id = stored_batches[batch_id].mapped_FFN_id
            problem_FFN_worker=FFN_workers[problem_FFN_id]
            problem_FFN_worker.debug_print_pipeline()
            # 打印疑似出现问题的FFN的流水线信息
        stats.record_batch(stored_batches[batch_id])

    print("Experiment finished.")
    print(f"Total cycles: {global_time}")
    print(f"Total finished: {finished_requests}")
    print("\n=== STATISTICS SUMMARY ===")
    stats.record_finish_cycle(global_time)
    stats.dump_records_to_json()
    stats.dump_summary_to_json()
    stats.dump_batch_info_to_json()
    stats.dump_per_agent_to_json()
    print("All finished")

if __name__ == "__main__":
    main()