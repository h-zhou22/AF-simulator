import math
import argparse
from generator import UniformGenerator, UniformRandomGenerator
from attention import Server
from typing import Dict, List, Tuple
from stats import StatsCollector
from FFN import FFN
from batch import Batch
from collections import deque

def parse_args():
    parser = argparse.ArgumentParser(description="Simulation Experiment Controller")

    parser.add_argument("--generator", type=int, default=0,
                        help="0 uniform generator, 1 random-uniform generator, 2 geometry generator, 3 poisson generator")
    parser.add_argument("--num_server", type=int, default=1,
                        help="number of servers to create")
    
    parser.add_argument("--num_batch", type=int, default=2,
                        help="number of batches inside each server")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="number of requests inside each batch")
    parser.add_argument("--use_length_limit", action="store_true",
                        help="use max-length limitation for batches")
    parser.add_argument("--batch_max_length", type=int, default=65536,
                        help="maximal allowed tokens inneach batch")
    
    
    parser.add_argument("--gen_prob", type=float, default=0.995,
                        help="Probability to generate next token")
    parser.add_argument("--rate", type=int, default=1,
                        help="Frequency of cycles to generate a token (in Uniformgenerator)")
    parser.add_argument("--basic_num", type=int, default=0,
                        help="Number of requests generated at the first cycle")
    parser.add_argument("--gen_req_per_cyc", type=int, default=1,
                        help="requests generated in each cycle (UniformRandomGenerator)")
    parser.add_argument("--total_request", type=int, default=100,
                        help="total number of requests to generate before stopping experiment")
    
    parser.add_argument("--max_prompt_len", type=int, default=4096,
                        help="maximum prompt length for generated requests")
    parser.add_argument("--maximal_generation", type=int, default=100)
    
    parser.add_argument("--num_server", type=int, default=1,
                        help="number of servers to create")
    parser.add_argument("--num_FFN", type=int, default=1,
                        help="number of FFN workers to create")

    parser.add_argument("--alpha_A", type=float, default=0.1)
    parser.add_argument("--alpha_T", type=float, default=0.001)
    parser.add_argument("--alpha_F", type=float, default=0.1)
    parser.add_argument("--beta_A", type=float, default=512)
    parser.add_argument("--beta_T", type=float, default=16)
    parser.add_argument("--beta_F", type=float, default=512)

    parser.add_argument(
        "--out_prefix",
        type=str,
        default="",
        help="output file prefix for statistics json files"
    )
    parser.add_argument("--basic_length", type = int, default=96)
    
    
    return parser.parse_args()

def main():
    args = parse_args()
    stats = StatsCollector(args.out_prefix)

    num_servers = args.num_server
    servers = []

    batch_id = 0
    num_batch = args.num_batch
    batch_size = args.batch_size
    use_length_limit = args.use_length_limit
    stored_batches: Dict[int, Batch] = {}

    for idx in range(num_servers):
        batches: Dict[int, Batch] = {}
        for i in range(num_batch):
            new_batch = Batch(batch_id, batch_size, use_length_limit, args.batch_max_length)
            batches[batch_id] =  new_batch
            stored_batches[batch_id] = new_batch
            batch_id += 1
        server = Server(idx, args.num_batch, batches)
        servers.append(server)

    generator_seed = 4
    if args.generator == 0:
        generator = UniformGenerator(
            #arranger=arranger,
            next_token_prob= args.next_token_prob,
            seed= generator_seed,
            rate=args.rate,
            max_length=args.max_prompt_len,
            #next_token_prob=0.7,
            num_per_cyc= args.gen_req_per_cyc,
            maximal_generation = args.maximal_generation
        )  
    elif args.generator == 1:
        generator = UniformRandomGenerator(
            next_token_prob= args.next_token_prob,
            seed= generator_seed,
            rate=args.gen_prob,
            max_length=args.max_prompt_len,
            num_per_cyc= args.gen_req_per_cyc,
            maximal_generation = args.maximal_generation,
            basic_length=args.basic_num
        ) 

    FFN_workers: List[FFN] = []
    num_FFN = args.num_FFN
    for FFN_id in range(num_FFN):
        FFN_worker = FFN(FFN_id)
        FFN_workers.append(FFN_worker)

    global_time = 0
    finished_requests = 0
    test_print = False

    alpha_A = args.alpha_A
    alpha_F = args.alpha_F
    alpha_T = args.alpha_T
    beta_A = args.beta_A
    beta_F = args.beta_F
    beta_T = args.beta_T

    buffer = deque()
    req_inq = 0

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

        while (available_batches is not None) and buffer():
            request = buffer.pop()
            best_batch_info = min(available_batches)
            batch_id0 = best_batch_info[2]
            server_id0 = best_batch_info[3]
            best_batch = stored_batches[batch_id0]
            target_server = servers[server_id0]
            target_server.load_request_to_batch(global_time, best_batch_info[2], request)
            available_batches.remove(best_batch)
            if best_batch.has_free_slot(global_time):
                info0, info1 = best_batch.update_info(current_time=global_time)
                new_info = [info0, info1, batch_id0, server_id0]
                available_batches.append(new_info)

        for server in servers:
            server.attention_work(global_time, alpha_A, beta_A)

        FFN_server.cycle_work(global_time, alpha_F, beta_F)

        finished_requests = stats.finished_request
        global_time += 1
    
    print("Experiment finished.")
    print(f"Total cycles: {global_time}")
    print(f"Total finished: {finished_requests}")
    print("\n=== STATISTICS SUMMARY ===")
    stats.dump_records_to_json()
    stats.dump_summary_to_json()
    stats.dump_batch_info_to_json()
    print("All finished")