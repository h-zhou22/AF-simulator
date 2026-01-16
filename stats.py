import json
import os
from request import Request
from batch import Batch
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

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

        # 调试输出：检查记录的数据
        if batch.batch_size_history:
            batch_sizes = [s for t, s in batch.batch_size_history]
            min_bs = min(batch_sizes)
            max_bs = max(batch_sizes)
            print(f"[DEBUG record_batch] Batch {batch.bids}: "
                  f"num_req_history_count={len(batch.batch_size_history)}, "
                  f"num_req_range=[{min_bs}, {max_bs}], "
                  f"ever_served={batch.ever_served_request}, "
                  f"batch_size_limit={batch.batch_size}, "
                  f"current_num_req={batch.num_req}")
            if max_bs > batch.batch_size:
                print(f"[ERROR] Batch {batch.bids} has num_req > batch_size limit! "
                      f"max_num_req={max_bs}, batch_size_limit={batch.batch_size}")

        self.batch_info.append(
            {
                "batch_id": batch.bids,
                "served_requests": batch.ever_served_request,
                "Acost": batch.Acost,
                "Fcost": batch.Fcost,
                "attention_waiting_time_history": batch.attention_waiting_time_history,
                "ffn_waiting_time_history": batch.ffn_waiting_time_history,
                "Round_cost": batch.round_cost,
                "Avg_Round_cost": avg_cost,
                "timeline": batch.timeline,  # 保存时间轴数据
                "batch_size_history": batch.batch_size_history,  # 保存batch_size历史
                "token_count_history": batch.token_count_history,  # 保存token_count历史
                "status_history": batch.status_history  # 保存status历史
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

    def plot_timeline(self, batch_id=None, max_rounds=None):
        """
        绘制batch的时间轴甘特图
        Args:
            batch_id: 要绘制的batch ID，如果为None则绘制所有batch（合并到一个图）
            max_rounds: 最多显示的轮数，如果为None则显示所有
        """
        if not self.batch_info:
            print("No batch info to plot")
            return
        
        # 确定要绘制的batches
        batches_to_plot = self.batch_info
        if batch_id is not None:
            batches_to_plot = [b for b in self.batch_info if b["batch_id"] == batch_id]
            if not batches_to_plot:
                print(f"Batch {batch_id} not found")
                return
        
        # 收集所有timeline数据
        batch_timelines = {}
        all_times_global = []
        
        for batch_info in batches_to_plot:
            batch_id_val = batch_info["batch_id"]
            timeline = batch_info.get("timeline", [])
            if not timeline:
                print(f"Batch {batch_id_val} has no timeline data")
                continue
            
            # 过滤轮数
            if max_rounds is not None:
                timeline = [t for t in timeline if t[3] < max_rounds]
            
            if timeline:
                batch_timelines[batch_id_val] = timeline
                for stage_type, start, end, round_id in timeline:
                    if end > 0:
                        all_times_global.extend([start, end])
        
        if not batch_timelines:
            print("No timeline data to plot")
            return
        
        # 定义颜色
        colors = {
            'A': '#FFD700',  # 黄色 - Attention
            'C1': '#90EE90',  # 浅绿色 - A2F传输
            'C2': '#FFB6C1',  # 浅粉色 - F2A传输
            'F': '#87CEEB',  # 天蓝色 - FFN
            'WA': '#D3D3D3',  # 灰色 - Attention等待
            'WF': '#C0C0C0'   # 银色 - FFN等待
        }
        
        # 计算全局时间范围
        if all_times_global:
            min_time_global = min(all_times_global)
            max_time_global = max(all_times_global)
            time_range_global = max_time_global - min_time_global
        else:
            return
        
        # 创建图形 - 多个batch上下排列
        num_batches = len(batch_timelines)
        fig, axes = plt.subplots(num_batches, 1, figsize=(16, 4 * num_batches))
        if num_batches == 1:
            axes = [axes]
        
        fig.suptitle('Batch Timeline (Gantt Chart)', fontsize=14, fontweight='bold')
        
        # 绘制每个batch
        for idx, (batch_id_val, timeline) in enumerate(sorted(batch_timelines.items())):
            ax = axes[idx]
            
            # 找到该batch的时间范围
            all_times = []
            for stage_type, start, end, round_id in timeline:
                if end > 0:
                    all_times.extend([start, end])
            
            if not all_times:
                continue
            
            # 设置坐标轴
            ax.set_ylabel(f'Batch {batch_id_val}', fontsize=11, fontweight='bold')
            ax.set_xlim(min_time_global - time_range_global * 0.05, max_time_global + time_range_global * 0.05)
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([])
            ax.grid(True, alpha=0.3, axis='x')
            if idx == num_batches - 1:  # 最后一个子图显示x轴标签
                ax.set_xlabel('Time (cycles)', fontsize=10)
            
            # 绘制所有阶段在同一行
            y_pos = 0
            bar_height = 0.4
            
            for stage_type, start, end, round_id in timeline:
                if end < 0:  # 未结束的等待
                    continue
                
                color = colors.get(stage_type, '#808080')
                label = ''
                
                # 生成标签
                if stage_type == 'A':
                    label = f'A{round_id+1}'
                elif stage_type == 'C1':
                    label = f'C{round_id+1}1'  # A2F传输
                elif stage_type == 'C2':
                    label = f'C{round_id+1}2'  # F2A传输
                elif stage_type == 'F':
                    label = f'F{round_id+1}'
                elif stage_type == 'WA':
                    label = 'WA'
                elif stage_type == 'WF':
                    label = 'WF'
                
                width = end - start
                rect = mpatches.Rectangle((start, y_pos - bar_height/2), width, bar_height,
                                         facecolor=color, edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
                
                # 添加标签
                if width > time_range_global * 0.02:  # 只在足够宽的条上显示标签
                    ax.text(start + width/2, y_pos, label,
                           ha='center', va='center', fontsize=8, fontweight='bold')
        
        # 添加图例（只添加一次）
        legend_elements = [
            mpatches.Patch(facecolor=colors['A'], label='Attention'),
            mpatches.Patch(facecolor=colors['C1'], label='A2F Transmission'),
            mpatches.Patch(facecolor=colors['C2'], label='F2A Transmission'),
            mpatches.Patch(facecolor=colors['F'], label='FFN'),
            mpatches.Patch(facecolor=colors['WA'], label='Attention Waiting'),
            mpatches.Patch(facecolor=colors['WF'], label='FFN Waiting')
        ]
        fig.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if len(batch_timelines) == 1:
            batch_ids_str = str(list(batch_timelines.keys())[0])
        else:
            batch_ids_str = '_'.join([str(bid) for bid in sorted(batch_timelines.keys())])
        
        filename = os.path.join(
            self.output_dir,
            f"{self.prefix}_timeline_batch_{batch_ids_str}_{timestamp}.png"
        )
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Timeline plot saved to: {filename}")
        plt.close()
    
    def plot_all_timelines(self, max_rounds=None):
        """绘制所有batch的时间轴（合并到一个图中）"""
        self.plot_timeline(batch_id=None, max_rounds=max_rounds)
    
    def plot_merged_timeline(self, batch_ids=None, max_rounds=None):
        """
        绘制多个batch合并的时间轴甘特图
        Args:
            batch_ids: 要合并的batch ID列表，如果为None则合并所有batch
            max_rounds: 最多显示的轮数，如果为None则显示所有
        """
        if not self.batch_info:
            print("No batch info to plot")
            return
        
        # 确定要绘制的batches
        if batch_ids is None:
            batches_to_plot = self.batch_info
        else:
            batches_to_plot = [b for b in self.batch_info if b["batch_id"] in batch_ids]
        
        if len(batches_to_plot) < 2:
            print("Need at least 2 batches to merge")
            return
        
        # 收集所有timeline数据、batch_size历史和token_count历史
        batch_timelines = {}
        batch_size_histories = {}
        token_count_histories = {}
        all_times = []
        
        for batch_info in batches_to_plot:
            batch_id = batch_info["batch_id"]
            timeline = batch_info.get("timeline", [])
            batch_size_history = batch_info.get("batch_size_history", [])
            token_count_history = batch_info.get("token_count_history", [])
            
            if not timeline:
                continue
            
            # 过滤轮数
            if max_rounds is not None:
                timeline = [t for t in timeline if t[3] < max_rounds]
                # 过滤历史数据（只保留在时间范围内的）
                max_time_in_timeline = max([t[2] for t in timeline if t[2] > 0], default=0)
                batch_size_history = [(t, s) for t, s in batch_size_history if t <= max_time_in_timeline]
                token_count_history = [(t, s) for t, s in token_count_history if t <= max_time_in_timeline]
            
            if timeline:
                batch_timelines[batch_id] = timeline
                batch_size_histories[batch_id] = batch_size_history
                token_count_histories[batch_id] = token_count_history
                for stage_type, start, end, round_id in timeline:
                    if end > 0:
                        all_times.extend([start, end])
                # 添加历史数据的时间点
                for t, s in batch_size_history:
                    all_times.append(t)
                for t, s in token_count_history:
                    all_times.append(t)
        
        if not all_times:
            print("No timeline data to plot")
            return
        
        min_time = min(all_times)
        max_time = max(all_times)
        time_range = max_time - min_time
        
        # 定义batch颜色方案
        batch_colors = {
            0: {
                'A': '#FFD700',      # 黄色 - Batch0 Attention
                'C1': '#90EE90',     # 浅绿色 - Batch0 A2F传输
                'C2': '#FFB6C1',     # 浅粉色 - Batch0 F2A传输
                'F': '#87CEEB',       # 天蓝色 - Batch0 FFN
                'WA': '#D3D3D3',      # 灰色 - Batch0 Attention等待
                'WF': '#C0C0C0'       # 银色 - Batch0 FFN等待
            },
            1: {
                'A': '#FF8C00',       # 深橙色 - Batch1 Attention
                'C1': '#32CD32',      # 绿色 - Batch1 A2F传输
                'C2': '#FF69B4',      # 粉红色 - Batch1 F2A传输
                'F': '#4169E1',       # 蓝色 - Batch1 FFN
                'WA': '#A9A9A9',      # 深灰色 - Batch1 Attention等待
                'WF': '#808080'       # 灰色 - Batch1 FFN等待
            }
        }
        
        # 为其他batch分配颜色（如果需要）
        color_palettes = [
            {'A': '#FFD700', 'C1': '#90EE90', 'C2': '#FFB6C1', 'F': '#87CEEB', 'WA': '#D3D3D3', 'WF': '#C0C0C0'},
            {'A': '#FF8C00', 'C1': '#32CD32', 'C2': '#FF69B4', 'F': '#4169E1', 'WA': '#A9A9A9', 'WF': '#808080'},
            {'A': '#9ACD32', 'C1': '#00CED1', 'C2': '#DA70D6', 'F': '#1E90FF', 'WA': '#C0C0C0', 'WF': '#A0A0A0'},
            {'A': '#FF6347', 'C1': '#00FA9A', 'C2': '#FF1493', 'F': '#0000FF', 'WA': '#808080', 'WF': '#606060'}
        ]
        
        for idx, batch_id in enumerate(sorted(batch_timelines.keys())):
            if batch_id not in batch_colors:
                batch_colors[batch_id] = color_palettes[idx % len(color_palettes)]
        
        # 创建图形 - 4个子图：2个时间轴 + 1个batch_size曲线 + 1个attention/FFN时间曲线
        fig, axes = plt.subplots(4, 1, figsize=(20, 12))
        fig.suptitle('Merged Timeline (Gantt Chart) - Multiple Batches', fontsize=16, fontweight='bold')
        
        row_labels = ['Attention (A)', 'FFN (F)', 'Batch Size', 'Processing Time (Batch 1)']
        
        # 计算每个batch的y位置（在同一行中上下排列）
        num_batches = len(batch_timelines)
        batch_y_positions = {}
        if num_batches == 2:
            batch_y_positions = {list(batch_timelines.keys())[0]: 0.2, 
                                list(batch_timelines.keys())[1]: -0.2}
        else:
            y_spacing = 0.6 / num_batches
            y_start = 0.3
            for idx, batch_id in enumerate(sorted(batch_timelines.keys())):
                batch_y_positions[batch_id] = y_start - idx * y_spacing
        
        bar_height = 0.3 if num_batches == 2 else 0.25
        
        # 绘制每一行（前2行是时间轴：Attention 和 FFN）
        for row_idx, row_name in enumerate(row_labels[:2]):
            ax = axes[row_idx]
            ax.set_ylabel(row_name, fontsize=11, fontweight='bold')
            ax.set_xlim(min_time - time_range * 0.05, max_time + time_range * 0.05)
            ax.set_ylim(-0.6, 0.6)
            ax.set_yticks([])
            ax.grid(True, alpha=0.3, axis='x')
            # 前2行不显示x轴标签
            ax.set_xticklabels([])
            
            # 绘制每个batch的任务
            for batch_id, timeline in batch_timelines.items():
                y_pos = batch_y_positions[batch_id]
                colors = batch_colors[batch_id]
                
                for stage_type, start, end, round_id in timeline:
                    if end < 0:  # 未结束的等待
                        continue
                    
                    # 确定是否属于当前行
                    draw = False
                    color = colors.get(stage_type, '#808080')
                    label = ''
                    
                    if row_idx == 0:  # Attention行
                        if stage_type == 'A':
                            draw = True
                            label = f'B{batch_id}A{round_id+1}'
                        elif stage_type == 'WA':
                            draw = True
                            label = f'B{batch_id}WA'
                        elif stage_type == 'WF':
                            draw = True
                            label = f'B{batch_id}WF'
                    elif row_idx == 1:  # FFN行
                        if stage_type == 'F':
                            draw = True
                            label = f'B{batch_id}F{round_id+1}'
                    
                    if draw:
                        width = end - start
                        rect = mpatches.Rectangle((start, y_pos - bar_height/2), width, bar_height,
                                                 facecolor=color, edgecolor='black', linewidth=0.8, 
                                                 alpha=0.8)
                        ax.add_patch(rect)
                        
                        # 添加标签
                        if width > time_range * 0.015:  # 只在足够宽的条上显示标签
                            ax.text(start + width/2, y_pos, label,
                                   ha='center', va='center', fontsize=7, fontweight='bold')
        
        # 绘制batch_size和token_count曲线（第3个子图）
        ax_size = axes[2]
        ax_size.set_ylabel('Batch Size / Token Count', fontsize=11, fontweight='bold')
        ax_size.set_xlim(min_time - time_range * 0.05, max_time + time_range * 0.05)
        ax_size.grid(True, alpha=0.3, axis='both')
        ax_size.set_xticklabels([])  # 不显示x轴标签，最后一个子图显示
        
        # 定义batch_size曲线的颜色（实线）
        batch_size_colors = {
            0: '#FF0000',  # 红色 - Batch0 batch_size
            1: '#0000FF',  # 蓝色 - Batch1 batch_size
            2: '#00FF00',  # 绿色 - Batch2 batch_size
            3: '#FF00FF'   # 紫色 - Batch3 batch_size
        }
        
        # 定义token_count曲线的颜色（虚线）
        token_count_colors = {
            0: '#FF6666',  # 浅红色 - Batch0 token_count
            1: '#6666FF',  # 浅蓝色 - Batch1 token_count
            2: '#66FF66',  # 浅绿色 - Batch2 token_count
            3: '#FF66FF'   # 浅紫色 - Batch3 token_count
        }
        
        # 绘制batch_size曲线
        for idx, batch_id in enumerate(sorted(batch_size_histories.keys())):
            if batch_id not in batch_size_colors:
                colors_list = ['#FF0000', '#0000FF', '#00FF00', '#FF00FF', '#FFA500', '#800080']
                batch_size_colors[batch_id] = colors_list[idx % len(colors_list)]
                token_count_colors[batch_id] = colors_list[idx % len(colors_list)]
            
            history = batch_size_histories[batch_id]
            if history:
                times = [t for t, s in history]
                sizes = [s for t, s in history]
                ax_size.plot(times, sizes, 
                            color=batch_size_colors[batch_id], 
                            linewidth=2, 
                            linestyle='-',
                            marker='o', 
                            markersize=3,
                            label=f'Batch {batch_id} Size',
                            alpha=0.8)
        
        # 绘制token_count曲线
        for idx, batch_id in enumerate(sorted(token_count_histories.keys())):
            if batch_id not in token_count_colors:
                colors_list = ['#FF6666', '#6666FF', '#66FF66', '#FF66FF', '#FFA500', '#800080']
                token_count_colors[batch_id] = colors_list[idx % len(colors_list)]
            
            history = token_count_histories[batch_id]
            if history:
                times = [t for t, s in history]
                counts = [s for t, s in history]
                ax_size.plot(times, counts, 
                            color=token_count_colors[batch_id], 
                            linewidth=2, 
                            linestyle='--',
                            marker='s', 
                            markersize=3,
                            label=f'Batch {batch_id} Tokens',
                            alpha=0.8)
        
        ax_size.legend(loc='upper right', fontsize=9, ncol=2)
        
        # 绘制 attention_time 和 FFN_time 随时间的变化（第4个子图，只显示 batch1）
        ax_time = axes[3]
        ax_time.set_ylabel('Processing Time (cycles)', fontsize=11, fontweight='bold')
        ax_time.set_xlim(min_time - time_range * 0.05, max_time + time_range * 0.05)
        ax_time.grid(True, alpha=0.3, axis='both')
        ax_time.set_xlabel('Time (cycles)', fontsize=11)
        
        # 只绘制 batch1 的数据
        batch1_id = 1
        # 从batch_info中获取batch1的信息
        batch1_info = None
        for batch_info in self.batch_info:
            if batch_info["batch_id"] == batch1_id:
                batch1_info = batch_info
                break
        
        if batch1_id in batch_timelines:
            timeline_b1 = batch_timelines[batch1_id]
            
            # 提取 Attention、FFN 和 Waiting 的时间数据
            attention_times = []  # [(start_time, duration), ...]
            ffn_times = []  # [(start_time, duration), ...]
            attention_waiting_times = []  # [(start_time, cumulative_wait_time), ...] - Attention侧等待
            ffn_waiting_times = []  # [(start_time, duration), ...] - FFN侧等待
            
            # 用于跟踪当前waiting阶段的开始时间
            current_wa_start = None
            
            for stage_type, start, end, round_id in timeline_b1:
                duration = end - start if end >= 0 else 0
                
                if stage_type == 'A':  # Attention
                    if end >= 0:
                        attention_times.append((start, duration))
                    current_wa_start = None  # Attention开始，清除waiting状态
                elif stage_type == 'F':  # FFN
                    if end >= 0:
                        ffn_times.append((start, duration))
                elif stage_type == 'WA':  # Attention侧等待
                    if end < 0:
                        # 未结束的等待（开始时间记录，结束时间未定）
                        current_wa_start = start
                    elif start == end:
                        # 这是一个定期采样的点，需要找到对应的waiting阶段开始时间
                        if current_wa_start is None:
                            # 向前查找waiting阶段的开始（查找end < 0或start != end的条目）
                            for i in range(len(timeline_b1) - 1, -1, -1):
                                prev_type, prev_start, prev_end, _ = timeline_b1[i]
                                if prev_type == 'WA':
                                    if prev_end < 0:
                                        # 找到未结束的waiting开始
                                        current_wa_start = prev_start
                                        break
                                    elif prev_start != prev_end and prev_end >= 0:
                                        # 找到已结束的waiting开始
                                        current_wa_start = prev_start
                                        break
                        if current_wa_start is not None:
                            cumulative_wait = start - current_wa_start
                            attention_waiting_times.append((start, cumulative_wait))
                    else:
                        # 这是waiting阶段的结束（start != end 且 end >= 0）
                        if duration > 0:
                            attention_waiting_times.append((end, duration))
                            current_wa_start = None
                elif stage_type == 'WF':  # FFN侧等待
                    if end >= 0:
                        ffn_waiting_times.append((start, duration))
            
            # 绘制 Attention time
            if attention_times:
                attn_times = [t for t, d in attention_times]
                attn_durations = [d for t, d in attention_times]
                ax_time.plot(attn_times, attn_durations,
                            color='#FF8C00',  # 深橙色
                            linewidth=2,
                            linestyle='-',
                            marker='o',
                            markersize=4,
                            label='Batch 1 Attention Time',
                            alpha=0.8)
            
            # 绘制 FFN time
            if ffn_times:
                ffn_times_list = [t for t, d in ffn_times]
                ffn_durations = [d for t, d in ffn_times]
                ax_time.plot(ffn_times_list, ffn_durations,
                            color='#4169E1',  # 蓝色
                            linewidth=2,
                            linestyle='--',
                            marker='s',
                            markersize=4,
                            label='Batch 1 FFN Time',
                            alpha=0.8)
            
            # 绘制 Attention侧 Waiting time（直接绘制所有记录的数据点）
            if batch1_info and "attention_waiting_time_history" in batch1_info:
                attn_wait_history = batch1_info["attention_waiting_time_history"]
                if attn_wait_history:
                    # 直接使用所有记录的数据点，有几个就画几个
                    attn_wait_times = [t for t, w in attn_wait_history]
                    attn_wait_values = [w for t, w in attn_wait_history]
                    
                    ax_time.plot(attn_wait_times, attn_wait_values,
                                color='#FFA500',  # 橙色
                                linewidth=2,
                                linestyle=':',
                                marker='^',
                                markersize=4,
                                label='Batch 1 Attention Waiting Time',
                                alpha=0.8)
            
            # 绘制 FFN侧 Waiting time（直接绘制所有记录的数据点）
            if batch1_info and "ffn_waiting_time_history" in batch1_info:
                ffn_wait_history = batch1_info["ffn_waiting_time_history"]
                if ffn_wait_history:
                    # 直接使用所有记录的数据点，有几个就画几个
                    ffn_wait_times = [t for t, w in ffn_wait_history]
                    ffn_wait_values = [w for t, w in ffn_wait_history]
                    
                    ax_time.plot(ffn_wait_times, ffn_wait_values,
                                color='#32CD32',  # 绿色
                                linewidth=2,
                                linestyle='-.',
                                marker='v',
                                markersize=4,
                                label='Batch 1 FFN Waiting Time',
                                alpha=0.8)
            
            ax_time.legend(loc='upper right', fontsize=9)
        else:
            ax_time.text(0.5, 0.5, 'Batch 1 data not available', 
                        ha='center', va='center', transform=ax_time.transAxes, fontsize=12)
        
        # 添加图例 - 按batch分组
        legend_elements = []
        for batch_id in sorted(batch_timelines.keys()):
            colors = batch_colors[batch_id]
            legend_elements.append(mpatches.Patch(facecolor=colors['A'], label=f'Batch {batch_id} - Attention', alpha=0.8))
            legend_elements.append(mpatches.Patch(facecolor=colors['C1'], label=f'Batch {batch_id} - A2F Transmission', alpha=0.8))
            legend_elements.append(mpatches.Patch(facecolor=colors['C2'], label=f'Batch {batch_id} - F2A Transmission', alpha=0.8))
            legend_elements.append(mpatches.Patch(facecolor=colors['F'], label=f'Batch {batch_id} - FFN', alpha=0.8))
            legend_elements.append(mpatches.Patch(facecolor=colors['WA'], label=f'Batch {batch_id} - Attention Waiting', alpha=0.8))
            legend_elements.append(mpatches.Patch(facecolor=colors['WF'], label=f'Batch {batch_id} - FFN Waiting', alpha=0.8))
        
        fig.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=2)
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_ids_str = '_'.join([str(bid) for bid in sorted(batch_timelines.keys())])
        filename = os.path.join(
            self.output_dir,
            f"{self.prefix}_timeline_merged_{batch_ids_str}_{timestamp}.png"
        )
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Merged timeline plot saved to: {filename}")
        plt.close()
    
    def plot_status_history(self, batch_id=1):
        """
        绘制batch的status随时间变化的离散图
        batch_id: 要绘制的batch ID
        """
        # 找到对应的batch信息
        batch_info = None
        for b in self.batch_info:
            if b["batch_id"] == batch_id:
                batch_info = b
                break
        
        if not batch_info or "status_history" not in batch_info:
            print(f"Batch {batch_id} status history not found")
            return
        
        status_history = batch_info["status_history"]
        if not status_history:
            print(f"Batch {batch_id} has no status history")
            return
        
        # 提取时间和status
        times = [t for t, s in status_history]
        statuses = [s for t, s in status_history]
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # 绘制离散点
        ax.scatter(times, statuses, s=10, alpha=0.6, c=statuses, cmap='viridis')
        
        # 设置坐标轴
        ax.set_xlabel('Time (cycles)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Status', fontsize=12, fontweight='bold')
        ax.set_ylim(0.5, 6.5)
        ax.set_yticks([1, 2, 3, 4, 5, 6])
        ax.set_yticklabels([
            '1: Attention\nprocessing',
            '2: FFN\nprocessing',
            '3: A2F\ntransfer',
            '4: F2A\ntransfer',
            '5: Waiting\n(Attention)',
            '6: Waiting\n(FFN)'
        ], fontsize=10)
        ax.grid(True, alpha=0.3, axis='both')
        ax.set_title(f'Batch {batch_id} Status Over Time', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(
            self.output_dir,
            f"{self.prefix}_status_history_batch_{batch_id}_{timestamp}.png"
        )
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Status history plot saved to: {filename}")
        plt.close()

