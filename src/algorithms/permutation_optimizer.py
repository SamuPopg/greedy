"""
Permutation Optimizer to find the best supplier sequence.
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import permutations
from typing import List, Tuple, Dict

from tqdm import tqdm

from src.algorithms.greedy_optimizer import GreedyContainerOptimizer
from src.core.models import Cargo, PackingSolution
from src.reporting.generator import OutputGenerator

# This is the worker function that will be executed in parallel.
# It must be a top-level function to be pickleable by multiprocessing.
def _run_single_strategy(task: Tuple) -> PackingSolution:
    """
    [工作单元] 运行单个供应商顺序，这是所有并行任务的最小执行单元。
    :param task: 一个元组 (suppliers_sequence, all_cargo, container_dims, task_index)
    :return: 一个 PackingSolution 对象
    """
    suppliers_sequence, all_cargo, container_dims, task_index = task
    
    optimizer = GreedyContainerOptimizer(container_dims)
    solution = optimizer.optimize_single_sequence(suppliers_sequence, all_cargo, task_index)
    
    return solution


class PermutationOptimizer:
    """
    排列优化器，寻找最优的供应商取货顺序。
    通过并行计算，为每一种供应商顺序组合运行一次标准的贪心算法，并从中择优。
    """
    def __init__(self, container_dims: Tuple[int, int, int], all_cargo: List[Cargo], all_suppliers: List[str]):
        self.container_dims = container_dims
        self.all_cargo = all_cargo
        self.all_suppliers = all_suppliers
        self.reporter = OutputGenerator("output") # TODO: make output dir configurable

    def run_optimization(self, top_n_results: int = 3):
        """主优化流程，并行评估所有供应商顺序"""
        start_time = time.time()
        
        all_sequences = list(permutations(self.all_suppliers))
        num_sequences = len(all_sequences)

        print(f"\n========================================================")
        print(f"  启动供应商顺序优化流程")
        print(f"  发现 {len(self.all_suppliers)} 个供应商, 将探索 {num_sequences} 种取货顺序。")
        print(f"  (将使用 '体积从大到小' 单一策略进行评估)")
        print(f"========================================================")

        if num_sequences == 0:
            print("错误：货物数据中未发现任何供应商信息，无法执行优化。")
            return
        
        final_results = self._evaluate_all_sequences(all_sequences)

        # --- 最终结果报告与输出 ---
        duration = time.time() - start_time
        self._generate_final_reports(final_results, top_n_results, duration)

    def _evaluate_all_sequences(self, sequences_to_run: List[Tuple[str, ...]]) -> List[PackingSolution]:
        """对给定的顺序列表，并行执行单一策略优化"""
        tasks = []
        for i, seq in enumerate(sequences_to_run):
            # Each task now includes all necessary data to be self-contained
            tasks.append((list(seq), self.all_cargo, self.container_dims, i))
        
        print(f"共创建 {len(tasks)} 个并行计算任务...")
        
        all_solutions = []
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_task = {
                executor.submit(_run_single_strategy, task): task for task in tasks
            }
            
            for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="评估所有顺序"):
                solution = future.result()
                if solution and solution.placed_items:
                    all_solutions.append(solution)
        
        return all_solutions

    def _generate_final_reports(self, final_results: List[PackingSolution], top_n: int, duration: float):
        """生成最终的Top-N报告"""
        if not final_results:
            print("\n错误：未能生成任何有效的装载方案。")
            return
            
        print(f"\n\n========================================================")
        print(f"  所有计算完成，最终择优方案如下 (Top {top_n})")
        print(f"  (所有报告文件将保存在 ./output/ 文件夹中)")
        print(f"========================================================")
        
        final_results.sort(key=lambda x: x.rate, reverse=True)
        
        solutions_to_report = final_results[:min(top_n, len(final_results))]

        for i, result in enumerate(solutions_to_report):
            rank = i + 1
            sequence_str = ' -> '.join(result.sequence)
            rate_str = f"{result.rate:.2%}"
            
            print(f"\n--- [方案 Top {rank}] ---")
            print(f"  - 推荐取货顺序: {sequence_str}")
            print(f"  - 最高装柜率: {rate_str}")
            
            self.reporter.generate_all_outputs(result, self.container_dims, rank)
            
        print("\n所有报告生成完毕。")
        print(f">>> 供应商顺序探索总耗时: {duration:.2f} 秒")
