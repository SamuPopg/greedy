"""
The core Greedy Packer algorithm.
"""
import random
from typing import List, Set, Tuple

from tqdm import tqdm
import numpy as np

from src.core.models import (Cargo, LayerStrategy, PackingSolution, PlacementType,
                             Position, SortingStrategy, SupplierRegion)


class GreedyContainerOptimizer:
    """
    基于贪心策略的集装箱优化器。
    核心思想是构建一个高质量的初始解，并以此为最终结果，追求最快的计算速度。
    """
    def __init__(self, container_dims: Tuple[int, int, int]):
        self.container_dims = container_dims
        self.all_cargo: List[Cargo] = []
        self.regions: List[SupplierRegion] = []
        self.layer_strategy: LayerStrategy = None
        self.space: np.ndarray = None # Voxel space for faster checks

    def _create_supplier_regions(self, suppliers_sequence: List[str], all_cargo: List[Cargo]):
        # ... (This method remains unchanged)
        regions = []
        total_container_length = self.container_dims[0]
        supplier_volumes = {s: 0 for s in suppliers_sequence}
        for cargo in all_cargo:
            if cargo.supplier in supplier_volumes:
                supplier_volumes[cargo.supplier] += cargo.volume
        
        total_volume = sum(supplier_volumes.values())
        if total_volume == 0: return []

        current_pos = 0
        for supplier in suppliers_sequence:
            volume_ratio = supplier_volumes[supplier] / total_volume if total_volume > 0 else 0
            region_length = total_container_length * volume_ratio
            end_pos = current_pos + region_length
            regions.append(SupplierRegion(supplier, current_pos, end_pos))
            current_pos = end_pos
        
        if regions:
            regions[-1].end_position = total_container_length
        return regions

    def _get_candidate_points(self, solution: PackingSolution) -> List[Position]:
        # ... (This method remains unchanged)
        points = {Position(0, 0, 0)}
        
        for item in solution.placed_items:
            l, w, h = item.current_dims
            pos = item.position
            
            points.add(Position(pos.x + l, pos.y, pos.z))
            points.add(Position(pos.x, pos.y + w, pos.z))
            points.add(Position(pos.x, pos.y, pos.z + h))
            
            if h < 10 and pos.z + h < self.container_dims[2] - 10:
                points.add(Position(pos.x + l/2, pos.y + w/2, pos.z + h))
                points.add(Position(pos.x, pos.y, pos.z + h))
                points.add(Position(pos.x + l, pos.y, pos.z + h))
        
        return sorted(list(points), key=lambda p: (p.z, p.y, p.x))

    def _calculate_support_ratio(self, cargo: Cargo, position: Position, solution: PackingSolution) -> float:
        # ... (This method remains unchanged)
        if position.z == 0:
            return 1.0
        
        bottom_area = cargo.length * cargo.width
        if bottom_area == 0: return 0.0
        
        supported_area = 0.0
        for item in solution.placed_items:
            item_top_z = item.position.z + item.current_dims[2]
            if abs(item_top_z - position.z) > 0.1: continue
            
            x_overlap_start = max(position.x, item.position.x)
            x_overlap_end = min(position.x + cargo.length, item.position.x + item.current_dims[0])
            y_overlap_start = max(position.y, item.position.y)
            y_overlap_end = min(position.y + cargo.width, item.position.y + item.current_dims[1])
            
            if x_overlap_start < x_overlap_end and y_overlap_start < y_overlap_end:
                supported_area += (x_overlap_end - x_overlap_start) * (y_overlap_end - y_overlap_start)
        
        return supported_area / bottom_area

    def _check_center_stability(self, cargo: Cargo, position: Position, solution: PackingSolution) -> bool:
        # ... (This method remains unchanged)
        if position.z == 0: return True
        
        center_x = position.x + cargo.length / 2
        center_y = position.y + cargo.width / 2
        
        support_min_x, support_max_x = float('inf'), float('-inf')
        support_min_y, support_max_y = float('inf'), float('-inf')
        has_support = False
        
        for item in solution.placed_items:
            item_top_z = item.position.z + item.current_dims[2]
            if abs(item_top_z - position.z) > 0.1: continue
            
            x_overlap_start = max(position.x, item.position.x)
            x_overlap_end = min(position.x + cargo.length, item.position.x + item.current_dims[0])
            y_overlap_start = max(position.y, item.position.y)
            y_overlap_end = min(position.y + cargo.width, item.position.y + item.current_dims[1])
            
            if x_overlap_start < x_overlap_end and y_overlap_start < y_overlap_end:
                has_support = True
                support_min_x = min(support_min_x, x_overlap_start)
                support_max_x = max(support_max_x, x_overlap_end)
                support_min_y = min(support_min_y, y_overlap_start)
                support_max_y = max(support_max_y, y_overlap_end)
        
        if not has_support: return False
        
        margin_x = (support_max_x - support_min_x) * 0.15
        margin_y = (support_max_y - support_min_y) * 0.15
        
        return (support_min_x + margin_x <= center_x <= support_max_x - margin_x and
                support_min_y + margin_y <= center_y <= support_max_y - margin_y)

    def _check_domino_effect(self, cargo: Cargo, position: Position, orientation: PlacementType, placed_item_map: dict) -> bool:
        """
        检查是否存在多米诺骨牌效应的风险。
        '侧放1' (SIDE_LYING_X) 和 '躺放2' (LYING_Y) 禁止紧邻放置。
        """
        # 如果当前朝向不是风险朝向之一，则直接通过
        if orientation not in [PlacementType.SIDE_LYING_X, PlacementType.LYING_Y]:
            return True

        px, _, _ = position.x, position.y, position.z
        
        # 我们只关心新货物后表面(X坐标较小的一面)接触到的已放置货物
        # 因此，我们检查新货物起始X坐标-1的位置
        if px == 0:
            return True # 在集装箱最里面，不可能有东西在它后面

        neighbor_plane = self.space[int(px) - 1, :, :]
        neighbor_ids = np.unique(neighbor_plane[neighbor_plane > 0]) # 获取所有接触到的邻居ID

        for neighbor_id in neighbor_ids:
            neighbor_item = placed_item_map.get(int(neighbor_id))
            if not neighbor_item: continue

            # 检查邻居的朝向
            neighbor_orientation = neighbor_item.orientation
            
            # 定义不允许的组合
            forbidden_pairs = {
                (PlacementType.SIDE_LYING_X, PlacementType.LYING_Y),
                (PlacementType.LYING_Y, PlacementType.SIDE_LYING_X)
            }
            
            if (orientation, neighbor_orientation) in forbidden_pairs:
                # 在此可以增加更精细的接触面检查，但目前我们假设只要后方有风险物品就禁止
                # 检查两个物品的X坐标是否真的紧邻
                if abs((neighbor_item.position.x + neighbor_item.current_dims[0]) - px) < 0.1:
                    return False # 发现风险组合，禁止放置
        
        return True


    def _is_valid_placement(self, cargo: Cargo, position: Position, orientation: PlacementType, solution: PackingSolution, region: SupplierRegion) -> bool:
        # ... (This method remains unchanged)
        cargo.set_orientation(orientation)
        px, py, pz = position.x, position.y, position.z
        pl, pw, ph = cargo.length, cargo.width, cargo.height

        if not (py >= 0 and pz >= 0 and
                py + pw <= self.container_dims[1] and
                pz + ph <= self.container_dims[2]):
            return False
        
        if orientation not in self.layer_strategy.get_allowed_placements(pz):
            return False

        if not (px >= region.start_position and px + pl <= region.end_position):
            return False

        # Voxel-based collision check
        if np.any(self.space[int(px):int(px+pl), int(py):int(py+pw), int(pz):int(pz+ph)]):
            return False

        if pz > 0:
            if self._calculate_support_ratio(cargo, position, solution) < 0.7:
                return False
            if not self._check_center_stability(cargo, position, solution):
                return False
        
        if not self._check_domino_effect(cargo, position, orientation, solution.placed_item_map):
            return False
                
        return True

    def _create_initial_solution(self, strategy: SortingStrategy = SortingStrategy.VOLUME_DESC, strategy_index: int = 0) -> PackingSolution:
        # ... (This method remains unchanged)
        solution = PackingSolution(self.all_cargo)
        self.space = np.zeros(self.container_dims, dtype=np.int32)
        
        sorted_cargo = []
        if strategy == SortingStrategy.VOLUME_DESC:
            sorted_cargo = sorted(self.all_cargo, key=lambda c: c.volume, reverse=True)
        elif strategy == SortingStrategy.AREA_DESC:
            sorted_cargo = sorted(self.all_cargo, key=lambda c: c.original_dims[0] * c.original_dims[1], reverse=True)
        elif strategy == SortingStrategy.MAX_DIM_DESC:
            sorted_cargo = sorted(self.all_cargo, key=lambda c: max(c.original_dims), reverse=True)
        elif strategy == SortingStrategy.RANDOM_SHUFFLE:
            sorted_cargo = list(self.all_cargo)
            random.shuffle(sorted_cargo)
        
        for cargo in tqdm(sorted_cargo, desc=f"顺序评估", position=strategy_index, leave=True, ncols=100, dynamic_ncols=False):
            region = next((r for r in self.regions if r.name == cargo.supplier), None)
            if not region: continue

            best_placement = None
            
            point_set = set(self._get_candidate_points(solution))
            point_set.add(Position(region.start_position, 0, 0))
            candidate_points = sorted(list(point_set), key=lambda p: (p.z, p.y, p.x))
            
            for point in candidate_points:
                if not (region.start_position <= point.x < region.end_position):
                    continue

                allowed_placements = self.layer_strategy.get_allowed_placements(point.z)
                for orientation in allowed_placements:
                    cargo.set_orientation(orientation)
                    
                    if self._is_valid_placement(cargo, point, orientation, solution, region):
                        best_placement = (point, orientation)
                        break
                if best_placement:
                    break
            
            if best_placement:
                pos, orient = best_placement
                solution.add_item(cargo, pos, orient)
                px, py, pz = int(pos.x), int(pos.y), int(pos.z)
                pl, pw, ph = int(cargo.length), int(cargo.width), int(cargo.height)
                self.space[px:px+pl, py:py+pw, pz:pz+ph] = cargo.unique_id
        
        return solution

    def get_volume_ratio(self, solution: PackingSolution) -> float:
        # ... (This method remains unchanged)
        if not solution: return 0.0
        container_volume = self.container_dims[0] * self.container_dims[1] * self.container_dims[2]
        if container_volume == 0: return 0.0
        return solution.total_volume / container_volume

    def optimize_single_sequence(self, suppliers_sequence: List[str], all_cargo: List[Cargo], task_index: int = 0) -> PackingSolution:
        """
        Public method to run the optimization for a single supplier sequence.
        This is the main entry point for a single run of the greedy algorithm.
        """
        self.all_cargo = all_cargo
        self.regions = self._create_supplier_regions(suppliers_sequence, self.all_cargo)
        self.layer_strategy = LayerStrategy(self.container_dims, self.all_cargo)
        # self.layer_strategy.print_layers() # Optional: uncomment for debugging

        solution = self._create_initial_solution(strategy_index=task_index)
        
        if solution and solution.placed_items:
            solution.rate = self.get_volume_ratio(solution)
            solution.sequence = suppliers_sequence
        
        return solution
