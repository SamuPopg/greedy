import math
import time
import random
import copy
import pandas as pd
import numpy as np
import re
from typing import List, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class PlacementType(Enum):
    """定义货物的六种摆放方式"""
    UPRIGHT_X = '立放1'
    UPRIGHT_Y = '立放2'
    SIDE_LYING_X = '侧放1'
    SIDE_LYING_Y = '侧放2'
    LYING_X = '躺放1'
    LYING_Y = '躺放2'

    def __hash__(self):
        return id(self)

@dataclass
class Position:
    """代表一个三维空间中的坐标点。"""
    x: float
    y: float
    z: float

    def __eq__(self, other):
        return isinstance(other, Position) and self.x == other.x and self.y == other.y and self.z == other.z

    def __hash__(self):
        return hash((self.x, self.y, self.z))

class Cargo:
    """代表一件货物及其所有属性和状态"""
    def __init__(self, cargo_id: str, supplier: str, length: float, width: float, height: float, weight: float):
        self.cargo_id = cargo_id
        self.supplier = supplier
        self.original_dims = (length, width, height)
        self.length = length
        self.width = width
        self.height = height
        self.weight = weight
        self.volume = length * width * height
        self.current_orientation: PlacementType = None

    def set_orientation(self, placement_type: PlacementType):
        """根据摆放方式，更新货物的长宽高"""
        original_l, original_w, original_h = self.original_dims
        self.current_orientation = placement_type

        if placement_type == PlacementType.UPRIGHT_X:
            self.length, self.width, self.height = original_l, original_w, original_h
        elif placement_type == PlacementType.UPRIGHT_Y:
            self.length, self.width, self.height = original_w, original_l, original_h
        elif placement_type == PlacementType.SIDE_LYING_X:
            self.length, self.width, self.height = original_l, original_h, original_w
        elif placement_type == PlacementType.SIDE_LYING_Y:
            self.length, self.width, self.height = original_h, original_l, original_w
        elif placement_type == PlacementType.LYING_X:
            self.length, self.width, self.height = original_w, original_h, original_l
        elif placement_type == PlacementType.LYING_Y:
            self.length, self.width, self.height = original_h, original_w, original_l

    def __eq__(self, other):
        if not isinstance(other, Cargo):
            return False
        return id(self) == id(other)

    def __hash__(self):
        return id(self)
        
class PlacedItem:
    """代表一个已经放置在容器中的货物"""
    def __init__(self, cargo: Cargo, position: Position, orientation: PlacementType):
        self.cargo = cargo
        self.position = position
        self.orientation = orientation
        # 立即计算并缓存当前尺寸，避免重复计算
        temp_cargo = copy.copy(self.cargo)
        temp_cargo.set_orientation(self.orientation)
        self.current_dims = (temp_cargo.length, temp_cargo.width, temp_cargo.height)
    
    def __eq__(self, other):
        return isinstance(other, PlacedItem) and self.cargo == other.cargo and self.position == other.position and self.orientation == other.orientation

class PackingSolution:
    """代表一个完整的装载方案（状态），不再依赖SpaceManager"""
    def __init__(self, all_cargo: List[Cargo]):
        self.placed_items: List[PlacedItem] = []
        self.unloaded_cargo_set: Set[Cargo] = set(all_cargo)
        self.total_volume = 0.0

    def add_item(self, cargo_to_add: Cargo, position: Position, orientation: PlacementType):
        """向方案中添加一个新放置的货物"""
        new_item = PlacedItem(cargo_to_add, position, orientation)
        self.placed_items.append(new_item)
        self.total_volume += cargo_to_add.volume
        
        if cargo_to_add in self.unloaded_cargo_set:
            self.unloaded_cargo_set.remove(cargo_to_add)

    def remove_item(self, item_to_remove: PlacedItem):
        """从方案中移除一个已放置的货物"""
        self.placed_items.remove(item_to_remove)
        self.unloaded_cargo_set.add(item_to_remove.cargo)
        self.total_volume -= item_to_remove.cargo.volume

    def copy(self) -> 'PackingSolution':
        """创建一个当前解的浅拷贝，用于快速生成邻居解"""
        new_solution = PackingSolution([])
        new_solution.placed_items = self.placed_items[:]
        new_solution.total_volume = self.total_volume
        new_solution.unloaded_cargo_set = self.unloaded_cargo_set.copy()
        return new_solution

class SupplierRegion:
    """定义一个供应商的装载区域"""
    def __init__(self, name: str, start_position: float, end_position: float):
        self.name = name
        self.start_position = start_position
        self.end_position = end_position

    def is_inside(self, item_start_x: float, item_end_x: float) -> bool:
        """检查一个物品是否完全位于该区域内"""
        return self.start_position <= item_start_x and item_end_x <= self.end_position

@dataclass
class LayerInfo:
    """存储单层的信息"""
    start_z: float
    end_z: float
    index: int

class LayerStrategy:
    """负责根据货物高度分布和物理约束，动态地决定容器的分层策略"""
    def __init__(self, container_dims: Tuple[int, int, int], all_cargo: List[Cargo]):
        self.container_height = container_dims[2]
        self.layers = self._create_layers(all_cargo)

    def _create_layers(self, all_cargo: List[Cargo]) -> List[LayerInfo]:
        """基于货物高度创建分层"""
        if not all_cargo:
            return [LayerInfo(0, self.container_height, 0)]

        sorted_heights = sorted(list(set(c.height for c in all_cargo)))
        
        if len(sorted_heights) > 2:
            h1 = sorted_heights[len(sorted_heights) // 3]
            h2 = sorted_heights[len(sorted_heights) * 2 // 3]
            
            layer1_height = h1 + 1.0
            layer2_height = h2 + 1.0

            if layer1_height + layer2_height > self.container_height:
                layer1_height = self.container_height / 3
                layer2_height = self.container_height / 3

            return [
                LayerInfo(0, layer1_height, 0),
                LayerInfo(layer1_height, layer1_height + layer2_height, 1),
                LayerInfo(layer1_height + layer2_height, self.container_height, 2)
            ]
        else:
            return [LayerInfo(0, self.container_height, 0)]

    def get_layer_index(self, z: float) -> int:
        for layer in self.layers:
            if layer.start_z <= z < layer.end_z:
                return layer.index
        return -1

    def get_allowed_placements(self, z_position: float) -> Set[PlacementType]:
        layer_index = self.get_layer_index(z_position)
        if layer_index in [0, 1]:
            return {
                PlacementType.SIDE_LYING_X, PlacementType.SIDE_LYING_Y,
                PlacementType.LYING_X, PlacementType.LYING_Y
            }
        return set(PlacementType)

    def print_layers(self):
        print("\n=== 动态分层策略结果 ===")
        for layer in self.layers:
            print(f"  层 {layer.index + 1}: 高度 {layer.start_z:.1f}cm - {layer.end_z:.1f}cm (厚度: {layer.end_z - layer.start_z:.1f}cm)")

class SimulatedAnnealingOptimizer:
    def __init__(self, container_dims: Tuple[int, int, int]):
        self.container_dims = container_dims
        self.all_cargo: List[Cargo] = []
        self.regions: List[SupplierRegion] = []
        self.layer_strategy: LayerStrategy = None
        
        self.initial_temp = 10000.0
        self.final_temp = 1.0
        self.cooling_rate = 0.85 # 快速优化模式
        self.junction_tolerance = 50.0 # 交接区容差 (cm)
        self.is_in_optimization_phase = False # 两阶段优化开关

    def _preprocess_cargo(self, cargo_data: List[dict]) -> List[Cargo]:
        cargo_list = []
        supplier_pattern = re.compile(r'（(.*?)）')
        for item in cargo_data:
            match = supplier_pattern.search(item.get('貨物名稱', ''))
            supplier_name = match.group(1) if match else "UnknownSupplier"
            for _ in range(int(item['數量'])):
                cargo_list.append(Cargo(
                    cargo_id=item.get('貨物名稱', 'Unknown'),
                    supplier=supplier_name,
                    length=item['長度'], width=item['寬度'], height=item['高度'], weight=item['重量']
                ))
        return cargo_list

    def _create_supplier_regions(self, suppliers_sequence: List[str], all_cargo: List[Cargo]):
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

    def optimize(self, suppliers_sequence: List[str], cargo_data: List[dict]):
        print("开始模拟退火优化流程...")
        start_time = time.time()

        self.all_cargo = self._preprocess_cargo(cargo_data)
        self.regions = self._create_supplier_regions(suppliers_sequence, self.all_cargo)
        self.layer_strategy = LayerStrategy(self.container_dims, self.all_cargo)
        self.layer_strategy.print_layers()

        # --- Phase 1: Build Initial Solution with HARD Boundaries ---
        self.is_in_optimization_phase = False
        print("\n--> Phase 1: Building initial solution with HARD boundaries...")
        initial_solution = self._create_initial_solution()
        if not initial_solution:
            print("错误：无法生成初始解。")
            return
            
        current_solution = initial_solution
        best_solution = initial_solution
        print(f"Initial solution generated, load rate: {self._get_volume_ratio(initial_solution):.2%}")

        current_energy = self._calculate_energy(initial_solution)
        best_energy = current_energy
        temp = self.initial_temp

        # --- Phase 2: Start SA Optimization with FLEXIBLE Boundaries ---
        self.is_in_optimization_phase = True
        print("--> Phase 2: Starting SA optimization with FLEXIBLE boundaries...")
        print("开始模拟退火主循环...")
        
        # 引入内循环，并在每个温度点上进行多次迭代
        steps_at_each_temp = 20 # 快速优化模式
        
        # 修改冷却逻辑，以温度点为单位
        temp_steps = 0
        total_temp_steps = int(abs(math.log(self.final_temp / self.initial_temp) / math.log(self.cooling_rate))) if self.cooling_rate < 1 else 1

        while temp > self.final_temp:
            for _ in range(steps_at_each_temp): # 内循环
                neighbor_solution = self._generate_neighbor(current_solution)
                if not neighbor_solution or not neighbor_solution.placed_items: 
                    continue

                neighbor_energy = self._calculate_energy(neighbor_solution)
                delta_energy = neighbor_energy - current_energy

                if delta_energy > 0:
                    current_solution = neighbor_solution
                    current_energy = neighbor_energy
                    if current_energy > best_energy:
                        best_solution = neighbor_solution
                        best_energy = neighbor_energy
                else:
                    if random.random() < self._acceptance_probability(delta_energy, temp):
                        current_solution = neighbor_solution
                        current_energy = neighbor_energy
            
            temp *= self.cooling_rate
            temp_steps += 1
            
            # 更新日志
            print(f"\r优化中... [温度段: {temp_steps}/{total_temp_steps}] [当前温度: {temp:.1f}] [最优装载率: {self._get_volume_ratio(best_solution):.2%}]", end="")


        print("\n模拟退火完成。")
        self._print_final_report(best_solution, time.time() - start_time)
        self._create_3d_visualization(best_solution)

    def _get_candidate_points(self, solution: PackingSolution) -> List[Position]:
        """
        生成有价值的候选放置点。
        改进：在货物顶部生成多层候选点，充分利用垂直空间。
        """
        points = {Position(0, 0, 0)}
        
        for item in solution.placed_items:
            l, w, h = item.current_dims
            pos = item.position
            
            # 原有的边缘候选点
            points.add(Position(pos.x + l, pos.y, pos.z))
            points.add(Position(pos.x, pos.y + w, pos.z))
            points.add(Position(pos.x, pos.y, pos.z + h))
            
            # 新增：在矮货物（高度<20cm）顶部生成更多候选点
            if h < 20 and pos.z + h < self.container_dims[2] - 10:
                # 在货物顶部的不同位置生成候选点
                # 这样可以让多个小货物并排放在一个大货物上
                points.add(Position(pos.x + l/4, pos.y, pos.z + h))
                points.add(Position(pos.x + l/2, pos.y, pos.z + h))
                points.add(Position(pos.x + 3*l/4, pos.y, pos.z + h))
                points.add(Position(pos.x, pos.y + w/4, pos.z + h))
                points.add(Position(pos.x, pos.y + w/2, pos.z + h))
                points.add(Position(pos.x, pos.y + 3*w/4, pos.z + h))
                
                # 对于特别矮的货物（<10cm），生成更密集的网格
                if h < 10:
                    for x_ratio in [0.2, 0.4, 0.6, 0.8]:
                        for y_ratio in [0.2, 0.4, 0.6, 0.8]:
                            points.add(Position(pos.x + l*x_ratio, pos.y + w*y_ratio, pos.z + h))
        
        # 按LBD黄金法则排序 (z, y, x)
        return sorted(list(points), key=lambda p: (p.z, p.y, p.x))

    def _find_best_placement_for_cargo(self, cargo: Cargo, solution: PackingSolution, region: SupplierRegion) -> Tuple[Position, PlacementType]:
        """为单个货物在指定区域内寻找最佳放置位置(LBD策略)"""
        # 获取基础候选点
        point_set = set(self._get_candidate_points(solution))
        # 【关键修复】确保每个区域都有一个起始锚点
        point_set.add(Position(region.start_position, 0, 0))
        candidate_points = sorted(list(point_set), key=lambda p: (p.z, p.y, p.x))
        
        for point in candidate_points:
            allowed_placements = self.layer_strategy.get_allowed_placements(point.z)
            for orientation in allowed_placements:
                cargo.set_orientation(orientation)
                if not region.is_inside(point.x, point.x + cargo.length):
                    continue
                    
                if self._is_valid_placement(cargo, point, orientation, solution, region):
                    return point, orientation
        return None, None

    def _calculate_support_ratio(self, cargo: Cargo, position: Position, solution: PackingSolution) -> float:
        """
        计算货物底面的支撑率
        返回值：0.0-1.0之间，表示底面被支撑的比例
        """
        # 地面货物始终有100%支撑
        if position.z == 0:
            return 1.0
        
        cargo_length = cargo.length
        cargo_width = cargo.width
        bottom_area = cargo_length * cargo_width
        
        if bottom_area == 0:
            return 0.0
        
        supported_area = 0.0
        
        # 检查所有已放置的货物
        for item in solution.placed_items:
            item_top_z = item.position.z + item.current_dims[2]
            
            # 只考虑顶部正好接触当前货物底部的物品（容差0.1cm）
            if abs(item_top_z - position.z) > 0.1:
                continue
            
            # 计算XY平面上的重叠区域
            x_overlap_start = max(position.x, item.position.x)
            x_overlap_end = min(position.x + cargo_length, item.position.x + item.current_dims[0])
            y_overlap_start = max(position.y, item.position.y)
            y_overlap_end = min(position.y + cargo_width, item.position.y + item.current_dims[1])
            
            # 如果有重叠
            if x_overlap_start < x_overlap_end and y_overlap_start < y_overlap_end:
                overlap_area = (x_overlap_end - x_overlap_start) * (y_overlap_end - y_overlap_start)
                supported_area += overlap_area
        
        return supported_area / bottom_area

    def _check_center_stability(self, cargo: Cargo, position: Position, solution: PackingSolution) -> bool:
        """
        检查货物重心是否在支撑范围内，确保稳定性
        """
        if position.z == 0:
            return True  # 地面货物始终稳定
        
        # 计算货物重心
        center_x = position.x + cargo.length / 2
        center_y = position.y + cargo.width / 2
        
        # 找出支撑区域的边界
        support_min_x = float('inf')
        support_max_x = float('-inf')
        support_min_y = float('inf')
        support_max_y = float('-inf')
        has_support = False
        
        for item in solution.placed_items:
            item_top_z = item.position.z + item.current_dims[2]
            
            # 只考虑正好支撑的货物
            if abs(item_top_z - position.z) > 0.1:
                continue
            
            # 计算支撑区域
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
        
        if not has_support:
            return False
        
        # 重心必须在支撑区域内，保留15%的安全边距
        margin_x = (support_max_x - support_min_x) * 0.15
        margin_y = (support_max_y - support_min_y) * 0.15
        
        # 检查重心是否在安全范围内
        return (support_min_x + margin_x <= center_x <= support_max_x - margin_x and
                support_min_y + margin_y <= center_y <= support_max_y - margin_y)

    def _is_valid_placement(self, cargo: Cargo, position: Position, orientation: PlacementType, solution: PackingSolution, region: SupplierRegion) -> bool:
        """
        使用高效的AABB碰撞检测。
        【两阶段策略】根据当前是否在优化阶段，使用不同边界检查逻辑。
        - 初始解构建: 硬边界
        - 模拟退火: 柔性边界 (带交接区)
        """
        cargo.set_orientation(orientation)
        px, py, pz = position.x, position.y, position.z
        pl, pw, ph = cargo.length, cargo.width, cargo.height

        # Part 1: Shared Y/Z-axis and Layer Checks (Fast checks first)
        if not (py >= 0 and pz >= 0 and
                py + pw <= self.container_dims[1] and
                pz + ph <= self.container_dims[2]):
            return False
        
        if orientation not in self.layer_strategy.get_allowed_placements(pz):
            return False

        # Part 2: Phase-Dependent X-axis Boundary Check
        if self.is_in_optimization_phase:
            # 【优化阶段】使用柔性边界
            region_index = self.regions.index(region)
            effective_start_x = region.start_position
            effective_end_x = region.end_position
            
            if region_index > 0:
                effective_start_x -= self.junction_tolerance
            if region_index < len(self.regions) - 1:
                effective_end_x += self.junction_tolerance
            
            effective_start_x = max(0, effective_start_x)
            effective_end_x = min(self.container_dims[0], effective_end_x)

            if not (px >= effective_start_x and px + pl <= effective_end_x):
                return False
                
            # 锚点规则: 确保货物与原始区域有重叠
            if not (px < region.end_position and px + pl > region.start_position):
                return False
        else:
            # 【初始解阶段】使用硬边界
            if not (px >= region.start_position and px + pl <= region.end_position):
                return False

        # Part 3: Shared Collision Check (Most expensive check last)
        for item in solution.placed_items:
            ix, iy, iz = item.position.x, item.position.y, item.position.z
            il, iw, ih = item.current_dims
            
            if (px < ix + il and px + pl > ix and
                py < iy + iw and py + pw > iy and
                pz < iz + ih and pz + ph > iz):
                return False  # 发生碰撞
        
        # Part 4: Support Check (新增支撑检查)
        # 只对非地面货物进行支撑检查
        if pz > 0:
            support_ratio = self._calculate_support_ratio(cargo, position, solution)
            # 提高到70%的底面支撑要求，确保运输稳定性
            if support_ratio < 0.7:
                return False
            
            # Part 5: Center Stability Check (重心稳定性检查)
            if not self._check_center_stability(cargo, position, solution):
                return False
                
        return True # 所有检查通过，放置有效

    def _generate_neighbor(self, solution: PackingSolution) -> PackingSolution:
        """
        生成一个基于"候选点"的邻域解，取代纯随机。
        【方案C+D】引入多阶段策略，并加入'batch_repack'操作。
        """
        neighbor = solution.copy()
        
        current_rate = self._get_volume_ratio(solution)
        
        # 根据当前装载率动态选择策略
        if current_rate < 0.86:
            operations = ['add', 'move', 'remove', 'swap', 'reinsert', 'batch_repack']
            weights    = [0.3, 0.15, 0.1, 0.2, 0.2, 0.05] # 大幅降低batch_repack的权重
        elif current_rate < 0.90:
            operations = ['add', 'move', 'remove', 'swap', 'reinsert', 'batch_repack']
            weights    = [0.1, 0.15, 0.1, 0.25, 0.3, 0.1]  # 中期，权重也较低
        else:
            operations = ['move', 'remove', 'swap', 'reinsert', 'batch_repack']
            weights    = [0.1, 0.05, 0.3, 0.4, 0.15] # 后期，作为重要的破局手段，但权重仍受控
        
        operation = random.choices(operations, weights=weights, k=1)[0]

        if operation == 'add':
            if not neighbor.unloaded_cargo_set: return neighbor
            
            cargo_to_add = random.choice(list(neighbor.unloaded_cargo_set))
            region = next((r for r in self.regions if r.name == cargo_to_add.supplier), None)
            if region:
                pos, orientation = self._find_best_placement_for_cargo(cargo_to_add, neighbor, region)
                if pos and orientation:
                    neighbor.add_item(cargo_to_add, pos, orientation)
            return neighbor

        if operation == 'remove':
            if not neighbor.placed_items: return neighbor
            item_to_remove = random.choice(neighbor.placed_items)
            neighbor.remove_item(item_to_remove)
            return neighbor

        if operation == 'move':
            if not neighbor.placed_items: return neighbor
            
            item_to_move = random.choice(neighbor.placed_items)
            original_pos = item_to_move.position
            original_orient = item_to_move.orientation
            
            neighbor.remove_item(item_to_move)

            region = next((r for r in self.regions if r.name == item_to_move.cargo.supplier), None)
            if region:
                pos, orientation = self._find_best_placement_for_cargo(item_to_move.cargo, neighbor, region)
                if pos and orientation:
                    neighbor.add_item(item_to_move.cargo, pos, orientation)
                else: # 如果找不到新位置，把物品放回去，保持解的有效性
                    neighbor.add_item(item_to_move.cargo, original_pos, original_orient)

            return neighbor

        if operation == 'swap':
            if len(neighbor.placed_items) < 2: return neighbor

            item_a, item_b = random.sample(neighbor.placed_items, 2)
            
            # 必须确保交换的两个物品来自同一个供应商，以遵守区域约束
            if item_a.cargo.supplier != item_b.cargo.supplier:
                return neighbor

            # 【关键修复】动态获取正确的供应商区域
            region = next((r for r in self.regions if r.name == item_a.cargo.supplier), None)
            if not region:
                return neighbor # 如果找不到区域，则不执行任何操作

            pos_a, orient_a = item_a.position, item_a.orientation
            pos_b, orient_b = item_b.position, item_b.orientation
            
            # 移除A和B，以便检查新位置
            temp_solution = neighbor.copy()
            temp_solution.remove_item(item_a)
            temp_solution.remove_item(item_b)

            # 交叉检查有效性
            can_b_go_to_a = self._is_valid_placement(item_b.cargo, pos_a, orient_b, temp_solution, region)
            can_a_go_to_b = self._is_valid_placement(item_a.cargo, pos_b, orient_a, temp_solution, region)
            
            if can_a_go_to_b and can_b_go_to_a:
                # 真正执行交换
                neighbor.remove_item(item_a)
                neighbor.remove_item(item_b)
                neighbor.add_item(item_a.cargo, pos_b, orient_a)
                neighbor.add_item(item_b.cargo, pos_a, orient_b)
            
            return neighbor

        if operation == 'reinsert':
            if not neighbor.placed_items: return neighbor

            item_to_reinsert = random.choice(neighbor.placed_items)
            original_pos = item_to_reinsert.position
            original_orient = item_to_reinsert.orientation
            
            neighbor.remove_item(item_to_reinsert)

            region = next((r for r in self.regions if r.name == item_to_reinsert.cargo.supplier), None)
            if region:
                pos, orientation = self._find_best_placement_for_cargo(item_to_reinsert.cargo, neighbor, region)
                if pos and orientation:
                    neighbor.add_item(item_to_reinsert.cargo, pos, orientation)
                else: # 安全回退保障
                    neighbor.add_item(item_to_reinsert.cargo, original_pos, original_orient)
            else: # 如果找不到区域，也放回去
                neighbor.add_item(item_to_reinsert.cargo, original_pos, original_orient)

            return neighbor

        if operation == 'batch_repack':
            if not neighbor.placed_items: return neighbor

            # 1. 随机选择一个区域进行重整
            target_region = random.choice(self.regions)
            items_in_region = [item for item in neighbor.placed_items if item.cargo.supplier == target_region.name]

            # 2. 如果物品太少，不执行操作
            if len(items_in_region) < 5: return neighbor

            # 3. 随机移除该区域5%-10%的货物 (降低强度)
            repack_count = max(1, int(len(items_in_region) * random.uniform(0.05, 0.1)))
            items_to_repack = random.sample(items_in_region, repack_count)
            
            # 4. 从当前解中移除这些货物
            cargos_to_repack = []
            for item in items_to_repack:
                neighbor.remove_item(item)
                cargos_to_repack.append(item.cargo)
            
            # 5. 将移除的货物按体积从大到小排序，尝试用LBD策略重新装入
            sorted_cargos = sorted(cargos_to_repack, key=lambda c: c.volume, reverse=True)
            for cargo in sorted_cargos:
                # 复用我们最核心的LBD放置函数
                pos, orientation = self._find_best_placement_for_cargo(cargo, neighbor, target_region)
                if pos and orientation:
                    neighbor.add_item(cargo, pos, orientation)
            # 任何无法装回的货物，会自动留在unloaded_cargo_set中，这完全符合模拟退火的逻辑

            return neighbor
            
        return neighbor

    def _calculate_energy(self, solution: PackingSolution) -> float:
        return solution.total_volume

    def _acceptance_probability(self, delta_energy: float, temperature: float) -> float:
        if delta_energy > 0: return 1.0
        return math.exp(delta_energy / temperature)

    def _create_initial_solution(self) -> PackingSolution:
        """
        创建一个高质量的初始解，基于LBD（左-下-后）启发式规则。
        该策略通过系统性地探索高质量的候选点来取代随机性，旨在达到最高的初始装载率。
        """
        print("开始构建高质量LBD初始解...")
        solution = PackingSolution(self.all_cargo)
        
        # 按体积从大到小排序，优先处理大件货物
        sorted_cargo = sorted(self.all_cargo, key=lambda c: c.volume, reverse=True)
        
        total_cargo_count = len(sorted_cargo)
        for i, cargo in enumerate(sorted_cargo):
            print(f"\r构建初始解... [处理: {i+1}/{total_cargo_count}]", end="")
            
            region = next((r for r in self.regions if r.name == cargo.supplier), None)
            if not region: continue

            best_placement = None
            
            # 1. 获取候选点
            point_set = set(self._get_candidate_points(solution))
            # 【关键修复】确保每个区域都有一个起始锚点
            point_set.add(Position(region.start_position, 0, 0))
            candidate_points = sorted(list(point_set), key=lambda p: (p.z, p.y, p.x))
            
            # 2. 遍历所有候选点，寻找最佳位置
            for point in candidate_points:
                # 这是一个基础检查，更详细的检查在_is_valid_placement中
                if not (region.start_position <= point.x < region.end_position):
                    continue

                allowed_placements = self.layer_strategy.get_allowed_placements(point.z)
                for orientation in allowed_placements:
                    cargo.set_orientation(orientation)
                    
                    if self._is_valid_placement(cargo, point, orientation, solution, region):
                        # 找到第一个可行的位置就接受，因为点是排序过的
                        best_placement = (point, orientation)
                        break
                if best_placement:
                    break
            
            # 3. 如果找到了位置，就放置货物
            if best_placement:
                pos, orient = best_placement
                solution.add_item(cargo, pos, orient)
        
        print(f"\n初始解构建完成。")
        return solution

    def _get_volume_ratio(self, solution: PackingSolution) -> float:
        if not solution: return 0.0
        container_volume = self.container_dims[0] * self.container_dims[1] * self.container_dims[2]
        if container_volume == 0: return 0.0
        return solution.total_volume / container_volume

    def _print_final_report(self, solution: PackingSolution, duration: float):
        print("\n\n=== 最终装载报告 ===")
        final_ratio = self._get_volume_ratio(solution)
        print(f"最终装柜率: {final_ratio:.2%}")
        container_volume = self.container_dims[0] * self.container_dims[1] * self.container_dims[2]
        print(f"总装载体积: {solution.total_volume:.2f} cm³ / {container_volume:.2f} cm³")
        print(f"总装载件数: {len(solution.placed_items)}")
        if solution.placed_items:
            suppliers_in_solution = sorted(list(set(item.cargo.supplier for item in solution.placed_items)))
            print(f"涉及供应商: {', '.join(suppliers_in_solution)}")
        print(f">>> 优化总耗时: {duration:.2f} 秒")

    def _create_3d_visualization(self, solution: PackingSolution, save_path: str = "container_3d_visualization.png"):
        """创建装载方案的3D可视化图"""
        print("\n正在生成3D可视化图...")
        
        # 设置中文字体，解决中文显示问题
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 设置供应商颜色映射
        supplier_colors = {
            '纽蓝': 'lightblue',
            '海信': 'lightgreen',
            '福美高': 'lightcoral'
        }
        
        # 绘制集装箱边框
        container_x, container_y, container_z = self.container_dims
        # 定义集装箱的8个顶点
        container_vertices = [
            [0, 0, 0], [container_x, 0, 0], [container_x, container_y, 0], [0, container_y, 0],  # 底面
            [0, 0, container_z], [container_x, 0, container_z], [container_x, container_y, container_z], [0, container_y, container_z]  # 顶面
        ]
        
        # 定义集装箱的12条边
        container_edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面边
            [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面边
            [0, 4], [1, 5], [2, 6], [3, 7]   # 垂直边
        ]
        
        # 绘制集装箱边框线
        for edge in container_edges:
            points = [container_vertices[edge[0]], container_vertices[edge[1]]]
            ax.plot3D(*zip(*points), 'k-', linewidth=1, alpha=0.3)
        
        # 绘制每个已放置的货物
        for i, item in enumerate(solution.placed_items):
            x, y, z = item.position.x, item.position.y, item.position.z
            l, w, h = item.current_dims
            
            # 获取供应商对应的颜色
            color = supplier_colors.get(item.cargo.supplier, 'gray')
            
            # 定义立方体的8个顶点
            vertices = [
                [x, y, z], [x+l, y, z], [x+l, y+w, z], [x, y+w, z],  # 底面
                [x, y, z+h], [x+l, y, z+h], [x+l, y+w, z+h], [x, y+w, z+h]  # 顶面
            ]
            
            # 定义立方体的6个面
            faces = [
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # 后面
                [vertices[0], vertices[3], vertices[7], vertices[4]],  # 左面
                [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
                [vertices[4], vertices[5], vertices[6], vertices[7]]   # 顶面
            ]
            
            # 创建3D多边形集合并添加到图中
            face_collection = Poly3DCollection(faces, alpha=0.7, facecolor=color, edgecolor='black', linewidth=0.5)
            ax.add_collection3d(face_collection)
        
        # 设置轴标签和标题
        ax.set_xlabel('长度 (cm)', fontsize=12)
        ax.set_ylabel('宽度 (cm)', fontsize=12)
        ax.set_zlabel('高度 (cm)', fontsize=12)
        ax.set_title(f'集装箱装载3D可视化\n装载率: {self._get_volume_ratio(solution):.2%} | 装载件数: {len(solution.placed_items)}件', fontsize=16)
        
        # 设置轴的显示范围
        ax.set_xlim([0, container_x])
        ax.set_ylim([0, container_y])
        ax.set_zlim([0, container_z])
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=supplier) 
                          for supplier, color in supplier_colors.items()]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))
        
        # 设置最佳视角
        ax.view_init(elev=20, azim=45)
        
        # 保存图片
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"3D可视化图已保存至: {save_path}")
        
        # 显示图形（用户可以交互旋转查看）
        plt.show()

def load_cargo_data(file_path: str) -> Tuple[List[str], List[dict]]:
    try:
        excel_data = pd.read_excel(file_path)
        cargo_data = excel_data.to_dict('records')
        supplier_pattern = re.compile(r'（(.*?)）')
        suppliers_sequence = []
        seen_suppliers = set()
        for item in cargo_data:
            match = supplier_pattern.search(item.get('貨物名稱', ''))
            if match:
                supplier_name = match.group(1)
                if supplier_name not in seen_suppliers:
                    seen_suppliers.add(supplier_name)
                    suppliers_sequence.append(supplier_name)
        print("=== 真实数据加载完成 ===")
        return suppliers_sequence, cargo_data
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None, None

def main():
    container_dims = (1180, 230, 260)
    suppliers_sequence, cargo_data = load_cargo_data("装柜0538.xlsx")
    if not cargo_data: return

    total_items = sum(item['數量'] for item in cargo_data)
    total_volume_theoretical = sum(item['長度']*item['寬度']*item['高度']*item['數量'] for item in cargo_data)
    print(f"总货物种类: {len(cargo_data)}")
    print(f"总货物数量: {total_items}")
    print(f"理论总体积: {total_volume_theoretical:.2f}cm³")
    print(f"供应商顺序: {suppliers_sequence}")

    optimizer = SimulatedAnnealingOptimizer(container_dims)
    optimizer.optimize(suppliers_sequence=suppliers_sequence, cargo_data=cargo_data)

if __name__ == "__main__":
    main()