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
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

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

class SortingStrategy(Enum):
    """定义货物排序策略的枚举"""
    VOLUME_DESC = '体积从大到小'
    AREA_DESC = '最大底面积从大到小'
    MAX_DIM_DESC = '最长边从大到小'
    RANDOM_SHUFFLE = '随机打乱'

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
        """
        主优化流程。
        仅执行高质量初始解的构建，并直接输出结果。
        """
        print("开始贪心优化流程...")
        start_time = time.time()

        self.all_cargo = self._preprocess_cargo(cargo_data)
        self.regions = self._create_supplier_regions(suppliers_sequence, self.all_cargo)
        self.layer_strategy = LayerStrategy(self.container_dims, self.all_cargo)
        self.layer_strategy.print_layers()

        # --- 构建高质量的初始解 ---
        print("\n--> 正在构建初始解...")
        solution = self._create_initial_solution()
        if not solution:
            print("错误：无法生成初始解。")
            return
            
        print(f"\n初始解构建完成，装载率: {self._get_volume_ratio(solution):.2%}")

        # --- 输出最终报告和可视化 ---
        self._print_final_report(solution, time.time() - start_time)
        self._create_3d_visualization(solution)
        self._export_solution_to_excel(solution)

    def _get_candidate_points(self, solution: PackingSolution) -> List[Position]:
        """
        生成有价值的候选放置点。
        优化版本：减少候选点密度，提升性能。
        """
        points = {Position(0, 0, 0)}
        
        for item in solution.placed_items:
            l, w, h = item.current_dims
            pos = item.position
            
            # 原有的边缘候选点
            points.add(Position(pos.x + l, pos.y, pos.z))
            points.add(Position(pos.x, pos.y + w, pos.z))
            points.add(Position(pos.x, pos.y, pos.z + h))
            
            # 优化：只在特别矮的货物（高度<10cm）顶部生成少量候选点
            if h < 10 and pos.z + h < self.container_dims[2] - 10:
                # 只在货物顶部的中心和边缘生成候选点，减少数量
                points.add(Position(pos.x + l/2, pos.y + w/2, pos.z + h))  # 中心点
                points.add(Position(pos.x, pos.y, pos.z + h))              # 左下角
                points.add(Position(pos.x + l, pos.y, pos.z + h))          # 右下角
        
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
        使用高效的AABB碰撞检测和硬边界约束。
        """
        cargo.set_orientation(orientation)
        px, py, pz = position.x, position.y, position.z
        pl, pw, ph = cargo.length, cargo.width, cargo.height

        # Part 1: 容器Y/Z轴边界和分层检查
        if not (py >= 0 and pz >= 0 and
                py + pw <= self.container_dims[1] and
                pz + ph <= self.container_dims[2]):
            return False
        
        if orientation not in self.layer_strategy.get_allowed_placements(pz):
            return False

        # Part 2: 供应商区域硬边界检查 (X轴)
        if not (px >= region.start_position and px + pl <= region.end_position):
            return False

        # Part 3: 与已放置货物的碰撞检查
        for item in solution.placed_items:
            ix, iy, iz = item.position.x, item.position.y, item.position.z
            il, iw, ih = item.current_dims
            
            if (px < ix + il and px + pl > ix and
                py < iy + iw and py + pw > iy and
                pz < iz + ih and pz + ph > iz):
                return False  # 发生碰撞
        
        # Part 4: 支撑检查 (仅非地面货物)
        if pz > 0:
            support_ratio = self._calculate_support_ratio(cargo, position, solution)
            # 提高到70%的底面支撑要求，确保运输稳定性
            if support_ratio < 0.7:
                return False
            
            # Part 5: 重心稳定性检查
            if not self._check_center_stability(cargo, position, solution):
                return False
                
        return True # 所有检查通过，放置有效

    def _create_initial_solution(self, strategy: SortingStrategy = SortingStrategy.VOLUME_DESC, strategy_index: int = 0) -> PackingSolution:
        """
        创建一个高质量的初始解，基于LBD（左-下-后）启发式规则。
        该策略通过系统性地探索高质量的候选点来取代随机性，旨在达到最高的初始装载率。
        :param strategy: 决定货物处理顺序的排序策略。
        :param strategy_index: 用于tqdm进度条定位的索引。
        """
        # 不再需要这行打印，tqdm会提供更详细的信息
        # print(f"开始使用策略 [{strategy.value}] 构建初始解...")
        solution = PackingSolution(self.all_cargo)
        
        # 根据所选策略对货物进行排序
        sorted_cargo = []
        if strategy == SortingStrategy.VOLUME_DESC:
            sorted_cargo = sorted(self.all_cargo, key=lambda c: c.volume, reverse=True)
        elif strategy == SortingStrategy.AREA_DESC:
            # 使用原始尺寸计算最大底面积
            sorted_cargo = sorted(self.all_cargo, key=lambda c: c.original_dims[0] * c.original_dims[1], reverse=True)
        elif strategy == SortingStrategy.MAX_DIM_DESC:
            # 使用原始尺寸计算最长边
            sorted_cargo = sorted(self.all_cargo, key=lambda c: max(c.original_dims), reverse=True)
        elif strategy == SortingStrategy.RANDOM_SHUFFLE:
            sorted_cargo = list(self.all_cargo)
            random.shuffle(sorted_cargo)
        
        # 使用tqdm包装循环以显示进度条
        # desc是进度条的描述，position让每个进度条独立占一行
        # leave=True, 防止因进程结束时清除进度条导致的光标混乱问题
        # ncols=100, dynamic_ncols=False, 禁用动态宽度调整，进一步避免多进程渲染冲突
        for cargo in tqdm(sorted_cargo, desc=f"策略 {strategy.value}", position=strategy_index, leave=True, ncols=100, dynamic_ncols=False):
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
        
        final_rate = self._get_volume_ratio(solution)
        # print(f"策略 [{strategy.value}] 构建完成。装载率: {final_rate:.2%}, 件数: {len(solution.placed_items)}")
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

    def _export_solution_to_excel(self, solution: PackingSolution, save_path: str = "装载方案详情.xlsx"):
        """将详细的装载方案导出为Excel文件。"""
        print(f"\n正在将详细装载方案导出至Excel文件: {save_path}...")
        
        if not solution.placed_items:
            print("警告：装载方案为空，无法导出Excel。")
            return

        solution_data = []
        for item in solution.placed_items:
            solution_data.append({
                '貨物名稱': item.cargo.cargo_id,
                '供應商': item.cargo.supplier,
                '原始長度': item.cargo.original_dims[0],
                '原始寬度': item.cargo.original_dims[1],
                '原始高度': item.cargo.original_dims[2],
                '放置座標X': round(item.position.x, 2),
                '放置座標Y': round(item.position.y, 2),
                '放置座標Z': round(item.position.z, 2),
                '擺放方式': item.orientation.value,
                '當前長度': item.current_dims[0],
                '當前寬度': item.current_dims[1],
                '當前高度': item.current_dims[2],
            })
        
        df = pd.DataFrame(solution_data)
        
        # 按放置位置（X, Y, Z）排序，更好地反映实际装载顺序
        print("正在根据物理位置排序方案，以便查看...")
        df_sorted = df.sort_values(by=['放置座標X', '放置座標Y', '放置座標Z'])
        
        try:
            df_sorted.to_excel(save_path, index=False, engine='openpyxl')
            print(f"Excel文件已成功保存至: {save_path}")
        except Exception as e:
            print(f"错误：导出Excel文件失败。原因: {e}")

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
        
        # 不再调用plt.show()，因为它会阻塞程序，尤其是在非交互式环境下
        # plt.show()
        plt.close(fig) # 添加此行以关闭图形对象，释放内存

class PermutationOptimizer:
    """
    排列优化器，寻找最优的供应商取货顺序。
    实现了混合策略：
    - 当供应商数量较少(<=4)时，进行全量暴力计算，确保找到理论最优解。(方案A)
    - 当供应商数量较多(>4)时，启动两阶段优化，在效率和效果间取得平衡。(方案C)
    """
    def __init__(self, container_dims: Tuple[int, int, int], cargo_data: List[dict]):
        self.container_dims = container_dims
        self.cargo_data = cargo_data
        self.all_suppliers = self._extract_suppliers(cargo_data)
        # 将报告和可视化方法从Greedy类移到这里，由顶层控制器统一管理
        self.reporter = GreedyContainerOptimizer(container_dims)

    def _extract_suppliers(self, cargo_data: List[dict]) -> List[str]:
        """从货物数据中提取不重复的供应商列表"""
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
        return suppliers_sequence

    def run_optimization(self, top_n_results: int = 3, supplier_threshold: int = 4, fast_screen_top_k: int = 5):
        """主优化流程，根据供应商数量自动选择策略"""
        from itertools import permutations
        
        all_sequences = list(permutations(self.all_suppliers))
        num_sequences = len(all_sequences)

        print(f"\n========================================================")
        print(f"  启动供应商顺序优化流程")
        print(f"  发现 {len(self.all_suppliers)} 个供应商, 将探索 {num_sequences} 种取货顺序。")
        print(f"========================================================")

        if num_sequences == 0:
            print("错误：货物数据中未发现任何供应商信息，无法执行优化。")
            return
        
        final_results = []
        # --- 混合策略决策点 (已修复) ---
        # 判断条件应为供应商数量，而非排列组合的数量
        if len(self.all_suppliers) <= supplier_threshold:
            print(f"\n模式: [方案A] 全量优化 (供应商数量 <= {supplier_threshold})")
            print(f"将对全部 {num_sequences} 种供应商顺序进行完整的多策略优化。")
            final_results = self._run_full_optimization_on_sequences(all_sequences)
        else:
            print(f"\n模式: [方案C] 两阶段优化 (供应商数量 > {supplier_threshold})")
            # --- 第一阶段：快速筛选 ---
            print(f"\n[第一阶段] 快速筛选 {num_sequences} 种顺序...")
            screened_results = self._run_fast_screening(all_sequences)
            
            # 排序并筛选出Top-K种子选手
            screened_results.sort(key=lambda x: x['rate'], reverse=True)
            top_k_sequences = [res['sequence'] for res in screened_results[:fast_screen_top_k]]
            
            print(f"\n快速筛选完成。选出 Top-{len(top_k_sequences)} 的种子选手进入下一阶段。")
            for i, res in enumerate(screened_results[:fast_screen_top_k]):
                print(f"  - 候选 {i+1}: {' -> '.join(res['sequence'])}, 潜力分: {res['rate']:.2%}")

            # --- 第二阶段：精细优化 ---
            print(f"\n[第二阶段] 对 {len(top_k_sequences)} 名种子选手进行完整多策略优化...")
            final_results = self._run_full_optimization_on_sequences(top_k_sequences)

        # --- 最终结果报告与输出 ---
        self._generate_final_reports(final_results, top_n_results)

    def _run_full_optimization_on_sequences(self, sequences_to_run: List[Tuple[str, ...]]) -> List[dict]:
        """对给定的顺序列表，并行执行完整的多策略优化"""
        tasks = []
        strategies_to_run = [
            SortingStrategy.VOLUME_DESC, SortingStrategy.AREA_DESC,
            SortingStrategy.MAX_DIM_DESC, SortingStrategy.RANDOM_SHUFFLE
        ]
        
        for seq in sequences_to_run:
            for strat in strategies_to_run:
                # 每个任务包含 (供应商顺序, 货物排序策略, 任务索引)
                tasks.append((seq, strat, len(tasks)))
        
        print(f"共创建 {len(tasks)} 个并行计算任务...")
        
        results_by_sequence = {}
        with ProcessPoolExecutor(max_workers=len(strategies_to_run) * 2) as executor:
            future_to_task = {
                executor.submit(self._run_single_strategy, task): task for task in tasks
            }
            
            for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="完整优化"):
                res = future.result()
                if not res: continue
                
                sequence = res['sequence']
                if sequence not in results_by_sequence:
                    results_by_sequence[sequence] = []
                results_by_sequence[sequence].append(res)
        
        # 从每个顺序的结果中选出最好的那个
        best_results = []
        for sequence, strategy_results in results_by_sequence.items():
            best_for_seq = max(strategy_results, key=lambda x: x['rate'])
            best_results.append(best_for_seq)
        
        return best_results

    def _run_fast_screening(self, all_sequences: List[Tuple[str, ...]]) -> List[dict]:
        """对所有顺序组合进行快速的单策略评估"""
        tasks = []
        # 固定使用最高效的体积排序策略进行快速筛选
        screening_strategy = SortingStrategy.VOLUME_DESC
        for seq in all_sequences:
            tasks.append((seq, screening_strategy, len(tasks)))

        screened_results = []
        with ProcessPoolExecutor(max_workers=len(all_sequences)) as executor:
            future_to_task = {
                executor.submit(self._run_single_strategy, task): task for task in tasks
            }
            
            for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="快速评估"):
                res = future.result()
                if res:
                    screened_results.append(res)
        return screened_results

    def _run_single_strategy(self, task: Tuple) -> dict:
        """
        [工作单元] 运行单个策略，这是所有并行任务的最小执行单元。
        :param task: 一个元组 (suppliers_sequence, sorting_strategy, task_index)
        :return: 一个包含结果的字典
        """
        suppliers_sequence, sorting_strategy, task_index = task
        
        # 每个进程拥有自己独立的优化器实例，确保无状态和线程安全
        optimizer = GreedyContainerOptimizer(self.container_dims)
        optimizer.all_cargo = optimizer._preprocess_cargo(self.cargo_data)
        optimizer.regions = optimizer._create_supplier_regions(suppliers_sequence, optimizer.all_cargo)
        optimizer.layer_strategy = LayerStrategy(optimizer.container_dims, optimizer.all_cargo)
        
        solution = optimizer._create_initial_solution(sorting_strategy, task_index)
        
        if solution and solution.placed_items:
            rate = optimizer._get_volume_ratio(solution)
            return {
                'sequence': suppliers_sequence,
                'strategy': sorting_strategy,
                'rate': rate,
                'solution': solution
            }
        return None

    def _generate_final_reports(self, final_results: List[dict], top_n: int):
        """生成最终的Top-N报告"""
        if not final_results:
            print("\n错误：未能生成任何有效的装载方案。")
            return
            
        print(f"\n\n========================================================")
        print(f"  所有计算完成，最终择优方案如下 (Top {top_n})")
        print(f"========================================================")
        
        final_results.sort(key=lambda x: x['rate'], reverse=True)
        
        solutions_to_report = final_results[:min(top_n, len(final_results))]

        for i, result in enumerate(solutions_to_report):
            rank = i + 1
            sequence_str = ' -> '.join(result['sequence'])
            rate_str = f"{result['rate']:.2%}"
            
            print(f"\n--- [方案 Top {rank}] ---")
            print(f"  - 推荐取货顺序: {sequence_str}")
            print(f"  - 最高装柜率: {rate_str} (由'{result['strategy'].value}'策略达成)")
            
            # 生成带唯一标识的文件名
            sequence_file_str = '-'.join(result['sequence'])
            file_prefix = f"方案_{rank}_顺序_{sequence_file_str}_装载率_{result['rate']:.4f}"
            excel_path = f"{file_prefix}.xlsx"
            image_path = f"{file_prefix}.png"

            print(f"  - 正在生成详细报告: {excel_path}")
            print(f"  - 正在生成3D视图: {image_path}")

            # 调用从Greedy类移过来的报告和可视化方法
            self.reporter._export_solution_to_excel(result['solution'], save_path=excel_path)
            self.reporter._create_3d_visualization(result['solution'], save_path=image_path)
            
        print("\n所有报告生成完毕。")

def load_cargo_data(file_path: str) -> List[dict]:
    try:
        excel_data = pd.read_excel(file_path)
        cargo_data = excel_data.to_dict('records')
        print("=== 真实数据加载完成 ===")
        return cargo_data
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None

def main():
    container_dims = (1180, 230, 260)
    cargo_data = load_cargo_data("装柜0538.xlsx")
    if not cargo_data: return

    total_items = sum(item['數量'] for item in cargo_data)
    total_volume_theoretical = sum(item['長度']*item['寬度']*item['高度']*item['數量'] for item in cargo_data)
    print(f"总货物种类: {len(cargo_data)}")
    print(f"总货物数量: {total_items}")
    print(f"理论总体积: {total_volume_theoretical:.2f}cm³")

    # 使用新的排列优化器作为主入口
    optimizer = PermutationOptimizer(container_dims=container_dims, cargo_data=cargo_data)
    optimizer.run_optimization()

if __name__ == "__main__":
    main()