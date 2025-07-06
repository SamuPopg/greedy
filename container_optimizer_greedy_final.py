#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集装箱装载优化器 - 基于供应商顺序的分区装载策略
实现第一优先级模块：区域规划、垂直分层策略、基础空间管理
"""

import math
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time


class PlacementType(Enum):
    """货物摆放方式枚举"""
    STANDING_1 = "立放1"  # 宽度*长度的面为底面，长度*高度的面为正面
    STANDING_2 = "立放2"  # 宽度*长度的面为底面，宽度*高度的面为正面
    SIDE_1 = "侧放1"      # 高度*长度的面为底面，长度*宽度的面为正面
    SIDE_2 = "侧放2"      # 高度*长度的面为底面，高度*宽度的面为正面
    LYING_1 = "躺放1"     # 宽度*高度的面为底面，高度*长度的面为正面
    LYING_2 = "躺放2"     # 宽度*高度的面为底面，宽度*长度的面为正面


@dataclass
class Cargo:
    """货物信息类"""
    cargoId: str           # 货物ID
    supplier: str          # 供应商
    length: float          # 长度(cm)
    width: float           # 宽度(cm)  
    height: float          # 高度(cm)
    weight: float          # 重量(kg)
    quantity: int          # 数量
    
    @property
    def volume(self) -> float:
        """计算货物体积"""
        return self.length * self.width * self.height
    
    @property
    def availablePlacements(self) -> List[PlacementType]:
        """返回所有可用的摆放方式"""
        return list(PlacementType)

    def getDimensionsForPlacement(self, placementType: PlacementType) -> Tuple[float, float, float]:
        """根据摆放方式获取货物在空间中的尺寸(长,宽,高)"""
        try:
            if placementType == PlacementType.STANDING_1:
                # 立放1：长×宽×高（原始尺寸）
                return (self.length, self.width, self.height)
            elif placementType == PlacementType.STANDING_2:
                # 立放2：宽×长×高（长宽交换）
                return (self.width, self.length, self.height)
            elif placementType == PlacementType.SIDE_1:
                # 侧放1：长×高×宽（高度变成宽度方向）
                return (self.length, self.height, self.width)
            elif placementType == PlacementType.SIDE_2:
                # 侧放2：高×长×宽（长度变成宽度方向）
                return (self.height, self.length, self.width)
            elif placementType == PlacementType.LYING_1:
                # 躺放1：宽×高×长（长度变成高度）
                return (self.width, self.height, self.length)
            elif placementType == PlacementType.LYING_2:
                # 躺放2：高×宽×长（长度变成高度，长宽交换）
                return (self.height, self.width, self.length)
            else:
                raise ValueError(f"不支持的摆放方式: {placementType}")
        except Exception as e:
            print(f"计算货物尺寸时发生错误: {e}")
            return (self.length, self.width, self.height)


@dataclass(frozen=True)
class Position:
    """三维位置信息"""
    x: float  # X坐标(cm)
    y: float  # Y坐标(cm)
    z: float  # Z坐标(cm)


@dataclass
class LoadedItem:
    """用于记录已放置货物信息的数据类。"""
    cargo: Cargo
    position: Position
    placement_type: PlacementType


@dataclass
class Region:
    """定义一个集装箱内的物理区域"""
    startPosition: float
    endPosition: float
    width: float
    height: float
    supplier: str

    @property
    def volume(self) -> float:
        """计算区域体积"""
        return (self.endPosition - self.startPosition) * self.width * self.height

    def __repr__(self) -> str:
        return f"区域(供应商: {self.supplier}, X: {self.startPosition:.1f}-{self.endPosition:.1f}, 体积: {self.volume:.2f}cm³)"


@dataclass
class LayerInfo:
    """分层信息"""
    layerIndex: int        # 层索引(0为底层)
    startHeight: float     # 层起始高度
    endHeight: float       # 层结束高度
    
    @property
    def height(self) -> float:
        """计算层高度"""
        return self.endHeight - self.startHeight


class RegionPlanner:
    """区域规划模块 - 负责根据供应商顺序和货物体积分配区域"""
    
    def __init__(self, containerDimensions: Tuple[float, float, float]):
        """
        初始化区域规划器
        
        Args:
            containerDimensions: 集装箱尺寸(长,宽,高)
        """
        self.containerLength = containerDimensions[0]
        self.containerWidth = containerDimensions[1]
        self.containerHeight = containerDimensions[2]
        self.interactionZoneRatio = 0.05  # 交接区预留比例
        
    def calculateSupplierVolumes(self, supplierSequence: List[str], cargoData: Dict[str, List[Cargo]]) -> Dict[str, float]:
        """
        计算各供应商的货物总体积
        
        Args:
            supplierSequence: 供应商访问顺序
            cargoData: 货物数据，按供应商分组
            
        Returns:
            各供应商的总体积字典
        """
        try:
            supplierVolumes = {}
            for supplier in supplierSequence:
                if supplier in cargoData:
                    totalVolume = 0
                    for cargo in cargoData[supplier]:
                        totalVolume += cargo.volume * cargo.quantity
                    supplierVolumes[supplier] = totalVolume
                else:
                    print(f"警告: 供应商 {supplier} 没有货物数据")
                    supplierVolumes[supplier] = 0
            return supplierVolumes
        except Exception as e:
            print(f"计算供应商体积时发生错误: {e}")
            return {}
    
    def calculateRegionBoundaries(self, supplierSequence: List[str], cargoData: Dict[str, List[Cargo]]) -> List[Region]:
        """
        计算各供应商区域边界
        
        Args:
            supplierSequence: 供应商访问顺序
            cargoData: 货物数据，按供应商分组
            
        Returns:
            供应商区域列表
        """
        try:
            # 1. 计算各供应商总体积
            supplierVolumes = self.calculateSupplierVolumes(supplierSequence, cargoData)
            
            # 2. 计算体积比例
            totalVolume = sum(supplierVolumes.values())
            if totalVolume == 0:
                raise ValueError("总体积为0，无法分配区域")
            
            volumeRatios = {supplier: volume / totalVolume for supplier, volume in supplierVolumes.items()}
            
            # 3. 预留交接区空间
            availableLength = self.containerLength * (1 - self.interactionZoneRatio)
            
            # 4. 按供应商顺序从内到外分配区域
            regions = []
            currentPosition = 0
            
            for supplier in supplierSequence:
                regionLength = availableLength * volumeRatios[supplier]
                
                region = Region(
                    startPosition=currentPosition,
                    endPosition=currentPosition + regionLength,
                    width=self.containerWidth,
                    height=self.containerHeight,
                    supplier=supplier
                )
                
                regions.append(region)
                currentPosition += regionLength
                
                print(f"供应商 {supplier} 分配区域: {currentPosition-regionLength:.1f}cm - {currentPosition:.1f}cm, 体积比例: {volumeRatios[supplier]:.3f}")
            
            return regions
            
        except Exception as e:
            print(f"计算区域边界时发生错误: {e}")
            return []
    
    def calculateInteractionZones(self, regions: List[Region]) -> List[Dict[str, Any]]:
        """
        计算交接区边界
        
        Args:
            regions: 供应商区域列表
            
        Returns:
            交接区信息列表
        """
        try:
            interactionZones = []
            interactionZoneWidth = self.containerLength * self.interactionZoneRatio / max(1, len(regions) - 1)
            
            for i in range(len(regions) - 1):
                zone = {
                    'supplier1': regions[i].supplier,
                    'supplier2': regions[i + 1].supplier,
                    'startPosition': regions[i].endPosition - interactionZoneWidth / 2,
                    'endPosition': regions[i + 1].startPosition + interactionZoneWidth / 2,
                    'width': interactionZoneWidth,
                    'height': self.containerHeight
                }
                interactionZones.append(zone)
                
                print(f"交接区 {zone['supplier1']}-{zone['supplier2']}: {zone['startPosition']:.1f}cm - {zone['endPosition']:.1f}cm")
            
            return interactionZones
            
        except Exception as e:
            print(f"计算交接区时发生错误: {e}")
            return []


class LayerStrategy:
    """垂直分层策略 - 负责在每个供应商区域内实现垂直分层"""
    
    def __init__(self):
        self.layers: List[LayerInfo] = []

    def redesign_layers(self, all_cargo: List[Cargo], container_height: float):
        """
        基于所有货物的尺寸分布，重新设计一个通用的、动态的三层分层策略。
        """
        # 1. 收集所有货物所有可能的非零高度
        possible_heights = []
        for cargo in all_cargo:
            dims = [cargo.length, cargo.width, cargo.height]
            possible_heights.extend([d for d in dims if d > 0])
        
        if not possible_heights:
            print("警告: 无法收集到任何货物高度信息，无法进行分层。")
            self.layers = []
            return

        # 2. 基于高度分布设计层高
        sorted_heights = sorted(list(set(possible_heights)))
        
        # 简单的策略：使用百分位来决定层高
        p25 = sorted_heights[int(len(sorted_heights) * 0.25)]
        p75 = sorted_heights[int(len(sorted_heights) * 0.75)]
        
        # 第1层高度，至少30cm，适应较小的货物
        layer1_h = max(30, p25 + 5)
        # 第2层高度，至少40cm，适应中等货物
        layer2_h = max(40, p75 + 5)
        
        # 确保总高不超过容器高度
        if layer1_h + layer2_h >= container_height:
            layer1_h = container_height * 0.4
            layer2_h = container_height * 0.6
        
        layer3_h = container_height - layer1_h - layer2_h

        self.layers = [
            LayerInfo(layerIndex=0, startHeight=0, endHeight=layer1_h),
            LayerInfo(layerIndex=1, startHeight=layer1_h, endHeight=layer1_h + layer2_h),
            LayerInfo(layerIndex=2, startHeight=layer1_h + layer2_h, endHeight=container_height)
        ]

    def get_layers(self) -> List[LayerInfo]:
        return self.layers
        
    def print_layers(self):
        print("\n=== 动态分层策略结果 ===")
        if not self.layers:
            print("  未生成任何分层。")
            return
        for layer in self.layers:
            print(f"  层 {layer.layerIndex + 1}: 高度 {layer.startHeight:.1f}cm - {layer.endHeight:.1f}cm (厚度: {layer.height:.1f}cm)")


class SpaceManager:
    """空间管理器，负责跟踪集装箱内的空间占用情况。"""
    def __init__(self, containerDimensions: Tuple[float, float, float], gridSize: float = 5.0):
        self.containerDimensions = containerDimensions
        self.gridSize = float(gridSize)
        self.gridCount = (
            int(containerDimensions[0] / self.gridSize),
            int(containerDimensions[1] / self.gridSize),
            int(containerDimensions[2] / self.gridSize)
        )
        self.occupiedCells: Set[Tuple[int, int, int]] = set()
        print(f"空间管理器初始化完成: 网格数量 {self.gridCount[0]}x{self.gridCount[1]}x{self.gridCount[2]}, 网格尺寸 {self.gridSize}cm")

    def reset(self):
        """重置空间，清空所有已占用的网格。"""
        self.occupiedCells.clear()
        print("空间管理器已重置。")

    def calculateOccupiedCells(self, position: Position, cargo: Cargo, placementType: PlacementType) -> List[Tuple[int, int, int]]:
        """计算给定货物和位置将占用的所有网格单元。"""
        try:
            dims = cargo.getDimensionsForPlacement(placementType)
            
            startX = int(position.x / self.gridSize)
            startY = int(position.y / self.gridSize)
            startZ = int(position.z / self.gridSize)
            
            endX = int((position.x + dims[0]) / self.gridSize)
            endY = int((position.y + dims[1]) / self.gridSize)
            endZ = int((position.z + dims[2]) / self.gridSize)

            cells = []
            for x in range(startX, endX):
                for y in range(startY, endY):
                    for z in range(startZ, endZ):
                        cells.append((x, y, z))
            return cells
        except Exception as e:
            # 这里的 cargo 可能是元组，也可能是 Cargo 对象，需要统一
            cargo_id = cargo.cargoId if hasattr(cargo, 'cargoId') else 'N/A'
            print(f"计算网格占用时发生错误: {e}, Cargo ID: {cargo_id}")
            return []

    def isSpaceAvailable(self, position: Position, cargo: Cargo, placement_type: PlacementType) -> bool:
        """检查给定位置是否有足够空间放置一个货物（使用精确的边界计算）。"""
        dims = cargo.getDimensionsForPlacement(placement_type)

        # 检查物理边界
        if position.x + dims[0] > self.containerDimensions[0] or \
           position.y + dims[1] > self.containerDimensions[1] or \
           position.z + dims[2] > self.containerDimensions[2]:
            return False

        # 精确的网格坐标转换
        start_x = int(position.x / self.gridSize)
        end_x = int((position.x + dims[0] - 1e-9) / self.gridSize)
        start_y = int(position.y / self.gridSize)
        end_y = int((position.y + dims[1] - 1e-9) / self.gridSize)
        start_z = int(position.z / self.gridSize)
        end_z = int((position.z + dims[2] - 1e-9) / self.gridSize)

        for x in range(start_x, end_x + 1):
            for y in range(start_y, end_y + 1):
                for z in range(start_z, end_z + 1):
                    if (x, y, z) in self.occupiedCells:
                        return False
        return True

    def markOccupied(self, position: Position, cargo: Cargo, placement_type: PlacementType):
        """将货物占据的空间标记为已占用（使用精确的边界计算）。"""
        dims = cargo.getDimensionsForPlacement(placement_type)
        
        start_x = int(position.x / self.gridSize)
        end_x = int((position.x + dims[0] - 1e-9) / self.gridSize)
        start_y = int(position.y / self.gridSize)
        end_y = int((position.y + dims[1] - 1e-9) / self.gridSize)
        start_z = int(position.z / self.gridSize)
        end_z = int((position.z + dims[2] - 1e-9) / self.gridSize)

        cells_to_occupy = set()
        for x in range(start_x, end_x + 1):
            for y in range(start_y, end_y + 1):
                for z in range(start_z, end_z + 1):
                    cells_to_occupy.add((x, y, z))
        self.occupiedCells.update(cells_to_occupy)


    def check_foundations(self, position: Position, dimensions: Tuple[float, float, float]) -> bool:
        """
        检查一个物体放置位置的地基是否稳固。
        新策略：只需要检查物体的四个底面角点下方是否有支撑即可。
        """
        x, y, z = position.x, position.y, position.z
        w, d, _ = dimensions

        # 如果物体直接放在集装箱底部，地基总是稳固的
        if z == 0:
            return True

        # 定义四个角点的x, y坐标
        corners = [
            (x, y),
            (x + w, y),
            (x, y + d),
            (x + w, y + d)
        ]

        # 检查每个角点正下方一个单位(grid_size)的网格是否被占用
        # 我们需要将物理坐标转换为网格坐标
        target_z_grid = int((z / self.gridSize) - 1)
        if target_z_grid < 0:
            return False # 已经在最底层之下，理论上不会发生

        for corner_x, corner_y in corners:
            # -1epsilon 是为了处理边界情况，确保在地板上的点能被正确计算
            gx = int((corner_x - 1e-9) / self.gridSize)
            gy = int((corner_y - 1e-9) / self.gridSize)
            
            if (gx, gy, target_z_grid) not in self.occupiedCells:
                # 只要有一个角点没有支撑，就认为地基不稳
                return False
        
        # 所有角点都有支撑
        return True


class SupplierBasedContainerOptimizer:
    """
    一个基于供应商分区和动态层策略的集装箱装载优化器。
    """
    def __init__(self, container_dims: Tuple[float, float, float], grid_size: float = 1.0):
        self.container_dims = container_dims
        self.grid_size = grid_size
        self.spaceManager = SpaceManager(container_dims, grid_size)
        self.layerStrategy = LayerStrategy()
        self.loadedItems: List[LoadedItem] = []
        self.regions: List[Region] = []

    def _add_candidate_point(self, new_point: Position, region: Region, candidate_list: List[Position]):
        """
        验证并添加一个新的候选点到列表中，同时避免重复。
        确保候选点在货柜和当前供应商区域的边界内。
        """
        # 1. 检查是否在货柜物理边界内
        if not (0 <= new_point.x < self.container_dims[0] and
                0 <= new_point.y < self.container_dims[1] and
                0 <= new_point.z < self.container_dims[2]):
            return

        # 2. 检查是否在当前供应商的X轴区域内
        if not (region.startPosition <= new_point.x < region.endPosition):
            return

        # 3. 检查是否已存在，避免重复
        if new_point not in candidate_list:
            candidate_list.append(new_point)

    def _get_allowed_placements(self, cargo: Cargo, layer: LayerInfo) -> List[PlacementType]:
        """根据当前层数，决定货物允许的摆放方式。"""
        possible_placements = list(cargo.availablePlacements)
        if layer.layerIndex < 2:  # 对应第一层和第二层
            # 移除所有立放的选项
            possible_placements = [p for p in possible_placements if p not in [PlacementType.STANDING_1, PlacementType.STANDING_2]]
        return possible_placements

    def loadCargoInLayer(self, region: Region, layer: LayerInfo, cargo_list: List[Cargo], initial_points: List[Position]) -> int:
        """
        在指定的区域和层内，使用基于"角落"的三维候选点策略尝试放置一批货物。
        这是算法的核心，它实现了动态生成和消耗候选点的逻辑。
        """
        sorted_cargo = sorted(cargo_list, key=lambda c: c.volume, reverse=True)
        loaded_count = 0

        # 1. 初始化候选点列表，使用传入的初始点，并确保默认起点存在
        candidate_points = list(set(initial_points)) # 使用set去重
        # 确保该区域在本层的默认起点始终存在
        default_start_point = Position(region.startPosition, 0, layer.startHeight)
        if default_start_point not in candidate_points:
            candidate_points.append(default_start_point)

        # 范式转移：算法的"主角"从"货物"变为"候选点"
        while candidate_points:
            # 1. 选取并移除当前最优的候选点
            candidate_points.sort(key=lambda p: (p.z, p.y, p.x))
            current_point = candidate_points.pop(0)

            best_fit_cargo = None
            best_fit_placement = None
            # 使用一个评分来记录最优选择，例如，最大化体积
            best_score = -1

            # 2. 遍历所有"未放置"且"适合本层"的货物，为这一个"坑"找到最合适的"萝卜"
            for cargo in cargo_list:
                if cargo in [item.cargo for item in self.loadedItems]:
                    continue # 跳过已放置的

                for placement_type in self._get_allowed_placements(cargo, layer):
                    dims = cargo.getDimensionsForPlacement(placement_type)
                    if current_point.z + dims[2] > layer.startHeight + layer.height:
                        continue

                    if self.spaceManager.isSpaceAvailable(current_point, cargo, placement_type) and \
                       self.spaceManager.check_foundations(current_point, dims):
                        
                        # 启发式评分: 在这个点上，优先放置可行的、体积最大的货物
                        score = cargo.volume
                        if score > best_score:
                            best_score = score
                            best_fit_cargo = cargo
                            best_fit_placement = placement_type
            
            # 4. 如果为这个点找到了最适合的货物，就放置它
            if best_fit_cargo:
                dims = best_fit_cargo.getDimensionsForPlacement(best_fit_placement)
                self.spaceManager.markOccupied(current_point, best_fit_cargo, best_fit_placement)
                self.loadedItems.append(LoadedItem(best_fit_cargo, current_point, best_fit_placement))
                print(f"      在点 {current_point} 成功放置最适货物 {best_fit_cargo.cargoId}。")
                
                # 放置成功后，生成新的候选点并加入列表
                new_point_top = Position(current_point.x, current_point.y, current_point.z + dims[2])
                new_point_x = Position(current_point.x + dims[0], current_point.y, current_point.z)
                new_point_y = Position(current_point.x, current_point.y + dims[1], current_point.z)
                
                self._add_candidate_point(new_point_top, region, candidate_points)
                self._add_candidate_point(new_point_x, region, candidate_points)
                self._add_candidate_point(new_point_y, region, candidate_points)
                
                loaded_count += 1
        
        return loaded_count

    def optimize(self, cargo_data: Dict[str, List[Cargo]], suppliers_sequence: List[str]):
        """
        主优化流程。
        """
        self.spaceManager.reset()
        self.loadedItems = []
        
        # 1. 动态区域划分
        self.regions = self.calculate_regions(cargo_data, suppliers_sequence)
        
        # 2. 动态分层策略
        all_cargo_list = [cargo for sublist in cargo_data.values() for cargo in sublist]
        self.layerStrategy.redesign_layers(all_cargo_list, self.container_dims[2])
        self.layerStrategy.print_layers()
        
        # 跨供应商传递的"完成面"候选点
        cross_supplier_candidate_points = []

        for supplier_name in suppliers_sequence:
            region = next((r for r in self.regions if r.supplier == supplier_name), None)
            if not region:
                continue

            print(f"\n--- 开始处理供应商: {supplier_name} ---")
            print(f"  区域信息: {region}")
            
            supplier_cargo = cargo_data.get(supplier_name, [])
            
            # 优化：将前一个供应商的"完成面"作为初始候选点
            initial_points_for_supplier = cross_supplier_candidate_points
            
            for layer in self.layerStrategy.get_layers():
                print(f"  正在处理第 {layer.layerIndex + 1} 层 (高度: {layer.startHeight}cm - {layer.endHeight}cm)")
                
                cargo_to_load_in_layer = [
                    c for c in supplier_cargo 
                    if c not in [item.cargo for item in self.loadedItems] and \
                    any(c.getDimensionsForPlacement(p)[2] <= layer.height for p in self._get_allowed_placements(c, layer))
                ]
                
                if not cargo_to_load_in_layer:
                    print("      本层无合适尺寸的货物可装载，跳过。")
                    continue
                
                # 修正：将继承的候选点与本层的标准起点结合，并作为参数传递
                layer_initial_points = [p for p in initial_points_for_supplier if layer.startHeight <= p.z < layer.endHeight]
                
                self.loadCargoInLayer(region, layer, cargo_to_load_in_layer, layer_initial_points)

            # 为下一个供应商生成"完成面"
            cross_supplier_candidate_points = self._generate_frontier_points(region)

    def _generate_frontier_points(self, region: Region) -> List[Position]:
        """扫描区域的结束边界，生成一个"完成面"作为下一个区域的初始候选点。"""
        frontier_points = []
        
        # 修正：通过减去一个极小值，来精确找到边界前的最后一个网格索引
        frontier_x_grid = int((region.endPosition - 1e-9) / self.grid_size)

        for gy in range(self.spaceManager.gridCount[1]):
            # 从下往上扫描，找到每个(x,y)柱子上的最高点
            max_z_in_pillar = -1
            for gz in range(self.spaceManager.gridCount[2]):
                if (frontier_x_grid, gy, gz) in self.spaceManager.occupiedCells:
                    max_z_in_pillar = gz
            
            if max_z_in_pillar != -1:
                # 如果这个柱子被占用了，就在它的最高点的顶上生成一个候选点
                # 这个新点将位于下一个区域的起点
                new_x = (frontier_x_grid + 1) * self.grid_size
                new_y = gy * self.grid_size
                new_z = (max_z_in_pillar + 1) * self.grid_size
                frontier_points.append(Position(new_x, new_y, new_z))
        
        return frontier_points

    def calculate_regions(self, cargo_data: Dict[str, List[Cargo]], suppliers_sequence: List[str]) -> List[Region]:
        total_volume = sum(c.volume for sublist in cargo_data.values() for c in sublist)
        if total_volume == 0:
            return []

        regions = []
        current_position = 0.0
        container_length = self.container_dims[0]

        for supplier in suppliers_sequence:
            supplier_volume = sum(c.volume for c in cargo_data.get(supplier, []))
            ratio = supplier_volume / total_volume
            region_length = container_length * ratio
            
            region = Region(
                startPosition=current_position,
                endPosition=current_position + region_length,
                width=self.container_dims[1],
                height=self.container_dims[2],
                supplier=supplier
            )
            regions.append(region)
            current_position += region_length
        
        return regions

    def generate_report(self):
        """
        生成最终的装载报告。
        """
        print("\n\n=== 最终装载报告 ===")
        
        container_volume = self.container_dims[0] * self.container_dims[1] * self.container_dims[2]
        
        total_loaded_volume = sum(item.cargo.volume for item in self.loadedItems)
        
        final_rate = (total_loaded_volume / container_volume * 100) if container_volume > 0 else 0
        
        print(f"最终装柜率: {final_rate:.2f}%")
        print(f"总装载体积: {total_loaded_volume:.2f} cm³ / {container_volume:.2f} cm³")
        print(f"总装载件数: {len(self.loadedItems)}")

        print("\n--- 各供应商分区利用率 ---")
        if not self.regions:
            print("    未定义供应商分区。")
        else:
            for region in self.regions:
                region_volume = (region.endPosition - region.startPosition) * self.container_dims[1] * self.container_dims[2]
                
                loaded_in_region = [item for item in self.loadedItems if item.cargo.supplier == region.supplier]
                volume_in_region = sum(item.cargo.volume for item in loaded_in_region)
                
                region_rate = (volume_in_region / region_volume * 100) if region_volume > 0 else 0
                print(f"  - 供应商: {region.supplier}")
                print(f"    - 区域空间占比: {(region_volume / container_volume * 100):.2f}%")
                print(f"    - 空间利用率: {region_rate:.2f}%")


def loadRealDataFromExcel(filePath: str) -> Tuple[List[str], Dict[str, List[Cargo]]]:
    """
    从Excel文件加载真实货物数据
    
    Args:
        filePath: Excel文件路径
        
    Returns:
        供应商顺序列表和按供应商分组的货物数据
    """
    try:
        import pandas as pd
        
        # 读取Excel文件
        df = pd.read_excel(filePath)
        
        # 提取供应商信息
        df['供应商'] = df['貨物名稱'].str.extract(r'（(.+?)）')
        
        # 统计供应商货物数量，按数量降序排列作为访问顺序
        supplierCounts = df.groupby('供应商')['數量'].sum().sort_values(ascending=False)
        supplierSequence = supplierCounts.index.tolist()
        
        # 按供应商分组数据
        cargoData = {}
        for supplier in supplierSequence:
            supplierData = df[df['供应商'] == supplier]
            cargoList = []
            
            for _, row in supplierData.iterrows():
                cargo = Cargo(
                    cargoId=row['貨物名稱'],
                    supplier=supplier,
                    length=float(row['長度']),
                    width=float(row['寬度']),
                    height=float(row['高度']),
                    weight=float(row['重量']),
                    quantity=int(row['數量'])
                )
                cargoList.append(cargo)
            
            cargoData[supplier] = cargoList
        
        # 打印数据统计
        totalQuantity = df['數量'].sum()
        totalWeight = df['重量'].sum()
        totalVolume = (df['長度'] * df['寬度'] * df['高度'] * df['數量']).sum()
        
        print(f"=== 真实数据加载完成 ===")
        print(f"总货物种类: {len(df)}")
        print(f"总货物数量: {totalQuantity}")
        print(f"总重量: {totalWeight:.2f}kg")
        print(f"总体积: {totalVolume:.2f}cm³")
        print(f"供应商数量: {len(supplierSequence)}")
        print(f"供应商顺序: {supplierSequence}")
        
        for supplier in supplierSequence:
            supplierData = df[df['供应商'] == supplier]
            supplierQuantity = supplierData['數量'].sum()
            supplierVolume = (supplierData['長度'] * supplierData['寬度'] * supplierData['高度'] * supplierData['數量']).sum()
            print(f"  {supplier}: {len(supplierData)}种货物, {supplierQuantity}件, 体积{supplierVolume:.2f}cm³")
        
        return supplierSequence, cargoData
        
    except Exception as e:
        print(f"加载真实数据时发生错误: {e}")
        return [], {}


def main():
    """主函数，执行整个优化流程"""
    startTime = time.time()
    
    # 1. 从Excel加载真实数据
    supplier_sequence, cargo_by_supplier = loadRealDataFromExcel('装柜0538.xlsx')
    
    # 2. 初始化优化器
    container_dims = (1180, 230, 260)
    grid_size = 5.0
    optimizer = SupplierBasedContainerOptimizer(container_dims, grid_size)
    
    # 3. 执行优化
    optimizer.optimize(
        cargo_data=cargo_by_supplier,
        suppliers_sequence=supplier_sequence
    )
    
    # 4. 生成报告
    optimizer.generate_report()
    
    endTime = time.time()
    print(f"\n>>> 优化总耗时: {endTime - startTime:.2f} 秒")


if __name__ == '__main__':
    main() 