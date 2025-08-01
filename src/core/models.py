"""
Core data models for the container optimization project.
"""
import copy
from dataclasses import dataclass
from enum import Enum
from typing import List, Set, Tuple

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
    """代表一个完整的装载方案（状态）"""
    def __init__(self, all_cargo: List[Cargo]):
        self.placed_items: List[PlacedItem] = []
        self.unloaded_cargo_set: Set[Cargo] = set(all_cargo)
        self.total_volume = 0.0
        self.rate = 0.0
        self.sequence: Tuple[str, ...] = ()


    def add_item(self, cargo_to_add: Cargo, position: Position, orientation: PlacementType):
        """向方案中添加一个新放置的货物"""
        new_item = PlacedItem(cargo_to_add, position, orientation)
        self.placed_items.append(new_item)
        self.total_volume += cargo_to_add.volume

        if cargo_to_add in self.unloaded_cargo_set:
            self.unloaded_cargo_set.remove(cargo_to_add)

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
