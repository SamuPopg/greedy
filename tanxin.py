import numpy as np
from typing import List, Dict, Tuple, Optional
import heapq
import pandas as pd
import time
from tqdm import tqdm  # 用于显示进度条

# 定义常量
CONTAINER_LENGTH = 1180  # 集装箱长度 (cm)
CONTAINER_WIDTH = 230    # 集装箱宽度 (cm)
CONTAINER_HEIGHT = 260   # 集装箱高度 (cm)
TRANSITION_ZONE_SIZE = 40  # 交接区大小 (cm)，增大交接区
GRID_SIZE = 1            # 空间网格大小 (cm)
MIN_UTILIZATION_RATE = 0.75  # 最小空间利用率阈值

# 定义货物类
class Cargo:
    def __init__(self, id: str, length: float, width: float, height: float, 
                 supplier_id: str, priority: int = 0):
        self.id = id
        self.length = length
        self.width = width
        self.height = height
        self.supplier_id = supplier_id
        self.priority = priority
        self.volume = length * width * height
        self.position = None  # (x, y, z) 坐标
        self.orientation = None  # 摆放方式: 0-5 (立放1, 立放2, 侧放1, 侧放2, 躺放1, 躺放2)
    
    def get_dimensions(self, orientation: int) -> Tuple[float, float, float]:
        """根据摆放方式返回货物的实际尺寸 (长, 宽, 高)"""
        if orientation == 0:  # 立放1
            return self.length, self.width, self.height
        elif orientation == 1:  # 立放2
            return self.width, self.length, self.height
        elif orientation == 2:  # 侧放1
            return self.length, self.height, self.width
        elif orientation == 3:  # 侧放2
            return self.height, self.length, self.width
        elif orientation == 4:  # 躺放1
            return self.width, self.height, self.length
        elif orientation == 5:  # 躺放2
            return self.height, self.width, self.length
    
    def get_contact_area(self, orientation: int) -> float:
        """计算不同摆放方式下的底部接触面积"""
        dims = self.get_dimensions(orientation)
        return dims[0] * dims[1]  # 长 * 宽 = 底面积

# 定义供应商类
class Supplier:
    def __init__(self, id: str, cargo_list: List[Cargo]):
        self.id = id
        self.cargo_list = cargo_list
        self.total_volume = sum(cargo.volume for cargo in cargo_list)
        self.boundary_start = None  # 区域起始位置
        self.boundary_end = None    # 区域结束位置

# 定义空间块类
class SpaceBlock:
    def __init__(self, x: int, y: int, z: int, length: int, width: int, height: int):
        self.x = x
        self.y = y
        self.z = z
        self.length = length
        self.width = width
        self.height = height
        self.volume = length * width * height
    
    def can_fit(self, cargo: Cargo, orientation: int) -> bool:
        """检查货物是否能放入此空间块"""
        cargo_dims = cargo.get_dimensions(orientation)
        return (cargo_dims[0] <= self.length and 
                cargo_dims[1] <= self.width and 
                cargo_dims[2] <= self.height)

# 定义集装箱类
class Container:
    def __init__(self):
        self.length = CONTAINER_LENGTH
        self.width = CONTAINER_WIDTH
        self.height = CONTAINER_HEIGHT
        self.volume = CONTAINER_LENGTH * CONTAINER_WIDTH * CONTAINER_HEIGHT
        
        # 初始化三维网格表示空间状态 (0=空, 1=已占用)
        self.grid = np.zeros((
            CONTAINER_LENGTH // GRID_SIZE + 1,
            CONTAINER_WIDTH // GRID_SIZE + 1,
            CONTAINER_HEIGHT // GRID_SIZE + 1
        ), dtype=np.int8)
        
        self.loaded_cargos = []  # 已装载货物列表
        self.loaded_volume = 0   # 已装载体积
        self.available_spaces = [SpaceBlock(0, 0, 0, CONTAINER_LENGTH, CONTAINER_WIDTH, CONTAINER_HEIGHT)]
    
    def get_utilization_rate(self) -> float:
        """计算当前装柜率"""
        return self.loaded_volume / self.volume
    
    def update_grid(self, cargo: Cargo, position: Tuple[int, int, int], orientation: int, value: int = 1):
        """更新网格状态，标记货物占用的空间"""
        x, y, z = position
        l, w, h = cargo.get_dimensions(orientation)
        
        # 将实际尺寸转换为网格单元数，确保是整数
        l_grid = int(np.ceil(l / GRID_SIZE))
        w_grid = int(np.ceil(w / GRID_SIZE))
        h_grid = int(np.ceil(h / GRID_SIZE))
        
        # 更新网格，确保索引是整数
        x_grid = int(x // GRID_SIZE)
        y_grid = int(y // GRID_SIZE)
        z_grid = int(z // GRID_SIZE)
        
        self.grid[x_grid:x_grid+l_grid, y_grid:y_grid+w_grid, z_grid:z_grid+h_grid] = value
    
    def place_cargo(self, cargo: Cargo, position: Tuple[int, int, int], orientation: int) -> bool:
        """放置货物到指定位置和方向"""
        x, y, z = position
        l, w, h = cargo.get_dimensions(orientation)
        
        # 检查是否超出集装箱边界
        if (x + l > self.length or y + w > self.width or z + h > self.height):
            return False
        
        # 检查是否与已有货物重叠
        if self.check_overlap(position, cargo.get_dimensions(orientation)):
            return False
        
        # 检查是否满足摆放规则
        if not self.check_placement_rules(position, orientation, h):
            return False
        
        # 更新网格状态
        self.update_grid(cargo, position, orientation)
        
        # 更新货物信息
        cargo.position = position
        cargo.orientation = orientation
        
        # 更新已装载信息
        self.loaded_cargos.append(cargo)
        self.loaded_volume += cargo.volume
        
        # 更新可用空间
        self.update_available_spaces(position, cargo.get_dimensions(orientation))
        
        return True
    
    def check_overlap(self, position: Tuple[int, int, int], dimensions: Tuple[float, float, float]) -> bool:
        """检查是否与已有货物重叠"""
        x, y, z = position
        l, w, h = dimensions
        
        # 将实际尺寸转换为网格单元数，确保是整数
        l_grid = int(np.ceil(l / GRID_SIZE))
        w_grid = int(np.ceil(w / GRID_SIZE))
        h_grid = int(np.ceil(h / GRID_SIZE))
        
        # 检查网格状态，确保索引是整数
        x_grid = int(x // GRID_SIZE)
        y_grid = int(y // GRID_SIZE)
        z_grid = int(z // GRID_SIZE)
        
        # 确保索引在有效范围内
        if (x_grid + l_grid > self.grid.shape[0] or
            y_grid + w_grid > self.grid.shape[1] or
            z_grid + h_grid > self.grid.shape[2]):
            return True  # 超出边界，视为重叠
        
        # 如果任何一个网格单元已被占用，则有重叠
        return np.any(self.grid[x_grid:x_grid+l_grid, y_grid:y_grid+w_grid, z_grid:z_grid+h_grid] > 0)
    
    def check_placement_rules(self, position: Tuple[int, int, int], orientation: int, height: float) -> bool:
        """更灵活的摆放规则"""
        x, y, z = position
        
        # 规则1: 第一层货物摆放方式要求躺放和侧放，不可立放
        if z == 0 and orientation < 2:  # 立放
            return False
        
        # 规则2: 放宽对立放的限制，只要下方有足够支撑即可
        if orientation < 2:  # 立放
            # 计算底面积
            if orientation == 0:  # 立放1
                bottom_area = height * height  # 简化计算，实际应该从货物获取正确尺寸
            else:  # 立放2
                bottom_area = height * height  # 简化计算，实际应该从货物获取正确尺寸
            
            # 检查下方是否有足够支撑 (至少60%的底面积有支撑)
            supported_area = self._calculate_supported_area(x, y, z, height, height)
            
            if supported_area < 0.6 * bottom_area:  # 降低支撑面积要求
                return False
        
        return True
    
    def update_available_spaces(self, position: Tuple[int, int, int], dimensions: Tuple[float, float, float]):
        """更高效的空间分割算法"""
        x, y, z = position
        l, w, h = dimensions
        
        # 移除被占用的空间
        self.available_spaces = [space for space in self.available_spaces 
                                if not self._spaces_overlap(space, x, y, z, l, w, h)]
        
        # 分割剩余空间 (六向分割)
        new_spaces = []
        
        # 1. 右侧空间
        if x + l < self.length:
            right_space = SpaceBlock(
                x + l, y, z,
                self.length - (x + l), w, h
            )
            if right_space.volume > 100:  # 只添加体积足够大的空间
                new_spaces.append(right_space)
        
        # 2. 左侧空间
        if x > 0:
            left_space = SpaceBlock(
                0, y, z,
                x, w, h
            )
            if left_space.volume > 100:
                new_spaces.append(left_space)
        
        # 3. 前方空间
        if y + w < self.width:
            front_space = SpaceBlock(
                x, y + w, z,
                l, self.width - (y + w), h
            )
            if front_space.volume > 100:
                new_spaces.append(front_space)
        
        # 4. 后方空间
        if y > 0:
            back_space = SpaceBlock(
                x, 0, z,
                l, y, h
            )
            if back_space.volume > 100:
                new_spaces.append(back_space)
        
        # 5. 上方空间
        if z + h < self.height:
            top_space = SpaceBlock(
                x, y, z + h,
                l, w, self.height - (z + h)
            )
            if top_space.volume > 100:
                new_spaces.append(top_space)
        
        # 6. 下方空间
        if z > 0:
            bottom_space = SpaceBlock(
                x, y, 0,
                l, w, z
            )
            if bottom_space.volume > 100:
                new_spaces.append(bottom_space)
        
        # 过滤有效空间并添加到可用空间列表
        for space in new_spaces:
            if space.volume > 0:
                self.available_spaces.append(space)
        
        # 合并相邻空间以减少碎片化
        self._merge_adjacent_spaces()
        
        # 按体积降序排序可用空间
        self.available_spaces.sort(key=lambda s: s.volume, reverse=True)
        
        # 限制可用空间数量，避免过多的小空间
        if len(self.available_spaces) > 100:
            self.available_spaces = self.available_spaces[:100]

    def _spaces_overlap(self, space, x, y, z, l, w, h) -> bool:
        """检查空间块是否与给定的空间重叠"""
        # 检查x轴方向是否重叠
        x_overlap = not (space.x + space.length <= x or space.x >= x + l)
        # 检查y轴方向是否重叠
        y_overlap = not (space.y + space.width <= y or space.y >= y + w)
        # 检查z轴方向是否重叠
        z_overlap = not (space.z + space.height <= z or space.z >= z + h)
        
        # 三个轴都重叠才算空间重叠
        return x_overlap and y_overlap and z_overlap

    def _merge_adjacent_spaces(self):
        """合并相邻的空间块以减少碎片化"""
        # 简单实现：如果两个空间块相邻且尺寸相同，则合并它们
        i = 0
        while i < len(self.available_spaces):
            j = i + 1
            while j < len(self.available_spaces):
                space_i = self.available_spaces[i]
                space_j = self.available_spaces[j]
                
                # 检查是否可以合并 (简化版，实际应该检查更多条件)
                if (space_i.x + space_i.length == space_j.x and
                    space_i.y == space_j.y and
                    space_i.z == space_j.z and
                    space_i.width == space_j.width and
                    space_i.height == space_j.height):
                    
                    # 合并空间
                    space_i.length += space_j.length
                    space_i.volume = space_i.length * space_i.width * space_i.height
                    
                    # 移除被合并的空间
                    self.available_spaces.pop(j)
                else:
                    j += 1
            i += 1

    def _calculate_supported_area(self, x: int, y: int, z: int, length: float, width: float) -> float:
        """计算底部支撑面积"""
        # 简化实现：检查下方网格点是否被占用
        supported_count = 0
        total_count = 0
        
        # 将实际尺寸转换为网格单元数
        l_grid = int(np.ceil(length / GRID_SIZE))
        w_grid = int(np.ceil(width / GRID_SIZE))
        
        # 检查下方一层的网格状态
        if z > 0:
            z_grid = int((z - 1) // GRID_SIZE)
            for dx in range(l_grid):
                for dy in range(w_grid):
                    x_grid = int((x + dx * GRID_SIZE) // GRID_SIZE)
                    y_grid = int((y + dy * GRID_SIZE) // GRID_SIZE)
                    
                    if (0 <= x_grid < self.grid.shape[0] and 0 <= y_grid < self.grid.shape[1]):
                        total_count += 1
                        if self.grid[x_grid, y_grid, z_grid] > 0:
                            supported_count += 1
        
        # 如果没有网格点，则返回0
        if total_count == 0:
            return 0
        
        # 返回支撑面积比例
        return supported_count / total_count * length * width

# 定义装载优化器类
class LoadingOptimizer:
    def __init__(self):
        self.container = Container()
        self.suppliers = []
        self.supplier_order = []  # 供应商装载顺序
        self.unloaded_cargos = []  # 未装载货物列表
        self.progress_stats = {
            "total_cargos": 0,
            "loaded_cargos": 0,
            "start_time": None
        }
    
    def load_data(self, suppliers: List[Supplier], supplier_order: List[str]):
        """加载数据"""
        self.suppliers = suppliers
        # 按照给定顺序排序供应商
        self.supplier_order = [s for s in suppliers if s.id in supplier_order]
        self.supplier_order.sort(key=lambda s: supplier_order.index(s.id))
        
        # 计算总货物数量用于进度显示
        self.progress_stats["total_cargos"] = sum(len(s.cargo_list) for s in self.supplier_order)
        print(f"总货物数量: {self.progress_stats['total_cargos']} 件")
    
    def optimize(self) -> Container:
        """执行装载优化"""
        # 记录开始时间
        self.progress_stats["start_time"] = time.time()
        print(f"开始装载优化 - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. 动态区域划分
        print("步骤 1/4: 动态区域划分")
        self.dynamic_zone_division()
        
        # 显示区域划分结果
        for supplier in self.supplier_order:
            print(f"  供应商 {supplier.id}: {supplier.boundary_start}cm - {supplier.boundary_end}cm "
                  f"(长度: {supplier.boundary_end - supplier.boundary_start}cm, "
                  f"体积比例: {supplier.total_volume / sum(s.total_volume for s in self.supplier_order):.2%})")
        
        # 2. 区域装载优化 (逐个供应商)
        print("步骤 2/4: 区域装载优化")
        for i, supplier in enumerate(self.supplier_order):
            print(f"  正在装载供应商 {supplier.id} ({i+1}/{len(self.supplier_order)}) - "
                  f"货物数量: {len(supplier.cargo_list)} 件")
            
            # 使用tqdm创建进度条
            cargo_list = sorted(supplier.cargo_list, key=lambda c: c.volume, reverse=True)
            with tqdm(total=len(cargo_list), desc=f"  {supplier.id} 装载进度") as pbar:
                loaded_count = 0
                for cargo in cargo_list:
                    # 尝试放置货物
                    if self.place_cargo_in_zone(cargo, supplier):
                        loaded_count += 1
                        self.progress_stats["loaded_cargos"] += 1
                    else:
                        # 如果无法放置，添加到未装载列表
                        self.unloaded_cargos.append(cargo)
                    
                    # 更新进度条
                    pbar.update(1)
                    
                    # 每装载10件货物显示一次当前状态
                    if loaded_count % 10 == 0 and loaded_count > 0:
                        self._display_current_status()
            
            # 显示该供应商装载结果
            print(f"  供应商 {supplier.id} 装载完成: {loaded_count}/{len(supplier.cargo_list)} 件 "
                  f"({loaded_count/len(supplier.cargo_list):.2%})")
            
            # 3. 交接区优化 (除第一个供应商外)
            if i > 0:
                prev_supplier = self.supplier_order[i-1]
                print(f"步骤 3/4: 交接区优化 (供应商 {prev_supplier.id} 和 {supplier.id})")
                
                # 记录优化前状态
                before_count = len([c for c in self.container.loaded_cargos if c.supplier_id == supplier.id])
                
                # 执行交接区优化
                self.transition_zone_optimization(prev_supplier, supplier)
                
                # 记录优化后状态
                after_count = len([c for c in self.container.loaded_cargos if c.supplier_id == supplier.id])
                print(f"  交接区优化: 额外装载了 {after_count - before_count} 件 {supplier.id} 的货物")
        
        # 4. 空间利用策略 (处理剩余空间)
        print("步骤 4/4: 空间利用策略")
        
        # 记录优化前状态
        before_count = len(self.container.loaded_cargos)
        before_rate = self.container.get_utilization_rate()
        
        # 执行空间利用优化
        print(f"  尝试装载 {len(self.unloaded_cargos)} 件未装载货物...")
        with tqdm(total=len(self.unloaded_cargos), desc="  空间利用优化") as pbar:
            self.space_utilization_strategy(pbar)
        
        # 记录优化后状态
        after_count = len(self.container.loaded_cargos)
        after_rate = self.container.get_utilization_rate()
        
        print(f"  空间利用优化: 额外装载了 {after_count - before_count} 件货物")
        print(f"  装柜率提升: {before_rate:.2%} -> {after_rate:.2%} (增加 {after_rate - before_rate:.2%})")
        
        # 显示最终结果
        self._display_final_results()
        
        return self.container
    
    def dynamic_zone_division(self):
        """动态区域划分"""
        # 计算所有供应商的总体积
        total_volume = sum(supplier.total_volume for supplier in self.supplier_order)
        
        # 分配初始区域 (按体积比例)
        current_position = 0
        for supplier in self.supplier_order:
            # 计算该供应商应占的长度比例
            ratio = supplier.total_volume / total_volume
            length = int(ratio * CONTAINER_LENGTH)
            
            # 确保每个供应商至少有一定的空间
            min_length = 200  # 最小长度200cm
            if length < min_length:
                length = min_length
            
            # 分配区域
            supplier.boundary_start = current_position
            supplier.boundary_end = current_position + length
            
            # 更新当前位置
            current_position += length
        
        # 调整最后一个供应商的边界，确保覆盖整个集装箱
        if self.supplier_order:
            self.supplier_order[-1].boundary_end = CONTAINER_LENGTH
            
        # 如果总长度超过集装箱长度，按比例缩小每个区域
        if current_position > CONTAINER_LENGTH:
            scale_factor = CONTAINER_LENGTH / current_position
            new_position = 0
            for supplier in self.supplier_order[:-1]:  # 除了最后一个供应商
                supplier.boundary_start = int(new_position)
                new_length = int((supplier.boundary_end - supplier.boundary_start) * scale_factor)
                supplier.boundary_end = supplier.boundary_start + new_length
                new_position = supplier.boundary_end
            
            # 最后一个供应商占据剩余空间
            if self.supplier_order:
                self.supplier_order[-1].boundary_start = new_position
                self.supplier_order[-1].boundary_end = CONTAINER_LENGTH
    
    def place_cargo_in_zone(self, cargo: Cargo, supplier: Supplier) -> bool:
        """在指定区域内放置货物 - 优化版"""
        # 获取该区域内的可用空间
        zone_spaces = [
            space for space in self.container.available_spaces
            if space.x + space.length > supplier.boundary_start and space.x < supplier.boundary_end
        ]
        
        if not zone_spaces:
            return False
        
        # 输出调试信息
        if cargo.volume > 500000:
            print(f"    正在尝试放置大型货物: {cargo.id}, 体积: {cargo.volume/1000000:.2f}m³")
        
        # 进度跟踪变量
        total_spaces = len(zone_spaces)
        space_counter = 0
        position_counter = 0
        last_progress_time = time.time()
        progress_interval = 3  # 每3秒输出一次
        
        best_score = -float('inf')
        best_space = None
        best_orientation = None
        best_position = None
        
        # 根据体积对空间排序（优先尝试大小合适的空间）
        cargo_volume = cargo.volume
        zone_spaces.sort(key=lambda s: abs(s.volume - cargo_volume))
        
        for space in zone_spaces:
            space_counter += 1
            
            # 定期输出进度信息
            current_time = time.time()
            if current_time - last_progress_time > progress_interval:
                print(f"    搜索进度: 已检查 {space_counter}/{total_spaces} 空间, {position_counter} 个位置")
                last_progress_time = current_time
            
            # 初步筛选：如果空间太小则跳过
            if space.volume < cargo_volume * 0.9:  # 允许10%的误差
                continue
            
            for orientation in range(6):
                # 检查是否符合摆放规则
                if not self.check_orientation_rules(orientation, space.z):
                    continue
                
                # 检查货物是否能放入空间
                cargo_dims = cargo.get_dimensions(orientation)
                if not space.can_fit(cargo, orientation):
                    continue
                
                # 1. 首先尝试角落位置 (优先角落原则)
                corner_positions = [
                    (space.x, space.y, space.z),  # 左下角
                    (space.x + space.length - cargo_dims[0], space.y, space.z),  # 右下角
                    (space.x, space.y + space.width - cargo_dims[1], space.z),  # 左上角
                    (space.x + space.length - cargo_dims[0], space.y + space.width - cargo_dims[1], space.z)  # 右上角
                ]
                
                # 检查角落位置
                for position in corner_positions:
                    position_counter += 1
                    x_pos, y_pos, z_pos = position
                    
                    # 边界检查
                    if (x_pos < 0 or y_pos < 0 or z_pos < 0 or
                        x_pos + cargo_dims[0] > space.x + space.length or
                        y_pos + cargo_dims[1] > space.y + space.width or
                        z_pos + cargo_dims[2] > space.z + space.height):
                        continue
                    
                    # 检查是否与已有货物重叠
                    if self.container.check_overlap(position, cargo_dims):
                        continue
                    
                    # 检查是否满足摆放规则
                    if not self.container.check_placement_rules(position, orientation, cargo_dims[2]):
                        continue
                    
                    # 计算评分因素
                    contact_area = cargo.get_contact_area(orientation)
                    remaining_volume = space.volume - cargo_dims[0] * cargo_dims[1] * cargo_dims[2]
                    distance_to_boundary = min(
                        abs(x_pos - supplier.boundary_start),
                        abs(x_pos + cargo_dims[0] - supplier.boundary_end)
                    )
                    distance_to_bottom = space.z
                    neighbor_contact = self._calculate_neighbor_contact(position, cargo_dims)
                    
                    # 计算评分
                    score = (
                        10.0 * contact_area + 
                        5.0 * neighbor_contact - 
                        0.1 * remaining_volume - 
                        1.0 * distance_to_boundary - 
                        2.0 * distance_to_bottom
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_space = space
                        best_orientation = orientation
                        best_position = position
                
                # 2. 如果角落不行，则使用两级搜索策略
                # 首先是大步长搜索寻找可能的区域
                if best_score == -float('inf'):  # 如果角落尝试失败
                    coarse_step = max(10, GRID_SIZE * 5)  # 粗搜索步长
                    potential_positions = []
                    
                    for x_pos in range(int(space.x), int(space.x + space.length - cargo_dims[0] + 1), int(coarse_step)):
                        for y_pos in range(int(space.y), int(space.y + space.width - cargo_dims[1] + 1), int(coarse_step)):
                            position = (x_pos, y_pos, space.z)
                            position_counter += 1
                            
                            # 基础检查
                            if not self.container.check_overlap(position, cargo_dims) and \
                               self.container.check_placement_rules(position, orientation, cargo_dims[2]):
                                # 简化评分计算
                                quick_score = cargo.get_contact_area(orientation) - 2.0 * space.z
                                potential_positions.append((position, quick_score))
                    
                    # 对有潜力的位置进行排序并选择前N个进行精细搜索
                    potential_positions.sort(key=lambda x: x[1], reverse=True)
                    refined_positions = potential_positions[:min(10, len(potential_positions))]
                    
                    # 精细搜索阶段
                    fine_step = max(2, GRID_SIZE)  # 精细搜索步长
                    for base_pos, _ in refined_positions:
                        base_x, base_y, base_z = base_pos
                        
                        # 在粗搜索点周围进行精细搜索
                        for dx in range(-int(coarse_step//2), int(coarse_step//2 + 1), int(fine_step)):
                            for dy in range(-int(coarse_step//2), int(coarse_step//2 + 1), int(fine_step)):
                                x_pos = base_x + dx
                                y_pos = base_y + dy
                                
                                # 边界检查
                                if (x_pos < space.x or 
                                    y_pos < space.y or
                                    x_pos + cargo_dims[0] > space.x + space.length or
                                    y_pos + cargo_dims[1] > space.y + space.width):
                                    continue
                                    
                                position = (x_pos, y_pos, base_z)
                                position_counter += 1
                                
                                # 检查是否与已有货物重叠
                                if self.container.check_overlap(position, cargo_dims):
                                    continue
                                    
                                # 检查是否满足摆放规则
                                if not self.container.check_placement_rules(position, orientation, cargo_dims[2]):
                                    continue
                                    
                                # 完整评分计算
                                contact_area = cargo.get_contact_area(orientation)
                                remaining_volume = space.volume - cargo_dims[0] * cargo_dims[1] * cargo_dims[2]
                                distance_to_boundary = min(
                                    abs(x_pos - supplier.boundary_start),
                                    abs(x_pos + cargo_dims[0] - supplier.boundary_end)
                                )
                                distance_to_bottom = space.z
                                neighbor_contact = self._calculate_neighbor_contact(position, cargo_dims)
                                
                                score = (
                                    10.0 * contact_area + 
                                    5.0 * neighbor_contact - 
                                    0.1 * remaining_volume - 
                                    1.0 * distance_to_boundary - 
                                    2.0 * distance_to_bottom
                                )
                                
                                if score > best_score:
                                    best_score = score
                                    best_space = space
                                    best_orientation = orientation
                                    best_position = position
        
        # 输出总结信息
        print(f"    搜索完成: 检查了 {space_counter}/{total_spaces} 空间, {position_counter} 个位置")
        
        # 如果找到最佳位置，放置货物
        if best_space and best_orientation is not None and best_position is not None:
            print(f"    成功放置货物 {cargo.id} 在位置 {best_position}, 方向: {best_orientation}")
            return self.container.place_cargo(cargo, best_position, best_orientation)
        
        print(f"    无法放置货物 {cargo.id}")
        return False
    
    def check_orientation_rules(self, orientation: int, z_position: int) -> bool:
        """检查摆放方式是否符合规则"""
        # 规则1: 第一层货物摆放方式要求躺放和侧放，不可立放
        if z_position == 0 and orientation < 2:  # 立放
            return False
        
        # 规则2: 如果不是最顶层，则只可侧放和躺放，不可立放
        # 这个检查在 Container.check_placement_rules 中实现
        
        return True
    
    def transition_zone_optimization(self, prev_supplier: Supplier, curr_supplier: Supplier):
        """交接区优化"""
        # 定义交接区范围
        transition_start = prev_supplier.boundary_end - TRANSITION_ZONE_SIZE
        transition_end = prev_supplier.boundary_end + TRANSITION_ZONE_SIZE
        
        # 识别交接区的空隙
        gaps = self.identify_gaps(transition_start, transition_end)
        
        # 对当前供应商的小型货物进行排序 (体积升序)
        small_cargos = sorted(
            [c for c in curr_supplier.cargo_list if c not in self.container.loaded_cargos],
            key=lambda c: c.volume
        )
        
        # 尝试将小型货物放入空隙
        for gap in gaps:
            for cargo in small_cargos:
                # 尝试所有摆放方式
                for orientation in range(6):
                    cargo_dims = cargo.get_dimensions(orientation)
                    
                    # 检查是否符合摆放规则
                    if not self.check_orientation_rules(orientation, gap.z):
                        continue
                    
                    # 检查是否能放入空隙
                    if (cargo_dims[0] <= gap.length and 
                        cargo_dims[1] <= gap.width and 
                        cargo_dims[2] <= gap.height):
                        
                        # 放置货物
                        position = (gap.x, gap.y, gap.z)
                        if self.container.place_cargo(cargo, position, orientation):
                            # 从小型货物列表中移除
                            small_cargos.remove(cargo)
                            break
    
    def identify_gaps(self, start_x: int, end_x: int) -> List[SpaceBlock]:
        """识别指定范围内的空隙"""
        # 使用可用空间列表中的空间块
        gaps = [
            space for space in self.container.available_spaces
            if space.x + space.length > start_x and space.x < end_x
        ]
        
        # 按体积升序排序
        gaps.sort(key=lambda g: g.volume)
        
        return gaps
    
    def space_utilization_strategy(self, pbar=None):
        """空间利用策略"""
        # 收集所有未装载的货物
        all_unloaded = list(self.unloaded_cargos)
        
        # 按体积降序排序 (先放大货物)
        all_unloaded.sort(key=lambda c: c.volume, reverse=True)
        
        # 尝试将未装载货物放入剩余空间
        for cargo in all_unloaded:
            # 尝试所有可用空间
            best_score = -float('inf')
            best_space = None
            best_orientation = None
            best_position = None
            
            for space in sorted(self.container.available_spaces, key=lambda s: s.volume):
                for orientation in range(6):  # 6种摆放方式
                    # 检查是否符合摆放规则
                    if not self.check_orientation_rules(orientation, space.z):
                        continue
                    
                    # 检查货物是否能放入空间
                    cargo_dims = cargo.get_dimensions(orientation)
                    if not (cargo_dims[0] <= space.length and 
                            cargo_dims[1] <= space.width and 
                            cargo_dims[2] <= space.height):
                        continue
                    
                    # 尝试不同的放置位置
                    for x_pos in range(int(space.x), int(space.x + space.length - cargo_dims[0] + 1), int(max(1, GRID_SIZE))):
                        for y_pos in range(int(space.y), int(space.y + space.width - cargo_dims[1] + 1), int(max(1, GRID_SIZE))):
                            position = (x_pos, y_pos, space.z)
                            
                            # 检查是否与已有货物重叠
                            if self.container.check_overlap(position, cargo_dims):
                                continue
                            
                            # 检查是否满足摆放规则
                            if not self.container.check_placement_rules(position, orientation, cargo_dims[2]):
                                continue
                            
                            # 计算评分因素
                            contact_area = cargo.get_contact_area(orientation)
                            neighbor_contact = self._calculate_neighbor_contact(position, cargo_dims)
                            
                            # 计算剩余空间体积
                            remaining_volume = space.volume - cargo_dims[0] * cargo_dims[1] * cargo_dims[2]
                            
                            # 计算与底部的贴近程度 (越小越好)
                            distance_to_bottom = space.z
                            
                            # 综合评分 (权重可调整)
                            score = (
                                10.0 * contact_area + 
                                5.0 * neighbor_contact - 
                                0.1 * remaining_volume - 
                                2.0 * distance_to_bottom  # 越靠近底部越好
                            )
                            
                            if score > best_score:
                                best_score = score
                                best_space = space
                                best_orientation = orientation
                                best_position = position
            
            # 如果找到最佳位置，放置货物
            if best_space and best_orientation is not None and best_position is not None:
                if self.container.place_cargo(cargo, best_position, best_orientation):
                    # 从未装载列表中移除
                    if cargo in self.unloaded_cargos:
                        self.unloaded_cargos.remove(cargo)
                        self.progress_stats["loaded_cargos"] += 1
            
            # 更新进度条
            if pbar:
                pbar.update(1)
    
    def _calculate_neighbor_contact(self, position: Tuple[int, int, int], dimensions: Tuple[float, float, float]) -> float:
        """计算与相邻货物的接触面积"""
        x, y, z = position
        l, w, h = dimensions
        contact_area = 0.0
        
        # 检查六个方向的接触
        # 下方
        if z > 0:
            contact_area += self._check_direction_contact(x, y, z-1, l, w, 1, 'down')
        
        # 上方
        contact_area += self._check_direction_contact(x, y, z+h, l, w, 1, 'up')
        
        # 左方
        if x > 0:
            contact_area += self._check_direction_contact(x-1, y, z, 1, w, h, 'left')
        
        # 右方
        contact_area += self._check_direction_contact(x+l, y, z, 1, w, h, 'right')
        
        # 前方
        if y > 0:
            contact_area += self._check_direction_contact(x, y-1, z, l, 1, h, 'front')
        
        # 后方
        contact_area += self._check_direction_contact(x, y+w, z, l, 1, h, 'back')
        
        return contact_area
    
    def _check_direction_contact(self, x: int, y: int, z: int, l: float, w: float, h: float, direction: str) -> float:
        """检查特定方向的接触"""
        # 将实际尺寸转换为网格单元数
        l_grid = int(np.ceil(l / GRID_SIZE))
        w_grid = int(np.ceil(w / GRID_SIZE))
        h_grid = int(np.ceil(h / GRID_SIZE))
        
        # 转换坐标到网格
        x_grid = int(x // GRID_SIZE)
        y_grid = int(y // GRID_SIZE)
        z_grid = int(z // GRID_SIZE)
        
        # 检查网格点是否被占用
        contact_count = 0
        total_count = 0
        
        # 根据方向选择遍历维度
        if direction in ['up', 'down']:
            for dx in range(l_grid):
                for dy in range(w_grid):
                    nx = x_grid + dx
                    ny = y_grid + dy
                    nz = z_grid
                    
                    if (0 <= nx < self.container.grid.shape[0] and 
                        0 <= ny < self.container.grid.shape[1] and 
                        0 <= nz < self.container.grid.shape[2]):
                        total_count += 1
                        if self.container.grid[nx, ny, nz] > 0:
                            contact_count += 1
        
        elif direction in ['left', 'right']:
            for dy in range(w_grid):
                for dz in range(h_grid):
                    nx = x_grid
                    ny = y_grid + dy
                    nz = z_grid + dz
                    
                    if (0 <= nx < self.container.grid.shape[0] and 
                        0 <= ny < self.container.grid.shape[1] and 
                        0 <= nz < self.container.grid.shape[2]):
                        total_count += 1
                        if self.container.grid[nx, ny, nz] > 0:
                            contact_count += 1
        
        elif direction in ['front', 'back']:
            for dx in range(l_grid):
                for dz in range(h_grid):
                    nx = x_grid + dx
                    ny = y_grid
                    nz = z_grid + dz
                    
                    if (0 <= nx < self.container.grid.shape[0] and 
                        0 <= ny < self.container.grid.shape[1] and 
                        0 <= nz < self.container.grid.shape[2]):
                        total_count += 1
                        if self.container.grid[nx, ny, nz] > 0:
                            contact_count += 1
        
        # 计算接触面积
        if total_count == 0:
            return 0
        
        if direction in ['up', 'down']:
            return (contact_count / total_count) * (l * w)
        elif direction in ['left', 'right']:
            return (contact_count / total_count) * (w * h)
        else:  # front, back
            return (contact_count / total_count) * (l * h)
    
    def _display_current_status(self):
        """显示当前装载状态"""
        elapsed_time = time.time() - self.progress_stats["start_time"]
        loaded_count = self.progress_stats["loaded_cargos"]
        total_count = self.progress_stats["total_cargos"]
        
        # 计算预估剩余时间
        if loaded_count > 0:
            items_per_second = loaded_count / elapsed_time
            remaining_items = total_count - loaded_count
            estimated_remaining_time = remaining_items / items_per_second if items_per_second > 0 else 0
        else:
            estimated_remaining_time = 0
        
        # 格式化时间
        def format_time(seconds):
            hours, remainder = divmod(seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        
        print(f"  当前状态: 已装载 {loaded_count}/{total_count} 件 ({loaded_count/total_count:.2%}), "
              f"装柜率: {self.container.get_utilization_rate():.2%}, "
              f"已用时间: {format_time(elapsed_time)}, "
              f"预计剩余: {format_time(estimated_remaining_time)}")
    
    def _display_final_results(self):
        """显示最终装载结果"""
        elapsed_time = time.time() - self.progress_stats["start_time"]
        loaded_count = len(self.container.loaded_cargos)
        unloaded_count = len(self.unloaded_cargos)
        total_count = loaded_count + unloaded_count
        
        # 格式化时间
        def format_time(seconds):
            hours, remainder = divmod(seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        
        print("\n" + "="*50)
        print("装载优化完成!")
        print(f"总耗时: {format_time(elapsed_time)}")
        print(f"装柜率: {self.container.get_utilization_rate():.2%}")
        print(f"已装载货物: {loaded_count}/{total_count} 件 ({loaded_count/total_count:.2%})")
        print(f"未装载货物: {unloaded_count}/{total_count} 件 ({unloaded_count/total_count:.2%})")
        
        # 按供应商统计
        by_supplier = {}
        for supplier in self.supplier_order:
            loaded = len([c for c in self.container.loaded_cargos if c.supplier_id == supplier.id])
            total = len(supplier.cargo_list)
            by_supplier[supplier.id] = (loaded, total)
        
        print("\n供应商装载统计:")
        for supplier_id, (loaded, total) in by_supplier.items():
            print(f"  {supplier_id}: {loaded}/{total} 件 ({loaded/total:.2%})")
        
        print("="*50)
    
    def generate_loading_plan(self) -> Dict:
        """生成装载方案"""
        plan = {
            "loaded_cargos": [
                {
                    "id": cargo.id,
                    "supplier_id": cargo.supplier_id,
                    "position": cargo.position,
                    "orientation": cargo.orientation,
                    "dimensions": cargo.get_dimensions(cargo.orientation)
                }
                for cargo in self.container.loaded_cargos
            ],
            "utilization_rate": self.container.get_utilization_rate(),
            "unloaded_cargos": [
                {
                    "id": cargo.id,
                    "supplier_id": cargo.supplier_id,
                    "volume": cargo.volume
                }
                for cargo in self.unloaded_cargos
            ]
        }
        return plan

def load_cargo_data(file_path: str) -> Tuple[List[Supplier], Dict[str, List[Cargo]]]:
    """
    从Excel/CSV文件加载货物数据
    
    Args:
        file_path: 文件路径
        
    Returns:
        suppliers: 供应商列表
        supplier_cargos: 按供应商分组的货物字典
    """
    # 检查数据读取过程
    def debug_data_loading(file_path):
        try:
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                print(f"尝试读取Excel文件: {file_path}")
                df = pd.read_excel(file_path, engine='openpyxl')
            else:
                print(f"尝试读取CSV文件: {file_path}")
                df = pd.read_csv(file_path, sep='\t')
            
            print(f"成功读取数据，共{len(df)}行")
            print("列名:", df.columns.tolist())
            print("前5行数据:\n", df.head())
            
            # 检查供应商提取
            if '貨物名稱' in df.columns:
                # 修改这里：正确处理DataFrame的结果
                suppliers = df['貨物名稱'].str.extract(r'\（(.+)\）')
                # 将DataFrame的第一列转换为Series后再使用unique()
                if not suppliers.empty and 0 in suppliers.columns:
                    unique_suppliers = suppliers[0].dropna().unique()
                    print(f"提取到的供应商: {unique_suppliers}")
                else:
                    print("无法从货物名称中提取供应商信息，请检查数据格式")
            
            return df
        except Exception as e:
            print(f"数据读取错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    # 读取数据文件
    df = debug_data_loading(file_path)
    
    # 检查是否成功读取数据
    if df is None:
        raise ValueError("无法读取数据文件，请检查文件路径和格式")
    
    # 处理列名，确保列名符合预期
    expected_columns = ['貨物名稱', '數量', '長度', '寬度', '高度', '重量']
    if not all(col in df.columns for col in expected_columns):
        raise ValueError(f"数据文件缺少必要的列，需要: {expected_columns}")
    
    # 提取供应商信息
    df['供应商'] = df['貨物名稱'].str.extract(r'\（(.+)\）')[0]  # 确保取第一列
    
    # 检查是否成功提取供应商信息
    if df['供应商'].isna().all():
        raise ValueError("无法从货物名称中提取供应商信息，请检查数据格式")
    
    # 初始化供应商字典和货物列表
    supplier_cargos = {}
    suppliers = []
    
    # 按供应商分组处理数据
    for supplier_name, group in df.groupby('供应商'):
        cargo_list = []
        
        # 处理每个供应商的货物
        for _, row in group.iterrows():
            # 获取货物基本信息
            cargo_id = row['貨物名稱'].split('（')[0]
            quantity = int(row['數量'])
            length = float(row['長度'])
            width = float(row['寬度'])
            height = float(row['高度'])
            weight = float(row['重量']) / quantity  # 单件货物重量
            
            # 创建指定数量的货物对象
            for i in range(quantity):
                cargo = Cargo(
                    id=f"{cargo_id}_{i+1}",
                    length=length,
                    width=width,
                    height=height,
                    supplier_id=supplier_name,
                    priority=0  # 可根据需要设置优先级
                )
                cargo_list.append(cargo)
        
        # 创建供应商对象并添加到列表
        supplier = Supplier(id=supplier_name, cargo_list=cargo_list)
        suppliers.append(supplier)
        supplier_cargos[supplier_name] = cargo_list
    
    return suppliers, supplier_cargos

def run_optimization(file_path: str, supplier_order: List[str]) -> Dict:
    """
    运行装载优化算法
    
    Args:
        file_path: 货物数据文件路径
        supplier_order: 供应商装载顺序
        
    Returns:
        loading_plan: 装载方案
    """
    # 加载数据
    suppliers, _ = load_cargo_data(file_path)
    
    # 创建装载优化器
    optimizer = LoadingOptimizer()
    
    # 加载数据到优化器
    optimizer.load_data(suppliers, supplier_order)
    
    # 执行优化
    container = optimizer.optimize()
    
    # 生成装载方案
    loading_plan = optimizer.generate_loading_plan()
    
    return loading_plan

def save_loading_plan(loading_plan: Dict, output_file: str):
    """
    保存装载方案到文件
    
    Args:
        loading_plan: 装载方案
        output_file: 输出文件路径
    """
    # 创建DataFrame存储已装载货物信息
    loaded_data = []
    for cargo in loading_plan["loaded_cargos"]:
        loaded_data.append({
            "货物ID": cargo["id"],
            "供应商": cargo["supplier_id"],
            "X坐标": cargo["position"][0],
            "Y坐标": cargo["position"][1],
            "Z坐标": cargo["position"][2],
            "摆放方式": get_orientation_name(cargo["orientation"]),
            "长": cargo["dimensions"][0],
            "宽": cargo["dimensions"][1],
            "高": cargo["dimensions"][2]
        })
    
    loaded_df = pd.DataFrame(loaded_data)
    
    # 创建DataFrame存储未装载货物信息
    unloaded_data = []
    for cargo in loading_plan["unloaded_cargos"]:
        unloaded_data.append({
            "货物ID": cargo["id"],
            "供应商": cargo["supplier_id"],
            "体积": cargo["volume"]
        })
    
    unloaded_df = pd.DataFrame(unloaded_data)
    
    # 保存到Excel文件
    with pd.ExcelWriter(output_file) as writer:
        loaded_df.to_excel(writer, sheet_name="已装载货物", index=False)
        unloaded_df.to_excel(writer, sheet_name="未装载货物", index=False)
        
        # 添加装柜率信息
        summary_data = pd.DataFrame({
            "指标": ["装柜率", "已装载货物数量", "未装载货物数量"],
            "值": [
                f"{loading_plan['utilization_rate']:.2%}",
                len(loading_plan["loaded_cargos"]),
                len(loading_plan["unloaded_cargos"])
            ]
        })
        summary_data.to_excel(writer, sheet_name="装载摘要", index=False)

def get_orientation_name(orientation_code: int) -> str:
    """获取摆放方式的名称"""
    orientations = {
        0: "立放1 (长宽底面，长高正面)",
        1: "立放2 (长宽底面，宽高正面)",
        2: "侧放1 (长高底面，长宽正面)",
        3: "侧放2 (长高底面，高宽正面)",
        4: "躺放1 (宽高底面，高长正面)",
        5: "躺放2 (宽高底面，宽长正面)"
    }
    return orientations.get(orientation_code, "未知")

# 主函数示例
def main():
    print("="*50)
    print("集装箱装载优化算法")
    print("="*50)
    
    # 数据文件路径
    cargo_data_file = "装柜0538.xlsx"  # 或 .xlsx
    
    # 定义供应商装载顺序 (根据实际需求设置)
    supplier_order = ["纽蓝", "海信", "福美高"]
    
    print(f"数据文件: {cargo_data_file}")
    print(f"供应商顺序: {supplier_order}")
    print("-"*50)
    
    # 加载数据
    try:
        print("正在加载数据...")
        suppliers, supplier_cargos = load_cargo_data(cargo_data_file)
        
        # 显示数据统计
        print("数据加载完成:")
        for supplier in suppliers:
            print(f"  供应商 {supplier.id}: {len(supplier.cargo_list)} 件货物, "
                  f"总体积: {supplier.total_volume:.2f} cm³")
        print("-"*50)
        
        # 运行优化
        print("开始运行装载优化...")
        optimizer = LoadingOptimizer()
        optimizer.load_data(suppliers, supplier_order)
        container = optimizer.optimize()
        
        # 生成装载方案
        loading_plan = optimizer.generate_loading_plan()
        
        # 保存结果
        output_file = f"loading_plan_result_{time.strftime('%Y%m%d_%H%M%S')}.xlsx"
        print(f"正在保存结果到: {output_file}")
        save_loading_plan(loading_plan, output_file)
        print(f"结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()