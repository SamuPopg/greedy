import numpy as np
from typing import List, Dict, Tuple, Optional
import heapq
import pandas as pd
import time
import sys
import re  # 添加re模块用于正则表达高
from tqdm import tqdm  # 用于显示进度高

# 定义常量
CONTAINER_LENGTH = 1180  # 集装箱长高(cm)
CONTAINER_WIDTH = 230    # 集装箱宽高(cm)
CONTAINER_HEIGHT = 260   # 集装箱高高(cm)
TRANSITION_ZONE_SIZE = 20  # 交接区大高(cm)
GRID_SIZE = 1            # 空间网格大小 (cm) - 高改为1，提高精高
MIN_UTILIZATION_RATE = 0.75  # 最小空间利用率阈高

# 定义货物高
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
        """根据摆放方式返回货物的实际尺高(高 高 高"""
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
        return dims[0] * dims[1]  # 高* 高= 底面高

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

        # 初始化三维网格表示空间状高(0=高 1=已占高
        self.grid = np.zeros((
            CONTAINER_LENGTH // GRID_SIZE + 1,
            CONTAINER_WIDTH // GRID_SIZE + 1,
            CONTAINER_HEIGHT // GRID_SIZE + 1
        ), dtype=np.int8)

        self.loaded_cargos = []  # 已装载货物列高
        self.loaded_volume = 0   # 已装载体高
        self.available_spaces = [SpaceBlock(0, 0, 0, CONTAINER_LENGTH, CONTAINER_WIDTH, CONTAINER_HEIGHT)]

    def get_utilization_rate(self) -> float:
        '''计算当前装柜高'''
        return self.loaded_volume / self.volume

    def update_grid(self, cargo: Cargo, position: Tuple[int, int, int], orientation: int, value: int = 1):
        """更新网格状态，标记货物占用的空高"""
        x, y, z = position
        l, w, h = cargo.get_dimensions(orientation)

        # 将实际尺寸转换为网格单元高
        l_grid = int(np.ceil(l / GRID_SIZE))
        w_grid = int(np.ceil(w / GRID_SIZE))
        h_grid = int(np.ceil(h / GRID_SIZE))

        # 更新网格
        x_grid = x // GRID_SIZE
        y_grid = y // GRID_SIZE
        z_grid = z // GRID_SIZE

        self.grid[x_grid:x_grid+l_grid, y_grid:y_grid+w_grid, z_grid:z_grid+h_grid] = value

    def place_cargo(self, cargo: Cargo, position: Tuple[int, int, int], orientation: int) -> bool:
        """放置货物到指定位置和方向"""
        x, y, z = position
        l, w, h = cargo.get_dimensions(orientation)

        print(f"尝试放置货物: ID={cargo.id}, 位置=({position[0]}, {position[1]}, {position[2]}), " f"方向={orientation}, 尺寸={cargo.get_dimensions(orientation)}")

        # 检查是否超出集装箱边界
        if (x + l > self.length or y + w > self.width or z + h > self.height):
            print(f"  失败: 超出集装箱边高")
            return False

        # 检查是否与已有货物重叠
        if self.check_overlap(position, cargo.get_dimensions(orientation)):
            print(f"  失败: 与已有货物重高")
            return False

        # 检查是否满足摆放规高
        if not self.check_placement_rules(position, orientation, h):
            print(f"  失败: 不满足摆放规高")
            return False

        # 更新网格状高
        self.update_grid(cargo, position, orientation)

        # 更新货物信息
        cargo.position = position
        cargo.orientation = orientation

        # 更新已装载信高
        self.loaded_cargos.append(cargo)
        self.loaded_volume += cargo.volume

        # 更新可用空间
        self.update_available_spaces(position, cargo.get_dimensions(orientation))

        return True

    def check_overlap(self, position: Tuple[int, int, int], dimensions: Tuple[float, float, float]) -> bool:
        """检查是否与已有货物重叠"""
        x, y, z = position
        l, w, h = dimensions

        # 将实际尺寸转换为网格单元高
        l_grid = int(np.ceil(l / GRID_SIZE))
        w_grid = int(np.ceil(w / GRID_SIZE))
        h_grid = int(np.ceil(h / GRID_SIZE))

        # 检查网格状高
        x_grid = x // GRID_SIZE
        y_grid = y // GRID_SIZE
        z_grid = z // GRID_SIZE

        # 如果任何一个网格单元已被占用，则有重叠
        return np.any(self.grid[x_grid:x_grid+l_grid, y_grid:y_grid+w_grid, z_grid:z_grid+h_grid] > 0)

    def check_placement_rules(self, position: Tuple[int, int, int], orientation: int, height: float) -> bool:
        """检查是否满足摆放规高- 放宽规则以提高装柜率"""
        x, y, z = position

        # 规则1: 下层货物摆放方式要求躺放和侧高(保留这条基本规则)
        if z == 0 and orientation < 2:  # 立放
            return False

        # 放宽规则2和规高: 允许更多的立放情高
        if orientation < 2:  # 立放
            # 简化为只检查是否有基本支撑
            has_support = False
            for cargo in self.loaded_cargos:
                if (cargo.position[2] + cargo.get_dimensions(cargo.orientation)[2] <= z and  # 在当前位置下高
                    cargo.position[0] <= x < cargo.position[0] + cargo.get_dimensions(cargo.orientation)[0] and  # x方向有重高
                    cargo.position[1] <= y < cargo.position[1] + cargo.get_dimensions(cargo.orientation)[1]):  # y方向有重高
                    has_support = True
                    break

            if not has_support and z > 0:  # 如果不是底层且没有支撑，则不允许放置
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
            new_spaces.append(SpaceBlock(
                x + l, y, z,
                self.length - (x + l), w, h
            ))

        # 2. 左侧空间
        if x > 0:
            new_spaces.append(SpaceBlock(
                0, y, z,
                x, w, h
            ))

        # 3. 前方空间
        if y + w < self.width:
            new_spaces.append(SpaceBlock(
                x, y + w, z,
                l, self.width - (y + w), h
            ))

        # 4. 后方空间
        if y > 0:
            new_spaces.append(SpaceBlock(
                x, 0, z,
                l, y, h
            ))

        # 5. 上方空间
        if z + h < self.height:
            new_spaces.append(SpaceBlock(
                x, y, z + h,
                l, w, self.height - (z + h)
            ))

        # 6. 下方空间
        if z > 0:
            new_spaces.append(SpaceBlock(
                x, y, 0,
                l, w, z
            ))

        # 过滤有效空间并添加到可用空间列表
        for space in new_spaces:
            if space.volume > 0:
                self.available_spaces.append(space)

        # 合并相邻空间以减少碎片化
        self._merge_adjacent_spaces()

# 定义装载优化器类
class LoadingOptimizer:
    def __init__(self):
        self.container = Container()
        self.suppliers = []
        self.supplier_order = []  # 供应商装载顺高
        self.unloaded_cargos = []  # 未装载货物列高
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

        # 计算总货物数量用于进度显高
        self.progress_stats["total_cargos"] = sum(len(s.cargo_list) for s in self.supplier_order)
        print(f"总货物数高 {self.progress_stats['total_cargos']} 高")

    def optimize(self) -> Container:
        """执行装载优化"""
        # 记录开始时高
        self.progress_stats["start_time"] = time.time()
        print(f"开始装载优高- {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 1. 动态区域划高
        print("步骤 1/4: 动态区域划高")
        self.dynamic_zone_division()

        # 显示区域划分结果
        for supplier in self.supplier_order:
            print(f"  供应商{supplier.id}: {supplier.boundary_start}cm - {supplier.boundary_end}cm " f"(长度: {supplier.boundary_end - supplier.boundary_start}cm, " f"体积比例: {supplier.total_volume / sum(s.total_volume for s in self.supplier_order):.2%})")

        # 2. 区域装载优化 (逐个供应商
        print("步骤 2/4: 区域装载优化")
        for i, supplier in enumerate(self.supplier_order):
            print(f"  正在装载供应商{supplier.id} ({i+1}/{len(self.supplier_order)}) - " f"货物数量: {len(supplier.cargo_list)} 高")

            # 创建简单的进度高
            cargo_list = sorted(supplier.cargo_list, key=lambda c: c.volume, reverse=True)
            total = len(cargo_list)
            loaded_count = 0

            for idx, cargo in enumerate(cargo_list):
                # 显示进度
                if idx % 10 == 0 or idx == total - 1:
                    progress = idx / total
                    bar_length = 30
                    filled_length = int(bar_length * progress)
                    bar = '=' * filled_length + '-' * (bar_length - filled_length)
                    sys.stdout.write(f"\r  [{bar}] {idx}/{total} ({progress:.1%})")
                    sys.stdout.flush()

                # 尝试放置货物
                if self.place_cargo_in_zone(cargo, supplier):
                    loaded_count += 1
                    self.progress_stats["loaded_cargos"] += 1
                else:
                    # 如果无法放置，添加到未装载列高
                    self.unloaded_cargos.append(cargo)

            # 完成进度高
            sys.stdout.write('\n')

            # 显示该供应商装载结果
            print(f"  供应商{supplier.id} 装载完成: {loaded_count}/{len(supplier.cargo_list)} 高" f"({loaded_count/len(supplier.cargo_list):.2%})")

            # 3. 交接区优高(除第一个供应商高
            if i > 0:
                prev_supplier = self.supplier_order[i-1]
                print(f"步骤 3/4: 交接区优高(供应商{prev_supplier.id} 高{supplier.id})")

                # 记录优化前状高
                before_count = len([c for c in self.container.loaded_cargos if c.supplier_id == supplier.id])

                # 执行交接区优高
                self.transition_zone_optimization(prev_supplier, supplier)

                # 记录优化后状高
                after_count = len([c for c in self.container.loaded_cargos if c.supplier_id == supplier.id])
                print(f"  交接区优高 额外装载高{after_count - before_count} 高{supplier.id} 的货高")

        # 4. 空间利用策略 (处理剩余空间)
        print("步骤 4/4: 空间利用策略")

        # 记录优化前状高
        before_count = len(self.container.loaded_cargos)
        before_rate = self.container.get_utilization_rate()

        # 执行空间利用优化
        print(f"  尝试装载 {len(self.unloaded_cargos)} 件未装载货物...")

        # 创建简单的进度高
        total = len(self.unloaded_cargos)
        for idx, cargo in enumerate(sorted(self.unloaded_cargos, key=lambda c: c.volume)):
            # 显示进度
            if idx % 10 == 0 or idx == total - 1:
                progress = idx / total if total > 0 else 1
                bar_length = 30
                filled_length = int(bar_length * progress)
                bar = '=' * filled_length + '-' * (bar_length - filled_length)
                sys.stdout.write(f"\r  [{bar}] {idx}/{total} ({progress:.1%})")
                sys.stdout.flush()

            # 尝试放置货物
            self._try_place_cargo_anywhere(cargo)

        # 完成进度高
        sys.stdout.write('\n')

        # 记录优化后状高
        after_count = len(self.container.loaded_cargos)
        after_rate = self.container.get_utilization_rate()

        print(f"  空间利用优化: 额外装载高{after_count - before_count} 件货高")
        print(f"  装柜率提高 {before_rate:.2%} -> {after_rate:.2%} (增加 {after_rate - before_rate:.2%})")

        # 显示最终结高
        self._display_final_results()

        return self.container

    def dynamic_zone_division(self):
        """动态区域划高"""
        # 计算所有供应商的总体高
        total_volume = sum(supplier.total_volume for supplier in self.supplier_order)

        # 分配初始区域 (按体积比高
        current_position = 0
        for supplier in self.supplier_order:
            # 计算该供应商应占的长度比高
            ratio = supplier.total_volume / total_volume
            length = int(ratio * CONTAINER_LENGTH)

            # 分配区域
            supplier.boundary_start = current_position
            supplier.boundary_end = current_position + length

            # 更新当前位置
            current_position += length

        # 调整最后一个供应商的边界，确保覆盖整个集装高
        if self.supplier_order:
            self.supplier_order[-1].boundary_end = CONTAINER_LENGTH

    def place_cargo_in_zone(self, cargo: Cargo, supplier: Supplier) -> bool:
        """在指定区域内放置货物"""
        # 获取该区域内的可用空高
        zone_spaces = [
            space for space in self.container.available_spaces
            if space.x >= supplier.boundary_start and space.x + space.length <= supplier.boundary_end
        ]

        # 如果没有可用空间，返回失高
        if not zone_spaces:
            return False

        # 尝试所有可能的摆放方式和位高
        best_score = -float('inf')
        best_space = None
        best_orientation = None

        for space in zone_spaces:
            for orientation in range(6):  # 6种摆放方高
                # 检查是否符合摆放规高
                if not self.check_orientation_rules(orientation, space.z):
                    continue

                # 检查货物是否能放入空间
                if not space.can_fit(cargo, orientation):
                    continue

                # 计算评分 (接触面积 - 剩余空间体积)
                contact_area = cargo.get_contact_area(orientation)
                cargo_dims = cargo.get_dimensions(orientation)
                remaining_volume = space.volume - cargo_dims[0] * cargo_dims[1] * cargo_dims[2]

                # 计算与集装箱边界的贴近程高(越小越好)
                distance_to_boundary = min(
                    abs(space.x - supplier.boundary_start),
                    abs(space.x + cargo_dims[0] - supplier.boundary_end)
                )

                # 计算与底部的贴近程度 (越小越好)
                distance_to_bottom = space.z

                # 综合评分 (权重可调高
                score = (
                    5.0 * contact_area -  # 接触面积越大越好
                    0.5 * remaining_volume -  # 剩余空间越小越好
                    2.0 * distance_to_boundary -  # 越靠近区域边界越高
                    3.0 * distance_to_bottom  # 越靠近底部越高
                )

                if score > best_score:
                    best_score = score
                    best_space = space
                    best_orientation = orientation

        # 如果找到最佳位置，放置货物
        if best_space and best_orientation is not None:
            position = (best_space.x, best_space.y, best_space.z)
            return self.container.place_cargo(cargo, position, best_orientation)

        return False

    def check_orientation_rules(self, orientation: int, z_position: int) -> bool:
        """检查摆放方式是否符合规高"""
        # 规则1: 下层货物摆放方式要求躺放和侧高
        if z_position == 0 and orientation < 2:  # 立放
            return False

        # 规则2: 如果不是最顶层，则只可侧放和躺放，不可立放
        # 这个检查在 Container.check_placement_rules 中实高

        return True

    def transition_zone_optimization(self, prev_supplier: Supplier, curr_supplier: Supplier):
        """交接区优高"""
        # 定义交接区范高
        transition_start = prev_supplier.boundary_end - TRANSITION_ZONE_SIZE
        transition_end = prev_supplier.boundary_end + TRANSITION_ZONE_SIZE

        # 识别交接区的空隙
        gaps = self.identify_gaps(transition_start, transition_end)

        # 对当前供应商的小型货物进行排高(体积升序)
        small_cargos = sorted(
            [c for c in curr_supplier.cargo_list if c not in self.container.loaded_cargos],
            key=lambda c: c.volume
        )

        # 尝试将小型货物放入空高
        for gap in gaps:
            for cargo in small_cargos:
                # 尝试所有摆放方高
                for orientation in range(6):
                    cargo_dims = cargo.get_dimensions(orientation)

                    # 检查是否符合摆放规高
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
        # 使用可用空间列表中的空间高
        gaps = [
            space for space in self.container.available_spaces
            if space.x + space.length > start_x and space.x < end_x
        ]

        # 按体积升序排高
        gaps.sort(key=lambda g: g.volume)

        return gaps

    def space_utilization_strategy(self, pbar=None):
        """空间利用策略"""
        # 收集所有未装载的货高
        all_unloaded = list(self.unloaded_cargos)

        # 按体积升序排高
        all_unloaded.sort(key=lambda c: c.volume)

        # 尝试将未装载货物放入剩余空间
        for cargo in all_unloaded:
            # 使用_try_place_cargo_anywhere方法尝试放置货物
            if self._try_place_cargo_anywhere(cargo):
                continue

            # 如果无法放置，更新进度条
            if pbar:
                pbar.update(1)

    def _try_place_cargo_anywhere(self, cargo):
        """尝试在任何可用空间放置货高"""
        # 尝试所有可用空高
        for space in sorted(self.container.available_spaces, key=lambda s: s.volume):
            # 尝试所有摆放方高
            for orientation in range(6):
                # 放宽摆放规则，只保留基本规则
                if not self._simplified_placement_rules(space.z, orientation):
                    continue

                # 检查货物是否能放入空间
                if space.can_fit(cargo, orientation):
                    # 放置货物
                    position = (space.x, space.y, space.z)
                    if self.container.place_cargo(cargo, position, orientation):
                        # 从未装载列表中移高
                        if cargo in self.unloaded_cargos:
                            self.unloaded_cargos.remove(cargo)
                            if hasattr(self, 'progress_stats'):
                                self.progress_stats["loaded_cargos"] += 1
                        return True
        return False

    def _simplified_placement_rules(self, z_position, orientation):
        """简化的摆放规则，只保留最基本的限高"""
        # 规则1: 底层不立高
        if z_position == 0 and orientation < 2:  # 立放
            return False
        return True

    def _display_current_status(self):
        """显示当前装载状高"""
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

        # 格式化时高
        def format_time(seconds):
            hours, remainder = divmod(seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

        print(f"  当前状高 已装高{loaded_count}/{total_count} 高({loaded_count/total_count:.2%}), " f"装柜高 {self.container.get_utilization_rate():.2%}, " f"已用时间: {format_time(elapsed_time)}, " f"预计剩余: {format_time(estimated_remaining_time)}")

    def _display_final_results(self):
        """显示最终装载结高"""
        elapsed_time = time.time() - self.progress_stats["start_time"]
        loaded_count = len(self.container.loaded_cargos)
        unloaded_count = len(self.unloaded_cargos)
        total_count = loaded_count + unloaded_count

        # 格式化时高
        def format_time(seconds):
            hours, remainder = divmod(seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

        print("\n" + "="*50)
        print("装载优化完成!")
        print(f"总耗时: {format_time(elapsed_time)}")
        print(f"装柜高 {self.container.get_utilization_rate():.2%}")
        print(f"已装载货高 {loaded_count}/{total_count} 高({loaded_count/total_count:.2%})")
        print(f"未装载货高 {unloaded_count}/{total_count} 高({unloaded_count/total_count:.2%})")

        # 按供应商统计
        by_supplier = {}
        for supplier in self.supplier_order:
            loaded = len([c for c in self.container.loaded_cargos if c.supplier_id == supplier.id])
            total = len(supplier.cargo_list)
            by_supplier[supplier.id] = (loaded, total)

        print("\n供应商装载统高")
        for supplier_id, (loaded, total) in by_supplier.items():
            print(f"  {supplier_id}: {loaded}/{total} 高({loaded/total:.2%})")

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
        suppliers: 供应商列高
        supplier_cargos: 按供应商分组的货物字高
    """
    try:
        # 读取数据文件
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path, engine="openpyxl")
        else:
            df = pd.read_csv(file_path, sep='\t')  # 假设是制表符分隔的文高

        print(f"成功读取数据，共{len(df)}高")
        print("列名:", df.columns.tolist())
        print("高行数高\n", df.head())

        # 处理列名，确保列名符合预高
        expected_columns = ['貨物名稱', '數量', '長度', '寬度', '高度', '重量']
        if not all(col in df.columns for col in expected_columns):
            raise ValueError(f"数据文件缺少必要的列，需高 {expected_columns}")

        # 提取供应商信高- 修复这里的错高
        # 使用正则表达式从货物名称中提取供应商名称
        df['供应商'] = df['貨物名稱'].apply(lambda x: re.search(r'((.+))', str(x)).group(1) if re.search(r'((.+))', str(x)) else "未知")

        # 打印提取的供应商信息进行验证
        unique_suppliers = df['供应商'].unique().tolist()
        print(f"提取到的供应商 {unique_suppliers}")

        # 初始化供应商字典和货物列高
        supplier_cargos = {}
        suppliers = []

        # 按供应商分组处理数据
        for supplier_name, group in df.groupby('供应商'):
            cargo_list = []

            # 处理每个供应商的货物
            for _, row in group.iterrows():
                # 获取货物基本信息
                cargo_id = row['貨物名稱'].split('高')[0]
                quantity = int(row['數量'])
                length = float(row['長度'])
                width = float(row['寬度'])
                height = float(row['高度'])
                weight = float(row['重量']) / quantity  # 单件货物重量

                # 创建指定数量的货物对高
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

            # 创建供应商对象并添加到列高
            supplier = Supplier(id=supplier_name, cargo_list=cargo_list)
            suppliers.append(supplier)
            supplier_cargos[supplier_name] = cargo_list

            print(f"供应商{supplier_name}: {len(cargo_list)} 件货高)

        return suppliers, supplier_cargos
    except Exception as e:
        print(f"数据读取错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], {}

def run_optimization(file_path: str, supplier_order: List[str]) -> Dict:
    """
    运行装载优化算法

    Args:
        file_path: 货物数据文件路径
        supplier_order: 供应商装载顺高

    Returns:
        loading_plan: 装载方案
    """
    # 加载数据
    suppliers, _ = load_cargo_data(file_path)

    # 创建装载优化高
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
    保存装载方案到文高

    Args:
        loading_plan: 装载方案
        output_file: 输出文件路径
    """
    # 创建DataFrame存储已装载货物信高
    loaded_data = []
    for cargo in loading_plan["loaded_cargos"]:
        loaded_data.append({
            "货物ID": cargo["id"],
            "供应商: cargo["supplier_id"],
            "X坐标": cargo["position"][0],
            "Y坐标": cargo["position"][1],
            "Z坐标": cargo["position"][2],
            "摆放方式": get_orientation_name(cargo["orientation"]),
            "高: cargo["dimensions"][0],
            "高: cargo["dimensions"][1],
            "高: cargo["dimensions"][2]
        })

    loaded_df = pd.DataFrame(loaded_data)

    # 创建DataFrame存储未装载货物信高
    unloaded_data = []
    for cargo in loading_plan["unloaded_cargos"]:
        unloaded_data.append({
            "货物ID": cargo["id"],
            "供应商: cargo["supplier_id"],
            "体积": cargo["volume"]
        })

    unloaded_df = pd.DataFrame(unloaded_data)

    # 保存到Excel文件
    with pd.ExcelWriter(output_file) as writer:
        loaded_df.to_excel(writer, sheet_name="已装载货高, index=False)
        unloaded_df.to_excel(writer, sheet_name="未装载货高, index=False)

        # 添加装柜率信高
        summary_data = pd.DataFrame({
            "指标": ["装柜高, "已装载货物数高, "未装载货物数高],
            "高: [
                f"{loading_plan['utilization_rate']:.2%}",
                len(loading_plan["loaded_cargos"]),
                len(loading_plan["unloaded_cargos"])
            ]
        })
        summary_data.to_excel(writer, sheet_name="装载摘要", index=False)

def get_orientation_name(orientation_code: int) -> str:
    ""获取摆放方式的名高""
    orientations = {
        0: "立放1 (长宽底面，长高正高",
        1: "立放2 (长宽底面，宽高正高",
        2: "侧放1 (长高底面，长宽正高",
        3: "侧放2 (长高底面，高宽正高",
        4: "躺放1 (宽高底面，高长正高",
        5: "躺放2 (宽高底面，宽长正高"
    }
    return orientations.get(orientation_code, "未知")

# 主函数示高
def main():
    print("="*50)
    print("集装箱装载优化算高)
    print("="*50)

    # 数据文件路径
    cargo_data_file = "装柜0538.xlsx"  # 高.xlsx

    # 定义供应商装载顺高(根据实际需求设高
    supplier_order = ["纽蓝", "海信", "福美高]

    print(f"数据文件: {cargo_data_file}")
    print(f"供应商顺高 {supplier_order}")
    print("-"*50)

    # 加载数据
    try:
        print("正在加载数据...")
        suppliers, supplier_cargos = load_cargo_data(cargo_data_file)

        if not suppliers:
            print("错误: 无法加载供应商数据，请检查数据文件格式和路径")
            return

        # 显示数据统计
        print("数据加载完成:")
        for supplier in suppliers:
            print(f"  供应商{supplier.id}: {len(supplier.cargo_list)} 件货高 " f"总体高 {supplier.total_volume:.2f} cm³")
        print("-"*50)

        # 运行优化
        print("开始运行装载优高..")
        optimizer = LoadingOptimizer()
        optimizer.load_data(suppliers, supplier_order)
        container = optimizer.optimize()

        # 生成装载方案
        loading_plan = optimizer.generate_loading_plan()

        # 保存结果
        output_file = f"loading_plan_result_{time.strftime('%Y%m%d_%H%M%S')}.xlsx"
        print(f"正在保存结果高 {output_file}")
        save_loading_plan(loading_plan, output_file)
        print(f"结果已保存到: {output_file}")

    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
