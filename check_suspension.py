import pandas as pd
import pickle
import re
from container_optimizer import SimulatedAnnealingOptimizer, load_cargo_data, LayerStrategy

def check_cargo_support(solution, container_dims):
    """检查每个货物是否有足够的支撑"""
    print("\n=== 悬空检测报告 ===")
    suspended_items = []
    
    for i, item in enumerate(solution.placed_items):
        x, y, z = item.position.x, item.position.y, item.position.z
        l, w, h = item.current_dims
        
        # 如果货物在地面上，不需要检查支撑
        if z == 0:
            continue
            
        # 检查货物底面的支撑情况
        bottom_area = l * w  # 货物底面积
        supported_area = 0  # 支撑面积
        
        # 检查所有其他货物，看是否能提供支撑
        for j, other_item in enumerate(solution.placed_items):
            if i == j:  # 跳过自己
                continue
                
            ox, oy, oz = other_item.position.x, other_item.position.y, other_item.position.z
            ol, ow, oh = other_item.current_dims
            
            # 检查other_item是否在当前货物正下方
            # 1. other_item的顶部高度必须正好等于当前货物的底部高度
            if abs((oz + oh) - z) > 0.01:  # 允许0.01cm的误差
                continue
                
            # 2. 计算X轴和Y轴的重叠区域
            x_overlap_start = max(x, ox)
            x_overlap_end = min(x + l, ox + ol)
            y_overlap_start = max(y, oy)
            y_overlap_end = min(y + w, oy + ow)
            
            if x_overlap_start < x_overlap_end and y_overlap_start < y_overlap_end:
                # 有重叠，计算重叠面积
                overlap_area = (x_overlap_end - x_overlap_start) * (y_overlap_end - y_overlap_start)
                supported_area += overlap_area
        
        # 计算支撑率
        support_ratio = supported_area / bottom_area
        
        # 如果支撑率小于50%，认为是悬空
        if support_ratio < 0.5:
            suspended_items.append({
                'item': item,
                'index': i,
                'support_ratio': support_ratio,
                'position': (x, y, z),
                'dimensions': (l, w, h),
                'supplier': item.cargo.supplier
            })
    
    # 输出悬空货物信息
    if suspended_items:
        print(f"\n发现 {len(suspended_items)} 个悬空或支撑不足的货物:")
        for info in suspended_items:
            print(f"\n货物 #{info['index']}:")
            print(f"  - 供应商: {info['supplier']}")
            print(f"  - 位置: X={info['position'][0]:.1f}, Y={info['position'][1]:.1f}, Z={info['position'][2]:.1f}")
            print(f"  - 尺寸: {info['dimensions'][0]:.1f} x {info['dimensions'][1]:.1f} x {info['dimensions'][2]:.1f}")
            print(f"  - 支撑率: {info['support_ratio']:.1%}")
    else:
        print("\n所有货物都有充分的支撑，没有悬空问题。")
    
    return suspended_items

def main():
    # 运行优化算法获取解决方案
    container_dims = (1180, 230, 260)
    suppliers_sequence, cargo_data = load_cargo_data("装柜0538.xlsx")
    if not cargo_data:
        return
    
    optimizer = SimulatedAnnealingOptimizer(container_dims)
    
    # 为了快速测试，只生成初始解
    optimizer.all_cargo = optimizer._preprocess_cargo(cargo_data)
    optimizer.regions = optimizer._create_supplier_regions(suppliers_sequence, optimizer.all_cargo)
    optimizer.layer_strategy = LayerStrategy(container_dims, optimizer.all_cargo)
    
    print("正在生成初始解...")
    optimizer.is_in_optimization_phase = False
    solution = optimizer._create_initial_solution()
    
    print(f"\n初始解装载率: {optimizer._get_volume_ratio(solution):.2%}")
    print(f"装载件数: {len(solution.placed_items)}")
    
    # 检查悬空情况
    suspended_items = check_cargo_support(solution, container_dims)
    
    # 如果有悬空，显示统计
    if suspended_items:
        suppliers_affected = {}
        for info in suspended_items:
            supplier = info['supplier']
            if supplier not in suppliers_affected:
                suppliers_affected[supplier] = 0
            suppliers_affected[supplier] += 1
        
        print("\n### 悬空问题统计 ###")
        print(f"总悬空货物数: {len(suspended_items)}")
        print("按供应商分布:")
        for supplier, count in suppliers_affected.items():
            print(f"  - {supplier}: {count} 个")

if __name__ == "__main__":
    main() 