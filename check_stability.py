import pandas as pd
import numpy as np
import re
from container_optimizer import SimulatedAnnealingOptimizer, load_cargo_data, LayerStrategy

def analyze_cargo_stability(solution, container_dims):
    """全面分析货物的稳定性"""
    print("\n=== 稳定性分析报告 ===")
    
    stability_issues = []
    support_stats = {
        '0-50%': 0,
        '50-70%': 0, 
        '70-90%': 0,
        '90-100%': 0
    }
    center_stable_count = 0
    center_unstable_count = 0
    
    for i, item in enumerate(solution.placed_items):
        x, y, z = item.position.x, item.position.y, item.position.z
        l, w, h = item.current_dims
        
        # 跳过地面货物
        if z == 0:
            support_stats['90-100%'] += 1
            center_stable_count += 1
            continue
        
        # 1. 计算支撑率
        bottom_area = l * w
        supported_area = 0
        
        # 支撑区域边界
        support_min_x = float('inf')
        support_max_x = float('-inf')
        support_min_y = float('inf')
        support_max_y = float('-inf')
        
        for j, other_item in enumerate(solution.placed_items):
            if i == j:
                continue
                
            ox, oy, oz = other_item.position.x, other_item.position.y, other_item.position.z
            ol, ow, oh = other_item.current_dims
            
            # 检查是否正好支撑
            if abs((oz + oh) - z) > 0.1:
                continue
            
            # 计算重叠区域
            x_overlap_start = max(x, ox)
            x_overlap_end = min(x + l, ox + ol)
            y_overlap_start = max(y, oy)
            y_overlap_end = min(y + w, oy + ow)
            
            if x_overlap_start < x_overlap_end and y_overlap_start < y_overlap_end:
                overlap_area = (x_overlap_end - x_overlap_start) * (y_overlap_end - y_overlap_start)
                supported_area += overlap_area
                
                # 更新支撑边界
                support_min_x = min(support_min_x, x_overlap_start)
                support_max_x = max(support_max_x, x_overlap_end)
                support_min_y = min(support_min_y, y_overlap_start)
                support_max_y = max(support_max_y, y_overlap_end)
        
        support_ratio = supported_area / bottom_area if bottom_area > 0 else 0
        
        # 统计支撑率分布
        if support_ratio < 0.5:
            support_stats['0-50%'] += 1
        elif support_ratio < 0.7:
            support_stats['50-70%'] += 1
        elif support_ratio < 0.9:
            support_stats['70-90%'] += 1
        else:
            support_stats['90-100%'] += 1
        
        # 2. 检查重心稳定性
        center_x = x + l / 2
        center_y = y + w / 2
        center_stable = False
        
        if support_min_x != float('inf'):  # 有支撑
            margin_x = (support_max_x - support_min_x) * 0.15
            margin_y = (support_max_y - support_min_y) * 0.15
            
            if (support_min_x + margin_x <= center_x <= support_max_x - margin_x and
                support_min_y + margin_y <= center_y <= support_max_y - margin_y):
                center_stable = True
                center_stable_count += 1
            else:
                center_unstable_count += 1
        else:
            center_unstable_count += 1
        
        # 记录有问题的货物
        if support_ratio < 0.7 or not center_stable:
            stability_issues.append({
                'index': i,
                'supplier': item.cargo.supplier,
                'position': (x, y, z),
                'dimensions': (l, w, h),
                'support_ratio': support_ratio,
                'center_stable': center_stable
            })
    
    # 输出统计信息
    total_items = len(solution.placed_items)
    ground_items = sum(1 for item in solution.placed_items if item.position.z == 0)
    elevated_items = total_items - ground_items
    
    print(f"\n总货物数: {total_items}")
    print(f"地面货物: {ground_items}")
    print(f"非地面货物: {elevated_items}")
    
    print("\n支撑率分布:")
    for range_name, count in support_stats.items():
        percentage = count / total_items * 100
        print(f"  {range_name}: {count} 个 ({percentage:.1f}%)")
    
    print(f"\n重心稳定性:")
    print(f"  稳定: {center_stable_count} 个")
    print(f"  不稳定: {center_unstable_count} 个")
    
    # 输出问题货物详情
    if stability_issues:
        print(f"\n发现 {len(stability_issues)} 个稳定性问题:")
        for issue in stability_issues[:5]:  # 只显示前5个
            print(f"\n货物 #{issue['index']}:")
            print(f"  供应商: {issue['supplier']}")
            print(f"  位置: ({issue['position'][0]:.1f}, {issue['position'][1]:.1f}, {issue['position'][2]:.1f})")
            print(f"  支撑率: {issue['support_ratio']:.1%}")
            print(f"  重心稳定: {'是' if issue['center_stable'] else '否'}")
    else:
        print("\n✅ 所有货物都满足稳定性要求！")
    
    # 计算整体稳定性评分
    avg_support_ratio = sum(s['support_ratio'] for s in stability_issues) / len(stability_issues) if stability_issues else 1.0
    stability_score = (1 - len(stability_issues) / total_items) * 100
    
    print(f"\n整体稳定性评分: {stability_score:.1f}/100")
    
    return stability_issues, stability_score

def main():
    # 运行优化算法
    container_dims = (1180, 230, 260)
    suppliers_sequence, cargo_data = load_cargo_data("装柜0538.xlsx")
    if not cargo_data:
        return
    
    optimizer = SimulatedAnnealingOptimizer(container_dims)
    
    # 生成初始解
    optimizer.all_cargo = optimizer._preprocess_cargo(cargo_data)
    optimizer.regions = optimizer._create_supplier_regions(suppliers_sequence, optimizer.all_cargo)
    optimizer.layer_strategy = LayerStrategy(container_dims, optimizer.all_cargo)
    
    print("正在生成初始解...")
    optimizer.is_in_optimization_phase = False
    solution = optimizer._create_initial_solution()
    
    print(f"\n初始解装载率: {optimizer._get_volume_ratio(solution):.2%}")
    print(f"装载件数: {len(solution.placed_items)}")
    
    # 分析稳定性
    stability_issues, stability_score = analyze_cargo_stability(solution, container_dims)

if __name__ == "__main__":
    main() 