import pandas as pd
import numpy as np
import re
from container_optimizer import SimulatedAnnealingOptimizer, load_cargo_data, LayerStrategy, PlacementType

def check_layer_rule(solution, layer_boundary=20.0):
    """
    检查装载方案是否遵守"下两层禁立放"的规则。
    """
    print(f"\n=== 下两层（Z < {layer_boundary}cm）摆放规则检测 ===")
    violations = []
    
    for item in solution.placed_items:
        z_position = item.position.z
        orientation = item.orientation
        
        # 检查是否在下两层
        if z_position < layer_boundary:
            # 检查是否是立放
            if orientation in [PlacementType.UPRIGHT_X, PlacementType.UPRIGHT_Y]:
                violations.append({
                    'item': item,
                    'position': item.position,
                    'orientation': orientation.value,
                    'supplier': item.cargo.supplier
                })
    
    if violations:
        print(f"\n❌ 发现 {len(violations)} 个违反“下两层禁立放”规则的货物:")
        for info in violations[:10]: # 只显示前10个
            item = info['item']
            print("\n--------------------")
            print(f"  货物ID: {item.cargo.cargo_id}")
            print(f"  供应商: {info['supplier']}")
            print(f"  位置: X={info['position'].x:.1f}, Y={info['position'].y:.1f}, Z={info['position'].z:.1f}")
            print(f"  尺寸 (LxWxH): {item.current_dims[0]:.1f} x {item.current_dims[1]:.1f} x {item.current_dims[2]:.1f}")
            print(f"  🔥 违规摆放方式: {info['orientation']}")
            print("--------------------")
    else:
        print("\n✅ 所有货物都遵守了“下两层禁立放”的规则。")
        
    return violations

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
    
    # 检查分层规则
    violations = check_layer_rule(solution)

if __name__ == "__main__":
    main() 