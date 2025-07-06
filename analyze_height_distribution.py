import pandas as pd
import numpy as np
import re
from collections import defaultdict

def analyze_cargo_height_distribution(file_path):
    """分析各供应商货物的高度分布"""
    excel_data = pd.read_excel(file_path)
    cargo_data = excel_data.to_dict('records')
    
    supplier_pattern = re.compile(r'（(.*?)）')
    supplier_heights = defaultdict(list)
    supplier_volumes = defaultdict(float)
    supplier_items = defaultdict(int)
    
    for item in cargo_data:
        match = supplier_pattern.search(item.get('貨物名稱', ''))
        if match:
            supplier = match.group(1)
            height = item['高度']
            quantity = item['數量']
            volume = item['長度'] * item['寬度'] * item['高度'] * quantity
            
            # 记录每个货物的高度（按数量展开）
            for _ in range(quantity):
                supplier_heights[supplier].append(height)
            
            supplier_volumes[supplier] += volume
            supplier_items[supplier] += quantity
    
    print("=== 各供应商货物高度分析 ===\n")
    
    for supplier in ['纽蓝', '海信', '福美高']:
        if supplier in supplier_heights:
            heights = supplier_heights[supplier]
            print(f"{supplier}:")
            print(f"  货物数量: {len(heights)}")
            print(f"  总体积: {supplier_volumes[supplier]:.0f} cm³")
            print(f"  高度范围: {min(heights):.1f} - {max(heights):.1f} cm")
            print(f"  平均高度: {np.mean(heights):.1f} cm")
            print(f"  中位数高度: {np.median(heights):.1f} cm")
            
            # 高度分布
            print(f"  高度分布:")
            bins = [0, 10, 20, 50, 100, 200, 300]
            hist, _ = np.histogram(heights, bins=bins)
            for i in range(len(bins)-1):
                count = hist[i]
                if count > 0:
                    percentage = count / len(heights) * 100
                    print(f"    {bins[i]}-{bins[i+1]}cm: {count}个 ({percentage:.1f}%)")
            print()
    
    # 分析可能的堆叠组合
    print("=== 垂直空间优化建议 ===")
    for supplier in ['纽蓝', '海信', '福美高']:
        if supplier in supplier_heights:
            heights = supplier_heights[supplier]
            unique_heights = sorted(set(heights))
            
            # 找出可以有效堆叠的高度组合
            container_height = 260
            print(f"\n{supplier}的堆叠优化:")
            
            # 统计每种高度的数量
            height_counts = {}
            for h in heights:
                height_counts[h] = height_counts.get(h, 0) + 1
            
            # 找出主要的高度类型（数量较多的）
            major_heights = sorted([(h, count) for h, count in height_counts.items() if count >= 5], 
                                 key=lambda x: x[1], reverse=True)[:5]
            
            if major_heights:
                print(f"  主要高度类型:")
                for h, count in major_heights:
                    print(f"    {h}cm: {count}个")
                
                # 建议堆叠方案
                print(f"  建议堆叠方案:")
                for i, (h1, c1) in enumerate(major_heights):
                    for j, (h2, c2) in enumerate(major_heights[i:]):
                        if h1 + h2 <= container_height - 20:  # 留20cm余量
                            print(f"    {h1}cm + {h2}cm = {h1+h2}cm (可用数量: {min(c1, c2)}组)")

if __name__ == "__main__":
    analyze_cargo_height_distribution("装柜0538.xlsx") 