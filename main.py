#!/usr/bin/env python3
"""
装柜优化系统 V3 - 主程序
结合了模块化架构和并行计算的核心优势
"""
import argparse
import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from src.data_parser import DataParser
from src.algorithms.permutation_optimizer import PermutationOptimizer

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="装柜优化系统 V3 - 并行优化版"
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        default="./data/装柜0538.xlsx",
        help="数据文件路径 (支持 .xlsx)"
    )
    args = parser.parse_args()

    print("🚀 装柜优化系统 V3 启动")
    print("=" * 50)

    # 1. 数据解析
    print("步骤 1/3: 解析数据文件...")
    parser = DataParser()
    # Note: We need the raw data dict for supplier extraction
    raw_cargo_data = parser._load_from_excel(args.data)
    if not raw_cargo_data:
        print("错误：无法从文件中加载数据。")
        return 1
    
    all_cargo_objects = parser._create_cargo_objects(raw_cargo_data)
    all_suppliers = parser.extract_suppliers(raw_cargo_data)

    if not all_cargo_objects:
        print("错误：数据文件为空或格式不正确。")
        return 1

    # 2. 核心优化
    print("\n步骤 2/3: 执行核心优化算法...")
    container_dims = (1180, 230, 260)
    optimizer = PermutationOptimizer(
        container_dims=container_dims,
        all_cargo=all_cargo_objects,
        all_suppliers=all_suppliers
    )
    optimizer.run_optimization()

    # 3. 完成
    print("\n步骤 3/3: 优化流程完成！")
    print("=" * 50)
    
    return 0

if __name__ == "__main__":
    # To ensure multiprocessing works correctly on all platforms,
    # especially Windows and macOS.
    main()