"""
交互式3D装载方案查看器
--------------------------
使用方法:
python3 view_solution.py <path_to_excel_solution_file>

示例:
python3 view_solution.py ./output/方案_1_顺序_海信-福美高-纽蓝_装载率_0.8315.xlsx
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch
import re
import os

def view_solution(file_path: str):
    """
    读取一个Excel格式的装载方案文件，并生成一个可交互的3D视图。

    :param file_path: Excel方案文件的路径。
    """
    # 1. 读取Excel文件
    try:
        df = pd.read_excel(file_path)
        print(f"✅ 成功读取方案文件: {os.path.basename(file_path)}")
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {file_path}")
        return
    except Exception as e:
        print(f"❌ 错误:读取Excel文件时出错: {e}")
        return

    # 2. 初始化3D绘图环境
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置中文字体，确保标题和标签能正确显示
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 3. 定义常量（集装箱尺寸和供应商颜色）
    container_dims = (1180, 230, 260)
    supplier_colors = {'纽蓝': 'lightblue', '海信': 'lightgreen', '福美高': 'lightcoral', 'UnknownSupplier': 'gray'}

    # 4. 绘制集装箱线框
    container_x, container_y, container_z = container_dims
    container_vertices = [
        [0, 0, 0], [container_x, 0, 0], [container_x, container_y, 0], [0, container_y, 0],
        [0, 0, container_z], [container_x, 0, container_z], [container_x, container_y, container_z], [0, container_y, container_z]
    ]
    container_edges = [
        [0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    for edge in container_edges:
        points = [container_vertices[edge[0]], container_vertices[edge[1]]]
        ax.plot3D(*zip(*points), 'k-', linewidth=1, alpha=0.2)
    
    # 5. 从DataFrame中读取数据并绘制每一个货物
    for _, item in df.iterrows():
        x, y, z = item['放置座標X'], item['放置座標Y'], item['放置座標Z']
        l, w, h = item['當前長度'], item['當前寬度'], item['當前高度']
        supplier = item.get('供應商', 'UnknownSupplier')
        color = supplier_colors.get(supplier, 'gray')

        vertices = [
            [x, y, z], [x+l, y, z], [x+l, y+w, z], [x, y+w, z],
            [x, y, z+h], [x+l, y, z+h], [x+l, y+w, z+h], [x, y+w, z+h]
        ]
        faces = [
            [vertices[0], vertices[1], vertices[5], vertices[4]], [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[0], vertices[3], vertices[7], vertices[4]], [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[1], vertices[2], vertices[3]], [vertices[4], vertices[5], vertices[6], vertices[7]]
        ]
        face_collection = Poly3DCollection(faces, alpha=0.8, facecolor=color, edgecolor='black', linewidth=0.5)
        ax.add_collection3d(face_collection)

    # 6. 美化图表：设置标题、坐标轴、图例等
    ax.set_xlabel('长度 (cm)', fontsize=12)
    ax.set_ylabel('宽度 (cm)', fontsize=12)
    ax.set_zlabel('高度 (cm)', fontsize=12)
    
    # 尝试从文件名解析信息以生成更丰富的标题
    filename = os.path.basename(file_path)
    title = f'集装箱装载3D可视化\n{filename}'
    match = re.search(r'装载率_(\d\.\d+)', filename)
    if match:
        rate = float(match.group(1))
        title = f'集装箱装载3D可视化\n装载率: {rate:.2%} | 装载件数: {len(df)}件'

    ax.set_title(title, fontsize=16, pad=20)
    
    ax.set_xlim([0, container_dims[0]])
    ax.set_ylim([0, container_dims[1]])
    ax.set_zlim([0, container_dims[2]])
    ax.grid(True, alpha=0.3)
    
    # 创建图例
    legend_elements = [Patch(facecolor=color, label=supplier) for supplier, color in supplier_colors.items() if supplier in df['供應商'].unique()]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.95))
    
    # 设置一个默认的最佳观察视角
    ax.view_init(elev=20, azim=45)

    # 7. 显示交互式窗口
    print("✨ 正在打开交互式3D视图...")
    print("   您可以按住鼠标左键拖动以旋转，按住右键拖动以缩放。")
    print("   关闭此窗口以结束程序。")
    plt.show()

if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(
        description="交互式3D装载方案查看器。接收一个Excel方案文件作为输入。",
        formatter_class=argparse.RawTextHelpFormatter # 保持帮助文本格式
    )
    parser.add_argument(
        "solution_file",
        type=str,
        help="由优化器生成的Excel方案文件路径。\n例如: ./output/方案_1_..._.xlsx"
    )
    args = parser.parse_args()
    
    view_solution(args.solution_file) 