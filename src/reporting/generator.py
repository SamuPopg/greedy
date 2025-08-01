"""
Output Generator Module
Handles the creation of all output files (Excel reports, 3D visualizations).
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List, Tuple

from src.core.models import PackingSolution, Cargo

class OutputGenerator:
    """
    Generates all output files for a given packing solution.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup matplotlib for Chinese characters
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def generate_all_outputs(self, solution: PackingSolution, container_dims: Tuple, rank: int):
        """
        Generates all standard output files for a solution.
        
        :param solution: The packing solution to report.
        :param container_dims: The dimensions of the container.
        :param rank: The rank of this solution (e.g., 1 for Top 1).
        """
        file_prefix = self._create_file_prefix(solution, rank)
        
        excel_path = os.path.join(self.output_dir, f"{file_prefix}.xlsx")
        self._export_solution_to_excel(solution, excel_path)
        
        image_path = os.path.join(self.output_dir, f"{file_prefix}.png")
        self._create_3d_visualization(solution, container_dims, image_path)

    def _create_file_prefix(self, solution: PackingSolution, rank: int) -> str:
        """Creates a unique and descriptive file prefix."""
        sequence_file_str = '-'.join(solution.sequence)
        return f"方案_{rank}_顺序_{sequence_file_str}_装载率_{solution.rate:.4f}"

    def _export_solution_to_excel(self, solution: PackingSolution, save_path: str):
        """将详细的装载方案导出为Excel文件。"""
        print(f"  - 正在生成详细报告: {save_path}")
        
        if not solution.placed_items:
            print(f"警告：方案 {save_path} 为空，无法导出Excel。")
            return

        solution_data = []
        for item in solution.placed_items:
            solution_data.append({
                '貨物名稱': item.cargo.cargo_id, '供應商': item.cargo.supplier,
                '原始長度': item.cargo.original_dims[0], '原始寬度': item.cargo.original_dims[1],
                '原始高度': item.cargo.original_dims[2], '放置座標X': round(item.position.x, 2),
                '放置座標Y': round(item.position.y, 2), '放置座標Z': round(item.position.z, 2),
                '擺放方式': item.orientation.value, '當前長度': item.current_dims[0],
                '當前寬度': item.current_dims[1], '當前高度': item.current_dims[2],
            })
        
        df = pd.DataFrame(solution_data)
        df_sorted = df.sort_values(by=['放置座標X', '放置座標Y', '放置座標Z'])
        
        try:
            df_sorted.to_excel(save_path, index=False, engine='openpyxl')
        except Exception as e:
            print(f"错误：导出Excel文件 {save_path} 失败。原因: {e}")

    def _create_3d_visualization(self, solution: PackingSolution, container_dims: Tuple, save_path: str):
        """创建装载方案的3D可视化图"""
        print(f"  - 正在生成3D视图: {save_path}")
        
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        supplier_colors = {'纽蓝': 'lightblue', '海信': 'lightgreen', '福美高': 'lightcoral'}
        
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
            ax.plot3D(*zip(*points), 'k-', linewidth=1, alpha=0.3)
        
        for item in solution.placed_items:
            x, y, z = item.position.x, item.position.y, item.position.z
            l, w, h = item.current_dims
            color = supplier_colors.get(item.cargo.supplier, 'gray')
            
            vertices = [
                [x, y, z], [x+l, y, z], [x+l, y+w, z], [x, y+w, z],
                [x, y, z+h], [x+l, y, z+h], [x+l, y+w, z+h], [x, y+w, z+h]
            ]
            faces = [
                [vertices[0], vertices[1], vertices[5], vertices[4]], [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[0], vertices[3], vertices[7], vertices[4]], [vertices[1], vertices[2], vertices[6], vertices[5]],
                [vertices[0], vertices[1], vertices[2], vertices[3]], [vertices[4], vertices[5], vertices[6], vertices[7]]
            ]
            
            face_collection = Poly3DCollection(faces, alpha=0.7, facecolor=color, edgecolor='black', linewidth=0.5)
            ax.add_collection3d(face_collection)
        
        ax.set_xlabel('长度 (cm)', fontsize=12)
        ax.set_ylabel('宽度 (cm)', fontsize=12)
        ax.set_zlabel('高度 (cm)', fontsize=12)
        ax.set_title(f'集装箱装载3D可视化\n装载率: {solution.rate:.2%} | 装载件数: {len(solution.placed_items)}件', fontsize=16)
        
        ax.set_xlim([0, container_dims[0]])
        ax.set_ylim([0, container_dims[1]])
        ax.set_zlim([0, container_dims[2]])
        ax.grid(True, alpha=0.3)
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=supplier) for supplier, color in supplier_colors.items()]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))
        
        ax.view_init(elev=20, azim=45)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
