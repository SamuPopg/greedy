"""
Data Parser Module
Handles loading and preprocessing of cargo data from various sources.
"""
import pandas as pd
import re
from typing import List, Dict

from src.core.models import Cargo

class DataParser:
    """
    Parses cargo data from input files into a list of Cargo objects.
    """

    def load_and_preprocess_cargo(self, file_path: str) -> List[Cargo]:
        """
        Loads data from an Excel file and preprocesses it into a list of Cargo objects.

        :param file_path: Path to the Excel file.
        :return: A list of individual Cargo objects.
        """
        try:
            raw_data = self._load_from_excel(file_path)
            cargo_list = self._create_cargo_objects(raw_data)
            print("=== 真实数据加载完成 ===")
            print(f"总货物种类: {len(raw_data)}")
            total_items = sum(item['數量'] for item in raw_data)
            print(f"总货物数量: {total_items}")
            total_volume_theoretical = sum(item['長度']*item['寬度']*item['高度']*item['數量'] for item in raw_data)
            print(f"理论总体积: {total_volume_theoretical:.2f}cm³")
            return cargo_list
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return []

    def _load_from_excel(self, file_path: str) -> List[Dict]:
        """Loads data from an excel file into a list of dictionaries."""
        excel_data = pd.read_excel(file_path)
        return excel_data.to_dict('records')

    def _create_cargo_objects(self, cargo_data: List[Dict]) -> List[Cargo]:
        """Converts raw data dictionaries into Cargo objects."""
        cargo_list = []
        supplier_pattern = re.compile(r'（(.*?)）')
        for item in cargo_data:
            match = supplier_pattern.search(item.get('貨物名稱', ''))
            supplier_name = match.group(1) if match else "UnknownSupplier"
            for _ in range(int(item['數量'])):
                cargo_list.append(Cargo(
                    cargo_id=item.get('貨物名稱', 'Unknown'),
                    supplier=supplier_name,
                    length=item['長度'], width=item['寬度'], height=item['高度'], weight=item['重量']
                ))
        return cargo_list

    def extract_suppliers(self, cargo_data: List[Dict]) -> List[str]:
        """Extracts a unique, ordered list of suppliers from the raw cargo data."""
        supplier_pattern = re.compile(r'（(.*?)）')
        suppliers_sequence = []
        seen_suppliers = set()
        for item in cargo_data:
            match = supplier_pattern.search(item.get('貨物名稱', ''))
            if match:
                supplier_name = match.group(1)
                if supplier_name not in seen_suppliers:
                    seen_suppliers.add(supplier_name)
                    suppliers_sequence.append(supplier_name)
        return suppliers_sequence
