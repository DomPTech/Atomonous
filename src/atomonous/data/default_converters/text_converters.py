import json
from pathlib import Path
from typing import List, Dict, Any, Type
import numpy as np
import pandas as pd
import h5py
from ..converters import DataConverter, FileDataConverter, HeuristicMismatchError
from ..types import FilePath

class CsvConverter(FileDataConverter[pd.DataFrame | FilePath]):
    """
    Converts CSV files or DataFrames to strings.
    """

    input_type = (pd.DataFrame, FilePath)
    supported_extensions = {".csv"}

    def convert(self, data: pd.DataFrame | FilePath) -> str:
        if isinstance(data, FilePath):
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"CSV file not found: {path}")
            df = pd.read_csv(str(path))
        else:
            df = data

        summary = {
            "columns": list(df.columns),
            "rows": len(df),
            "preview": df.head(5).to_dict(orient="records")
        }
        
        return json.dumps(summary, indent=2)

class Hdf5SummaryConverter(FileDataConverter[h5py.File | h5py.Group | FilePath]):
    """
    Converts HDF5 files or groups to hierarchical strings.
    """

    input_type = (h5py.File, h5py.Group, FilePath)
    supported_extensions = {".h5", ".hdf5"}

    def _summarize(self, item: h5py.File | h5py.Group) -> Dict[str, Any]:
        summary = {}
        for name, sub_item in item.items():
            if isinstance(sub_item, (h5py.File, h5py.Group)):
                summary[name] = {"type": "group", "children": self._summarize(sub_item)}
            elif isinstance(sub_item, h5py.Dataset):
                summary[name] = {
                    "type": "dataset",
                    "shape": list(sub_item.shape),
                    "dtype": str(sub_item.dtype)
                }
        return summary

    def convert(self, data: h5py.File | h5py.Group | FilePath) -> str:
        if isinstance(data, FilePath):
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"HDF5 file not found: {path}")
            with h5py.File(str(path), 'r') as f:
                full_summary = self._summarize(f)
        else:
            full_summary = self._summarize(data)

        return json.dumps(full_summary, indent=2)

class DictConverter(DataConverter[dict]):
    """Default converter for dictionaries."""
    input_type = dict

    def convert(self, data: dict) -> str:
        return json.dumps(data, indent=2)