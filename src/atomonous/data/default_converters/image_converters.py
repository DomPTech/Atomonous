from pathlib import Path
from typing import List, Type, Any
import numpy as np
from PIL import Image
from ..converters import DataConverter, FileDataConverter, HeuristicMismatchError
from ..types import FilePath

class TiffConverter(FileDataConverter[FilePath]):
    """
    Converts TIFF files to PIL.Image.Image.
    """

    input_type = FilePath
    supported_extensions = {".tiff", ".tif"}

    def convert(self, data: FilePath) -> Image.Image:
        path = Path(data)
        return Image.open(path)

class NumpyImageConverter(FileDataConverter[np.ndarray | FilePath]):
    """
    Uses heuristics to convert numpy arrays (or .npy files) to PIL.Image.Image.
    """

    input_type = (np.ndarray, FilePath)
    supported_extensions = {".npy"}

    def convert(self, data: np.ndarray | FilePath) -> Image.Image:
        if isinstance(data, FilePath):
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"NPY file not found: {path}")
            arr = np.load(str(path))
        else:
            arr = data

        # Heuristic check: 2D or 3D (small channels)
        is_plausible_image = False
        if arr.ndim == 2:
            is_plausible_image = True
        elif arr.ndim == 3 and arr.shape[-1] in [1, 3, 4]:
            is_plausible_image = True

        if not is_plausible_image:
            raise HeuristicMismatchError(f"Numpy array shape {arr.shape} is not image-like.")

        # PIL works best on 8bit data
        if arr.dtype != np.uint8:
            arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-5) * 255).astype(np.uint8)

        return Image.fromarray(arr)