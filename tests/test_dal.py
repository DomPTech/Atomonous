import numpy as np
import pandas as pd
from PIL import Image
from atomonous.data.factory import ConverterFactory
from atomonous.data.converters import DataConverter
import pytest
import h5py


# Mock class
class MicroscopeImage:
    def __init__(self, data: np.ndarray, name: str):
        self.data = data
        self.name = name

# Mock Converter returning PIL Image
class MicroscopeImageConverter(DataConverter[MicroscopeImage]):
    input_type = MicroscopeImage

    def convert(self, data: MicroscopeImage) -> Image.Image:
        arr = data.data
        if arr.dtype != np.uint8:
            arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-5) * 255).astype(np.uint8)
        return Image.fromarray(arr)

@pytest.fixture
def factory():
    return ConverterFactory(register_default=True)

@pytest.fixture
def test_files(tmp_path):
    # CSV
    csv_path = tmp_path / "test.csv"
    pd.DataFrame({"X": [1, 2]}).to_csv(csv_path, index=False)

    # TIFF
    tiff_path = tmp_path / "test.tiff"
    img_data = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
    Image.fromarray(img_data).save(tiff_path)

    # HDF5
    h5_path = tmp_path / "test.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("data", data=np.zeros((10, 10)))
        group = f.create_group("subgroup")
        group.create_dataset("meta", data=np.ones((5,)))

    # NPY
    npy_path = tmp_path / "test.npy"
    np.save(npy_path, np.random.randint(0, 255, (32, 32), dtype=np.uint8))

    return {
        "csv": csv_path,
        "tiff": tiff_path,
        "h5": h5_path,
        "npy": npy_path
    }


def test_csv_to_string(factory, test_files):
    result = factory.convert(test_files["csv"])
    assert isinstance(result, str)
    assert "columns" in result


def test_tiff_to_pil(factory, test_files):
    result = factory.convert(test_files["tiff"])
    assert isinstance(result, Image.Image)
    assert result.size == (50, 50)


def test_custom_class_to_pil(factory):
    factory.register_converter(MicroscopeImageConverter())
    mi = MicroscopeImage(
        np.random.randint(0, 255, (64, 64), dtype=np.uint8),
        "Test"
    )

    result = factory.convert(mi)
    assert isinstance(result, Image.Image)
    assert result.size == (64, 64)


def test_dict_to_string(factory):
    result = factory.convert({"status": "ok"})
    assert isinstance(result, str)
    assert '"status": "ok"' in result


def test_h5_to_summary(factory, test_files):
    result = factory.convert(test_files["h5"])
    assert isinstance(result, str)
    assert "subgroup" in result
    assert "data" in result


def test_numpy_file_to_pil(factory, test_files):
    result = factory.convert(test_files["npy"])
    assert isinstance(result, Image.Image)
    assert result.size == (32, 32)


def test_numpy_array_to_pil(factory):
    arr = np.random.randint(0, 255, (16, 16), dtype=np.uint8)
    result = factory.convert(arr)
    assert isinstance(result, Image.Image)
    assert result.size == (16, 16)


def test_mcp_json_to_string(factory):
    mcp_data = {
        "payload": "Hello MCP",
        "metadata": {"type": "text"},
        "encoding": "utf-8"
    }
    result = factory.convert(mcp_data)
    assert result == "Hello MCP"