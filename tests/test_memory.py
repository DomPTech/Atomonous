import pytest
from PIL import Image
import numpy as np
from pathlib import Path
from atomonous.utils.memory import SessionMemory

def test_save_pil_image(tmp_path):
    memory = SessionMemory(artifacts_base_dir=str(tmp_path), session_name="test_session")
    
    # Dummy PIL image
    arr = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
    img = Image.fromarray(arr)
    
    saved_path_str = memory.save_pil_image(img, "test_intercept")
    saved_path = Path(saved_path_str)
    
    # Verify the file was created and is a PNG
    assert saved_path.exists()
    assert saved_path.suffix == ".png"
    assert "image_test_intercept_" in saved_path.name
    
    # Verify we can open it
    opened_img = Image.open(saved_path)
    assert opened_img.size == (10, 10)
