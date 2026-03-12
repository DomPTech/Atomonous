import os
import sys
import re
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.tools.microscopy import start_server, connect_client, take_image, close_microscope

def test_minimal_pytango_image_flow():
    image_path = None
    try:
        start_res = start_server(mode="mock")
        assert "Started PyTango asyncroscopy context" in start_res or "already running" in start_res

        conn_res = connect_client()
        assert "Connected to PyTango microscope client" in conn_res

        img_res = take_image(detector="haadf")
        assert "Image captured from detector 'haadf'" in img_res
        assert ".npy" in img_res

        match = re.search(r"saved to (.+?) \(Shape", img_res)
        assert match is not None
        image_path = match.group(1)
        assert os.path.exists(image_path)
    finally:
        close_microscope()
        if image_path and os.path.exists(image_path):
            os.remove(image_path)

if __name__ == "__main__":
    # Fallback for running without pytest
    pytest.main([__file__])
