import time
import pytest
import numpy as np
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.microscope_api import (
    MicroscopeControl,
    StagePosition,
    start_server,
    connect_client,
    close_microscope,
    capture_image,
)
from app.config import settings


@pytest.fixture(scope="module", autouse=True)
def microscope_setup():
    print("\n--- Initializing Image Stitching Test Environment ---")
    start_res = start_server(mode="mock")
    print(f"Start Server: {start_res}")
    time.sleep(2)
    conn_res = connect_client(host="localhost")
    print(f"Connect Client: {conn_res}")
    yield
    print("\n--- Tearing Down Image Stitching Test Environment ---")
    close_microscope()


def test_image_stitching_workflow():
    print("\n--- Testing Image Stitching Workflow ---")
    resolutions = [(256, 256), (256, 256), (256, 256), (256, 256)]
    image_paths = []
    images = []
    control = MicroscopeControl(sim_mode=True)
    positions = [
        StagePosition(x=0, y=0, z=0), #llm needs to determine its own stage positions
        StagePosition(x=0, y=256, z=0),#as the sample is much larger 
        StagePosition(x=256, y=0, z=0), #basically these shouldn't be hardcoded
        StagePosition(x=256, y=256, z=0),
    ]

    for idx, res in enumerate(resolutions):
        pos = positions[idx]
        control.set_stage_position(pos)
        print(f"Capturing image at resolution: {res} at position: {(pos.x, pos.y, pos.z)}")
        img_res = control.acquire_image(detector="HAADF")
        print(f"Capture image: {img_res}")
        assert ".npy" in img_res
        image_paths.append(img_res)
        data = np.load(img_res)
        resized = np.resize(data, res)
        images.append(resized)

    top = np.concatenate(images[0:2], axis=1)
    bottom = np.concatenate(images[2:4], axis=1)
    stitched = np.concatenate([top, bottom], axis=0)
    print(f"Stitched image shape: {stitched.shape}")
    expected_shape = (resolutions[0][0] * 2, resolutions[0][1] * 2)
    assert stitched.shape == expected_shape

    artifacts_dir = Path(settings.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stitched_path = artifacts_dir / "stitched_image.npy"
    np.save(stitched_path, stitched)
    print(f"Stitched image saved: {stitched_path}")
    assert stitched_path.exists()

    for path in image_paths:
        if os.path.exists(path):
            os.remove(path)
    if stitched_path.exists():
        os.remove(stitched_path)

    print("Image stitching workflow test passed!")


if __name__ == "__main__":
    pytest.main([__file__])