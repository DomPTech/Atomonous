from typing import Optional, Dict, Any, Union
import numpy as np
import json
from pydantic import BaseModel, Field, validator
from app.config import settings

class StagePosition(BaseModel):
    x: float = Field(..., description="X position in microns")
    y: float = Field(..., description="Y position in microns")
    z: Optional[float] = Field(None, description="Z position in microns")
    rotation: Optional[float] = Field(None, description="Rotation in degrees")
    tilt: Optional[float] = Field(None, description="Tilt in degrees")

    @validator('x')
    def x_within_bounds(cls, v):
        if not (settings.stage_x_min <= v <= settings.stage_x_max):
            raise ValueError(f"X must be between {settings.stage_x_min} and {settings.stage_x_max}")
        return v

    @validator('y')
    def y_within_bounds(cls, v):
        if not (settings.stage_y_min <= v <= settings.stage_y_max):
            raise ValueError(f"Y must be between {settings.stage_y_min} and {settings.stage_y_max}")
        return v

class ImageResult(BaseModel):
    data: Any = Field(..., description="Numpy array or base64 encoded image data")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MicroscopeControl:
    """
    High-level, typed API for microscope control.
    Wraps asyncroscopy and handles safety checks.
    """
    def __init__(self, sim_mode: bool = None):
        self.sim_mode = sim_mode if sim_mode is not None else settings.sim_mode
        self._client = None
        self._backend = "sim"
        
        if not self.sim_mode:
            self._connect()

    def _connect(self):
        """Connect to the PyTango microscope device if not in simulation mode."""
        try:
            import tango

            device_name = "test/nodb/microscope"
            self._client = tango.DeviceProxy(device_name)
            _ = self._client.state()
            self._backend = "pytango"
            print(f"Connected to PyTango microscope device at {device_name}")
        except Exception as e:
            print(f"Failed to connect to microscope central server: {e}")
            self.sim_mode = True
            self._backend = "sim"

    def get_stage_position(self, destination: str = "AS") -> StagePosition:
        """Get the current stage position."""
        if self.sim_mode:
            return StagePosition(x=100.0, y=100.0, z=0.0)
        
        # asyncroscopy returns stage in nm/deg
        raw_pos = self._client.send_command(destination, "get_stage")
        
        # Mapping depends on whether AS_server_AtomBlastTwin or another is used
        # AtomBlastTwin returns random list [x, y, z, r, t]
        if isinstance(raw_pos, (list, np.ndarray)):
            return StagePosition(
                x=raw_pos[0] / 1000.0,
                y=raw_pos[1] / 1000.0,
                z=raw_pos[2] / 1000.0,
                rotation=raw_pos[3],
                tilt=raw_pos[4]
            )
        
        return StagePosition(
            x=raw_pos['x'] / 1000.0,
            y=raw_pos['y'] / 1000.0,
            z=raw_pos.get('z', 0) / 1000.0,
            rotation=raw_pos.get('r'),
            tilt=raw_pos.get('t')
        )

    def set_stage_position(self, pos: StagePosition, relative: bool = False, destination: str = "AS") -> StagePosition:
        """Set the stage position with safety checks."""
        # Validation is handled by Pydantic model initialization
        if self.sim_mode:
            print(f"SIMULATOR: Moving stage to {pos}")
            return pos

        # Convert back to nm for asyncroscopy
        target = {
            'x': pos.x * 1000.0,
            'y': pos.y * 1000.0
        }
        if pos.z is not None: target['z'] = pos.z * 1000.0
        if pos.rotation is not None: target['r'] = pos.rotation
        if pos.tilt is not None: target['t'] = pos.tilt

        self._client.send_command(destination, "set_stage", {"pos": target, "relative": relative})
        return self.get_stage_position(destination=destination)

    def acquire_image(self, detector: str = "Ceta", destination: str = "AS") -> ImageResult:
        """Acquire an image from the specified detector."""
        if self.sim_mode:
            print(f"SIMULATOR: Acquiring image from {detector}")
            # Return dummy noise
            dummy_data = np.random.rand(512, 512)
            return ImageResult(data=dummy_data, metadata={"detector": detector, "mode": "simulated"})

        try:
            detector_name = detector.lower().strip()
            encoded = self._client.get_image(detector_name)
            metadata_json, raw_bytes = encoded
            metadata = json.loads(metadata_json)
            img = np.frombuffer(raw_bytes, dtype=metadata["dtype"]).reshape(metadata["shape"])
            metadata.update({"detector": detector_name, "backend": self._backend})
            return ImageResult(data=img, metadata=metadata)
        except Exception as e:
            print(f"Error acquiring image: {e}")
            raise

    def set_beam_position(self, x: float, y: float, destination: str = "AS") -> bool:
        """Set the beam position in nm."""
        if self.sim_mode:
            print(f"SIMULATOR: Setting beam position to ({x}, {y})")
            return True
        
        self._client.send_command(destination, "place_beam", {"x": x, "y": y})
        return True
