import json
import re
import base64
import io
from typing import List, Dict, Any, Type

from PIL import Image

import numpy as np

from ..converters import DataConverter, HeuristicMismatchError
from ..types import AIFormat

class MCPJsonConverter(DataConverter[dict | str]):
    """
    De-serializes JSON format sent over by the asyncroscopy MCP Server.
    """

    input_type = (dict, str)

    def can_handle(self, data: Any) -> bool:
        if not super().can_handle(data):
            return False
            
        if isinstance(data, dict):
            return "payload" in data and "metadata" in data
        if isinstance(data, str):
            # Check if it looks like JSON
            try:
                parsed = self._get_json(data)
                return "payload" in parsed and "metadata" in parsed
            except ValueError:
                return False
        return False
    
    def _get_json(self, raw: str) -> dict:
        match = re.search(r'\{.*\}', raw, re.DOTALL)

        if match:
            json_str = match.group(0)
            data = json.loads(json_str)
            return data
        else:
            raise ValueError("No JSON found")

    def convert(self, data: dict | str) -> AIFormat:
        if isinstance(data, str):
            data_dict = self._get_json(data)
        else:
            data_dict = data

        payload = data_dict.get("payload")
        metadata_raw = data_dict.get("metadata", "{}")
        encoding = data_dict.get("encoding", "base64")

        if not payload:
            raise ValueError("MCP data missing 'payload'.")

        # MCP uses base64 for transporting binary data
        if encoding == "base64":
            try:
                decoded_bytes = base64.b64decode(payload)
            except Exception as e:
                raise ValueError(f"Failed to decode base64 payload: {e}")
        else:
            decoded_bytes = payload.encode("utf-8") if isinstance(payload, str) else payload

        meta = metadata_raw if isinstance(metadata_raw, dict) else json.loads(metadata_raw)

        # Heuristics for Image vs Text
        is_image = False
        if any(key in meta for key in ["dims", "shape", "width", "height"]):
            is_image = True
        if meta.get("type") in ["image", "Image", "IMAGE"]:
            is_image = True
        if meta.get("format") in ["png", "jpg", "jpeg", "tiff"]:
            is_image = True

        if is_image:
            try:
                dtype = meta.get("dtype", "float32")
                shape = meta.get("shape")
                
                if shape:
                    image_array = np.frombuffer(decoded_bytes, dtype=dtype).reshape(shape)
                    image_array = image_array.T
                    
                    # Normalize for AI vision (0-255 uint8)
                    img_min, img_max = image_array.min(), image_array.max()
                    if img_max > img_min:
                        normalized = (image_array - img_min) / (img_max - img_min) * 255
                    else:
                        normalized = image_array * 0
                    
                    return Image.fromarray(normalized.astype(np.uint8))
            except Exception:
                # Fallback to PIL.Image.open for standard formats (PNG/JPG)
                try:
                    return Image.open(io.BytesIO(decoded_bytes))
                except Exception:
                    pass
        
        # Default to string
        try:
            return decoded_bytes.decode("utf-8")
        except:
            return str(decoded_bytes)