import json
import base64
import io
from typing import List, Dict, Any, Type
from PIL import Image
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
            s = data.strip()
            return s.startswith('{') and s.endswith('}') and '"payload":' in s
        return False

    def convert(self, data: dict | str) -> AIFormat:
        if isinstance(data, str):
            try:
                data_dict = json.loads(data)
            except Exception as e:
                raise HeuristicMismatchError(f"Failed to parse MCP JSON string: {e}")
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

        try:
            meta = json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
        except:
             meta = {}

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
                return Image.open(io.BytesIO(decoded_bytes))
            except Exception as e:
                # Fallback to string if image open fails
                pass
        
        # Default to string
        try:
            return decoded_bytes.decode("utf-8")
        except:
            return str(decoded_bytes)