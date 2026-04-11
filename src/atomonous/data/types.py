from pathlib import Path
from PIL.Image import Image

# A type that can be ingested by the AI
AIFormat = str | Image

# A type that represents a filepath
FilePath = str | Path