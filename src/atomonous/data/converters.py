from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar
from dataclasses import dataclass
from .types import AIFormat, FilePath


class HeuristicMismatchError(Exception):
    """Raised when a converter supports a data type but the content doesn't match its heuristics."""
    pass

@dataclass(frozen=True)
class DataConverter[T](ABC):
    """
    Abstract Base Class for all scientific data converters.
    Handles mapping arbitrary inputs (T_in) to str or PIL.Image.Image.
    """
    input_type: ClassVar[type[T] | tuple[type, ...]]

    @abstractmethod
    def convert(self, data: T) -> AIFormat:
        """
        Converts the input data into either a str or a PIL.Image.Image.
        
        Args:
            data: The input object (Path, Array, Custom class, Dict, etc.)
            
        Returns:
            AIFormat: The data format natively ingestible by the Agent.
        """
        pass

    def can_handle(self, data: Any) -> bool:
        """
        Checks if this converter can handle the given input.
        Default implementation uses isinstance against self.input_type.
        """
        return isinstance(data, self.input_type)


@dataclass(frozen=True)
class FileDataConverter[T](DataConverter[T]):
    """
    Specialized converter for file-based data sources.
    Automates extension checking in can_handle.
    """
    supported_extensions: ClassVar[set[str]]

    def can_handle(self, data: Any) -> bool:
        """
        Validates type first, then checks extension if the data was provided as a path.
        """
        if not super().can_handle(data):
            return False

        # If it's a path/string, it must match the supported extensions
        if isinstance(data, FilePath):
            return Path(data).suffix.lower() in self.supported_extensions

        # If it's the internal type (e.g. pd.DataFrame), super().can_handle was enough
        return True
