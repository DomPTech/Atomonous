from typing import List, Dict, Type, Optional, Any
from .converters import DataConverter, HeuristicMismatchError, AIFormat
from .default_converters.image_converters import TiffConverter, NumpyImageConverter
from .default_converters.text_converters import CsvConverter, Hdf5SummaryConverter, DictConverter
from .default_converters.mcp_converter import MCPJsonConverter

class ConverterFactory:
    """
    Registry for managing and selecting DataConverters.
    Converts arbitrary objects into AI-ingestible formats.
    """

    def __init__(self, converters: Optional[List[DataConverter]] = None, register_default: bool = False):
        self._converters: List[DataConverter] = []
        
        if register_default:
            self.register_default_converters()
            
        if converters:
            for converter in converters:
                self.register_converter(converter)

    def register_default_converters(self):
        """
        Populate the registry with the default scientific data converters provided by Atomonous.
        """
        self.register_converter(DictConverter())
        self.register_converter(Hdf5SummaryConverter())
        self.register_converter(CsvConverter())
        self.register_converter(NumpyImageConverter())
        self.register_converter(TiffConverter())
        self.register_converter(MCPJsonConverter())

    def register_converter(self, converter: DataConverter):
        """
        Adds a new converter to the front of the registry.
        Providing LIFO priority: newer/custom converters override defaults.
        """
        self._converters.insert(0, converter)

    def convert(self, data: Any) -> AIFormat:
        """
        Converts any supported object into either a str or a PIL.Image.Image.
        
        Args:
            data: The object to convert (Path, Array, Dict, Custom instance, MCP JSON, etc.)
            
        Returns:
            AIFormat: The resulting ingestible data.
            
        Raises:
            ValueError: If no suitable converter is found or all fail heuristics.
        """
        
        # Filter candidates that can handle this data
        candidates = [c for c in self._converters if c.can_handle(data)]

        if not candidates:
            type_name = type(data).__name__
            raise ValueError(f"No converter found for input type '{type_name}'")

        # Try each candidate until one succeeds
        last_error = None
        for converter in candidates:
            try:
                return converter.convert(data)
            except HeuristicMismatchError as e:
                last_error = e
                continue
            except Exception as e:
                last_error = e
                continue

        raise ValueError(f"All matching converters for input failed. Last error: {last_error}")
