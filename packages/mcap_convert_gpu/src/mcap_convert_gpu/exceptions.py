"""Custom exceptions for mcap_convert_gpu package."""


class McapConverterError(Exception):
    """Base exception for mcap_convert_gpu package."""

    pass


class ConfigurationError(McapConverterError):
    """Raised when configuration is invalid or incomplete."""

    pass


class McapReadError(McapConverterError):
    """Raised when MCAP file cannot be read or parsed."""

    pass


class DataExtractionError(McapConverterError):
    """Raised when data extraction from MCAP fails."""

    pass


class TimeAlignmentError(McapConverterError):
    """Raised when sensor data time alignment fails."""

    pass


class DatasetWriteError(McapConverterError):
    """Raised when LeRobot dataset writing fails."""

    pass
