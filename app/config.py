import os
from typing import Dict, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class MicroscopeSettings(BaseSettings):
    """
    Centralized configuration for microscope hardware and paths.
    Uses environment variables with prefixes (e.g. MICROSCOPE_STAGE_X_MIN).
    """
    model_config = SettingsConfigDict(
        env_prefix='MICROSCOPE_', 
        env_file='.env', 
        extra='ignore',
        validate_assignment=True
    )

    # Hardware Bounds
    stage_x_min: float = Field(0.0, description="Minimum X coordinate for the stage in microns")
    stage_x_max: float = Field(100000.0, description="Maximum X coordinate for the stage in microns")
    stage_y_min: float = Field(0.0, description="Minimum Y coordinate for the stage in microns")
    stage_y_max: float = Field(100000.0, description="Maximum Y coordinate for the stage in microns")
    
    # Image Settings
    max_image_size: int = Field(4096, description="Maximum width/height for image acquisition")
    
    # Paths and Networks
    server_host: str = Field("127.0.0.1", description="Hostname for the asyncroscopy server")
    server_port: int = Field(9000, description="Port for the central asyncroscopy server")
    mcp_url: str = Field("http://localhost:8000/mcp", description="URL for the MCP server")
    
    # AutoScript / Hardware Paths
    autoscript_path: str = Field(
        "/Users/austin/Desktop/Projects/autoscript_tem_microscope_client", 
        description="Local path to the autoscript_tem_microscope_client library"
    )
    instrument_host: str = Field("10.46.217.241", description="IP/Hostname of the microscope instrument PC")
    instrument_port: int = Field(9001, description="Port the AutoScript server is listening on (AutoScript default is often 9007, check your setup)")

    autoscript_port: int = Field(9095, description="Port that the autoscript server is running on")

    # Simulation Mode
    sim_mode: bool = Field(False, description="Enable dry-run/simulator mode by default")

    # Agent Approval Control
    agent_autorun: bool = Field(False, description="If True, allow agent to execute tools without requiring manual approval. If False, each tool call requires user confirmation.")

    # Agent Execution Control
    agent_max_steps: int = Field(10, description="Maximum number of tool calls the agent can make in a single run to prevent infinite loops.")

    # Artifact & Memory Storage
    artifacts_dir: str = Field("./artifacts", description="Base directory for saving session artifacts (workflows, images, chat history, execution steps).")

    # Other stuff
    hf_cache_dir: str = Field("~/.cache/huggingface", description="To configure where Huggingface will locally store data, models, etc.")

    @field_validator('hf_cache_dir')
    @classmethod
    def sync_hf_home(cls, v: str) -> str:
        """didSet logic: Sync HF_HOME whenever hf_cache_dir changes."""
        os.environ["HF_HOME"] = os.path.expanduser(v)
        return v

# Global configuration instance
settings = MicroscopeSettings()
