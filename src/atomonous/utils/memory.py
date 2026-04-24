"""
Minimal session-based artifact and memory management system.
Saves workflow definitions, diagrams, captured images, and execution steps to dated folders.
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
from PIL import Image

class SessionMemory:
    """
    Manages a dated session folder for storing artifacts: workflow YAML/PNG, captured NPY images, and execution steps.
    """

    def __init__(self, artifacts_base_dir: str, session_name: str = ""):
        """
        Initialize a new session memory instance.
        
        Args:
            artifacts_base_dir: Base directory where session folders will be created.
            session_name: Optional slug/description for the session (e.g., "beam-calibration").
                         If empty, only timestamp is used.
        """
        self.artifacts_base_dir = Path(artifacts_base_dir)
        self.artifacts_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Make session folder
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if session_name and isinstance(session_name, str):
            # Sanitize session name
            session_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_name.strip()).lower()[:50]
            folder_name = f"{timestamp}_{session_name}"
        else:
            folder_name = timestamp
        
        self.session_dir = self.artifacts_base_dir / folder_name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        self.workflow_yaml_path: Optional[Path] = None
        self.workflow_png_path: Optional[Path] = None
        self.execution_steps_path = self.session_dir / "execution_steps.json"
        
        print(f"[SessionMemory] Created session: {self.session_dir}")

    def save_workflow(self, yaml_path: str, png_path: Optional[str] = None) -> None:
        """
        Save workflow YAML and optional PNG diagram to session folder.
        
        Args:
            yaml_path: Absolute path to the source YAML file.
            png_path: Absolute path to the source PNG diagram (optional).
        """
        yaml_source = Path(yaml_path).resolve()
        if yaml_source.exists():
            dest_yaml = (self.session_dir / yaml_source.name).resolve()
            if yaml_source != dest_yaml:
                shutil.copy2(yaml_source, dest_yaml)
                print(f"[SessionMemory] Saved workflow YAML: {dest_yaml}")
            else:
                print(f"[SessionMemory] Workflow YAML already in session dir: {dest_yaml}")
            self.workflow_yaml_path = dest_yaml
        else:
            print(f"[SessionMemory] Warning: YAML file not found: {yaml_path}")
        
        if png_path:
            png_source = Path(png_path).resolve()
            if png_source.exists():
                dest_png = (self.session_dir / png_source.name).resolve()
                if png_source != dest_png:
                    shutil.copy2(png_source, dest_png)
                    print(f"[SessionMemory] Saved workflow PNG: {dest_png}")
                else:
                    print(f"[SessionMemory] Workflow PNG already in session dir: {dest_png}")
                self.workflow_png_path = dest_png

    def save_execution_steps(
        self,
        history: List[str],
        errors: List[str],
        metrics: Dict[str, float],
        summary: str = ""
    ) -> None:
        """
        Save workflow execution steps and state to JSON file.
        
        Args:
            history: List of execution history entries.
            errors: List of error messages.
            metrics: Dictionary of execution metrics.
            summary: Optional summary of the execution.
        """
        execution_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "history": history,
            "errors": errors,
            "metrics": metrics,
        }
        
        with open(self.execution_steps_path, "w") as f:
            json.dump(execution_data, f, indent=2)
        
        print(f"[SessionMemory] Saved execution steps: {self.execution_steps_path}")

    def save_image(self, npy_path: str, description: str = "") -> str:
        """
        Save a NumPy array (.npy) image to the session folder.
        
        Args:
            npy_path: Absolute path to the source NPY file.
            description: Optional description for the image (used in filename).
        
        Returns:
            Absolute path to the saved image in the session folder.
        """
        source = Path(npy_path).resolve()
        if not source.exists():
            print(f"[SessionMemory] Warning: Image file not found: {npy_path}")
            return npy_path
        
        # Generate destination filename
        if description:
            # Sanitize description
            desc_clean = "".join(c if c.isalnum() or c in "-_" else "_" for c in description.strip()).lower()[:40]
            dest_name = f"image_{desc_clean}_{source.stem}.npy"
        else:
            dest_name = source.name
        
        dest_path = (self.session_dir / dest_name).resolve()
        if source != dest_path:
            shutil.copy2(source, dest_path)
        print(f"[SessionMemory] Saved image: {dest_path}")
        
        return str(dest_path)

    def save_pil_image(self, image: Image.Image, description: str = "") -> str:
        """
        Saves a PIL Image to the session directory as a PNG.
        
        Args:
            image: PIL Image object.
            description: Optional description for the image (used in filename).
            
        Returns:
            Absolute path to the saved image in the session folder.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if description:
            desc_clean = "".join(c if c.isalnum() or c in "-_" else "_" for c in description.strip()).lower()[:40]
            filename = f"image_{desc_clean}_{timestamp}.png"
        else:
            filename = f"image_{timestamp}.png"
            
        dest_path = (self.session_dir / filename).resolve()
        image.save(dest_path)
        print(f"[SessionMemory] Saved PIL image: {dest_path}")
        
        return str(dest_path)

    def get_session_dir(self) -> Path:
        """Get the session directory path."""
        return self.session_dir

    def list_artifacts(self) -> Dict[str, List[str]]:
        """
        List all artifacts in the session folder, grouped by type.
        
        Returns:
            Dictionary with keys: 'yaml', 'png', 'images', 'json', 'other'
        """
        artifacts = {
            "yaml": [],
            "png": [],
            "images": [],
            "json": [],
            "other": [],
        }
        
        if not self.session_dir.exists():
            return artifacts
        
        for item in self.session_dir.iterdir():
            if item.is_file():
                if item.suffix == ".yaml" or item.suffix == ".yml":
                    artifacts["yaml"].append(item.name)
                elif item.suffix == ".png":
                    artifacts["png"].append(item.name)
                elif item.suffix == ".npy":
                    artifacts["images"].append(item.name)
                elif item.suffix == ".json":
                    artifacts["json"].append(item.name)
                else:
                    artifacts["other"].append(item.name)
        
        return artifacts
