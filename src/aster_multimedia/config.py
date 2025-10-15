"""Configuration utilities for ASTER."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class HardwareConfig:
    """Hardware selection for a task."""

    device: str = "auto"
    dtype: str = "auto"


@dataclass
class BaseTaskConfig:
    """Common configuration shared by all tasks."""

    task: str
    name: str = "default"
    hardware: HardwareConfig = field(default_factory=HardwareConfig)


@dataclass
class TextToTextConfig(BaseTaskConfig):
    """Configuration for text generation tasks."""

    prompt: str = ""
    model: str = "microsoft/phi-2"
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95


@dataclass
class ImageToImageConfig(BaseTaskConfig):
    """Configuration for image to image diffusion."""

    model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    guidance_scale: float = 7.0
    strength: float = 0.6
    prompt: str = ""
    negative_prompt: Optional[str] = None
    init_image: Optional[Path] = None
    output_path: Path = Path("outputs/image_to_image.png")


@dataclass
class ImageToVideoConfig(BaseTaskConfig):
    """Configuration for image to video generation."""

    model: str = "stabilityai/stable-video-diffusion-img2vid"
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    num_frames: int = 25
    fps: int = 7
    motion_bucket_id: int = 127
    cond_aug: float = 0.02
    init_image: Path = Path("inputs/reference.png")
    output_path: Path = Path("outputs/image_to_video.mp4")


@dataclass
class VideoToImageConfig(BaseTaskConfig):
    """Configuration for video to image conversion."""

    sampler: str = "first-frame"
    frame_index: int = 0
    every_n_frames: int = 10
    max_frames: Optional[int] = 30
    init_frame_prompt: Optional[str] = None
    model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    input_video: Path = Path("inputs/video.mp4")
    output_dir: Path = Path("outputs/video_frames")


CONFIG_MAP: Dict[str, type[BaseTaskConfig]] = {
    "text-to-text": TextToTextConfig,
    "image-to-image": ImageToImageConfig,
    "image-to-video": ImageToVideoConfig,
    "video-to-image": VideoToImageConfig,
}


def load_config(path: Path) -> Dict[str, BaseTaskConfig]:
    """Load YAML configuration file and return config objects keyed by name."""

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    configs: Dict[str, BaseTaskConfig] = {}
    for name, params in raw.items():
        task_type = params.get("task")
        if not task_type:
            raise ValueError(f"Configuration '{name}' is missing the 'task' field")
        if task_type not in CONFIG_MAP:
            available = ", ".join(sorted(CONFIG_MAP))
            raise ValueError(f"Unsupported task '{task_type}'. Available: {available}")
        config_cls = CONFIG_MAP[task_type]
        payload = {k: v for k, v in params.items() if k != "task"}
        if isinstance(payload.get("hardware"), dict):
            payload["hardware"] = HardwareConfig(**payload["hardware"])
        configs[name] = config_cls(
            task=task_type,
            name=name,
            **payload,
        )
    return configs


def merge_cli_overrides(config: BaseTaskConfig, overrides: Dict[str, Any]) -> BaseTaskConfig:
    """Return a new config with CLI overrides applied."""

    data = config.__dict__ | overrides
    if isinstance(data.get("hardware"), dict):
        data["hardware"] = HardwareConfig(**data["hardware"])
    config_cls = type(config)
    return config_cls(**data)


def list_tasks() -> List[str]:
    return sorted(CONFIG_MAP)
