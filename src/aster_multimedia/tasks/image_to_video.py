"""Image-to-video diffusion task."""

from __future__ import annotations

import logging
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from ..config import ImageToVideoConfig
from ..registry import registry
from .base import SimpleTask

_LOG = logging.getLogger(__name__)

_DIFFUSERS_AVAILABLE = importlib.util.find_spec("diffusers") is not None
_PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None
_IMAGEIO_AVAILABLE = importlib.util.find_spec("imageio") is not None
_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if _DIFFUSERS_AVAILABLE:
    from diffusers import StableVideoDiffusionPipeline  # type: ignore[import]
else:
    StableVideoDiffusionPipeline = None  # type: ignore[assignment]

if _PIL_AVAILABLE:
    from PIL import Image  # type: ignore[import]
else:
    Image = None  # type: ignore[assignment]

if _IMAGEIO_AVAILABLE:
    import imageio  # type: ignore[import]
else:
    imageio = None  # type: ignore[assignment]

if _TORCH_AVAILABLE:
    import torch  # type: ignore[import]
else:
    torch = None  # type: ignore[assignment]


@dataclass
class ImageToVideoTask(SimpleTask):
    """Generate a short video clip from a reference image."""

    config: ImageToVideoConfig

    def run(self) -> Dict[str, Any]:
        output_path = Path(self.config.output_path)
        self.prepare_output_dir(output_path)

        if StableVideoDiffusionPipeline is None or Image is None or imageio is None:
            _LOG.warning(
                "Diffusers/Pillow/imageio missing. Creating a stub slideshow video."
            )
            self._stub_video(output_path)
            return {"output": str(output_path), "model": "stub"}

        if self.config.init_image is None or not Path(self.config.init_image).exists():
            raise ValueError("init_image must exist for image-to-video tasks")

        init_image = Image.open(self.config.init_image).convert("RGB")

        kwargs: Dict[str, Any] = {}
        if self.config.hardware.device != "auto":
            kwargs["device"] = self.config.hardware.device

        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            self.config.model,
            torch_dtype=(torch.float16 if torch and torch.cuda.is_available() else None),
        )
        if torch is not None and torch.cuda.is_available():
            pipeline.enable_model_cpu_offload()

        frames = pipeline(
            image=init_image,
            num_frames=self.config.num_frames,
            fps=self.config.fps,
            motion_bucket_id=self.config.motion_bucket_id,
            cond_aug=self.config.cond_aug,
            prompt=self.config.prompt,
            negative_prompt=self.config.negative_prompt,
            **kwargs,
        ).frames[0]

        self._write_video(frames, output_path)
        _LOG.info("Saved %d frames to %s", len(frames), output_path)
        return {"output": str(output_path), "model": self.config.model}

    def _write_video(self, frames: List[Any], output_path: Path) -> None:
        assert imageio is not None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(output_path, frames, fps=self.config.fps)

    def _stub_video(self, output_path: Path) -> None:
        if imageio is None or Image is None:
            output_path.write_text("Install diffusers + pillow + imageio to enable video generation.\n", encoding="utf-8")
            return
        if self.config.init_image is None or not Path(self.config.init_image).exists():
            output_path.write_text("Provide an init image to synthesize a slideshow.\n", encoding="utf-8")
            return
        frame = Image.open(self.config.init_image).convert("RGB")
        frames = [frame for _ in range(max(1, self.config.num_frames))]
        self._write_video(frames, output_path)


@registry.register("image-to-video", "Animate a reference image into a short clip")
def create_image_to_video_task(config: ImageToVideoConfig) -> ImageToVideoTask:
    return ImageToVideoTask(config=config)
