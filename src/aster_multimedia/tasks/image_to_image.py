"""Image-to-image diffusion task."""

from __future__ import annotations

import logging
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from ..config import ImageToImageConfig
from ..registry import registry
from .base import SimpleTask

_LOG = logging.getLogger(__name__)

_DIFFUSERS_AVAILABLE = importlib.util.find_spec("diffusers") is not None
_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
_PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None

if _DIFFUSERS_AVAILABLE:
    from diffusers import StableDiffusionImg2ImgPipeline  # type: ignore[import]
else:
    StableDiffusionImg2ImgPipeline = None  # type: ignore[assignment]

if _PIL_AVAILABLE:
    from PIL import Image  # type: ignore[import]
else:
    Image = None  # type: ignore[assignment]

if _TORCH_AVAILABLE:
    import torch  # type: ignore[import]
else:
    torch = None  # type: ignore[assignment]


@dataclass
class ImageToImageTask(SimpleTask):
    """Apply an image-to-image diffusion pipeline."""

    config: ImageToImageConfig

    def run(self) -> Dict[str, Any]:
        output_path = Path(self.config.output_path)
        self.prepare_output_dir(output_path)

        if StableDiffusionImg2ImgPipeline is None or Image is None:
            _LOG.warning(
                "Diffusers or Pillow is missing. Copying the input image as a stub output."
            )
            self._stub_copy(output_path)
            return {"output": str(output_path), "model": "stub"}

        if self.config.init_image is None:
            raise ValueError("An init_image path is required for image-to-image tasks")

        image = Image.open(self.config.init_image).convert("RGB")
        torch_dtype = None
        if self.config.hardware.dtype != "auto" and torch is not None:
            torch_dtype = getattr(torch, self.config.hardware.dtype)
        elif torch is not None and torch.cuda.is_available():
            torch_dtype = torch.float16

        kwargs: Dict[str, Any] = {"torch_dtype": torch_dtype}
        if self.config.hardware.device != "auto":
            kwargs["device"] = self.config.hardware.device

        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.config.model,
            **{k: v for k, v in kwargs.items() if v is not None},
        )
        if torch is not None and torch.cuda.is_available():
            pipeline = pipeline.to("cuda")

        generated = pipeline(
            prompt=self.config.prompt,
            image=image,
            strength=self.config.strength,
            guidance_scale=self.config.guidance_scale,
            negative_prompt=self.config.negative_prompt,
        ).images[0]
        generated.save(output_path)
        _LOG.info("Saved generated image to %s", output_path)
        return {"output": str(output_path), "model": self.config.model}

    def _stub_copy(self, output_path: Path) -> None:
        if self.config.init_image is None or not Path(self.config.init_image).exists():
            output_path.write_text("Install diffusers + pillow to enable real generation.\n", encoding="utf-8")
            return
        output_path.write_bytes(Path(self.config.init_image).read_bytes())


@registry.register("image-to-image", "Transform an image using diffusion models")
def create_image_to_image_task(config: ImageToImageConfig) -> ImageToImageTask:
    return ImageToImageTask(config=config)
