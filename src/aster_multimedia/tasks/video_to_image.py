"""Video-to-image extraction task."""

from __future__ import annotations

import logging
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from ..config import VideoToImageConfig
from ..registry import registry
from .base import SimpleTask

_LOG = logging.getLogger(__name__)

_IMAGEIO_AVAILABLE = importlib.util.find_spec("imageio") is not None
_PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None

if _IMAGEIO_AVAILABLE:
    import imageio  # type: ignore[import]
else:
    imageio = None  # type: ignore[assignment]

if _PIL_AVAILABLE:
    from PIL import Image  # type: ignore[import]
else:
    Image = None  # type: ignore[assignment]


@dataclass
class VideoToImageTask(SimpleTask):
    """Extract representative frames from a video."""

    config: VideoToImageConfig

    def run(self) -> Dict[str, List[str]]:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if imageio is None or Image is None:
            _LOG.warning("imageio/Pillow missing. Creating a placeholder frame file.")
            placeholder = output_dir / "frame_stub.txt"
            placeholder.write_text(
                "Install imageio + pillow to extract real frames.\n", encoding="utf-8"
            )
            return {"frames": [str(placeholder)]}

        if not Path(self.config.input_video).exists():
            raise FileNotFoundError(f"Video not found: {self.config.input_video}")

        reader = imageio.get_reader(self.config.input_video)
        saved_frames: List[str] = []
        try:
            for index, frame in enumerate(reader):
                if not self._should_keep(index):
                    continue
                image = Image.fromarray(frame)
                output_path = output_dir / f"frame_{index:05d}.png"
                image.save(output_path)
                saved_frames.append(str(output_path))
                if self.config.max_frames and len(saved_frames) >= self.config.max_frames:
                    break
        finally:
            reader.close()

        if not saved_frames:
            raise RuntimeError("No frames were extracted. Adjust your sampling settings.")

        _LOG.info("Extracted %d frames to %s", len(saved_frames), output_dir)
        return {"frames": saved_frames}

    def _should_keep(self, frame_index: int) -> bool:
        match self.config.sampler:
            case "first-frame":
                return frame_index == self.config.frame_index
            case "every-n-frames":
                return frame_index % max(1, self.config.every_n_frames) == 0
            case "all":
                return True
            case other:
                raise ValueError(f"Unsupported sampler '{other}'")


@registry.register("video-to-image", "Sample frames from a video source")
def create_video_to_image_task(config: VideoToImageConfig) -> VideoToImageTask:
    return VideoToImageTask(config=config)
