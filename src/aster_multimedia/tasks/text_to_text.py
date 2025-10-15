"""Text-to-text task implementation."""

from __future__ import annotations

import logging
import importlib.util
from dataclasses import dataclass
from typing import Any, Dict

from ..config import TextToTextConfig
from ..registry import registry
from .base import SimpleTask


_LOG = logging.getLogger(__name__)

_TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None
if _TRANSFORMERS_AVAILABLE:
    from transformers import pipeline  # type: ignore[import]
else:
    pipeline = None  # type: ignore[assignment]


@dataclass
class TextToTextTask(SimpleTask):
    """Generate text from a prompt using a local model."""

    config: TextToTextConfig

    def run(self) -> Dict[str, Any]:
        if pipeline is None:
            _LOG.warning(
                "Transformers is not installed. Falling back to a rule-based stub."
            )
            generated = self._stub_generate()
            return {"text": generated, "model": "stub"}

        kwargs: Dict[str, Any] = {}
        if self.config.hardware.device != "auto":
            kwargs["device"] = self.config.hardware.device

        generator = pipeline(
            "text-generation",
            model=self.config.model,
            torch_dtype=None if self.config.hardware.dtype == "auto" else self.config.hardware.dtype,
            **kwargs,
        )
        outputs = generator(
            self.config.prompt,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=self.config.temperature > 0,
        )
        text = outputs[0]["generated_text"]
        _LOG.info("Generated %d characters", len(text))
        return {"text": text, "model": self.config.model}

    def _stub_generate(self) -> str:
        prompt = self.config.prompt.strip()
        if not prompt:
            return "No prompt supplied; install transformers for full functionality."
        sentences = [part.strip() for part in prompt.split(".") if part.strip()]
        reversed_sentences = " ".join(reversed(sentences))
        return f"[stub:{self.config.max_new_tokens}] {reversed_sentences or prompt}"


@registry.register("text-to-text", "Generate text with a local language model")
def create_text_to_text_task(config: TextToTextConfig) -> TextToTextTask:
    return TextToTextTask(config=config)
