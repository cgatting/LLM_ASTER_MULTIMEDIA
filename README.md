# ASTER Multimedia Toolkit

ASTER (All-in-one Streaming & Transformation Engine for Research) is a lightweight framework for orchestrating local multimodal AI workflows.  It focuses on providing a common interface for running text, image, and video generation/transformation pipelines with open-source models that can operate offline.

## Key Capabilities

- **Task registry** with pluggable pipelines.
- **YAML-driven configuration** so you can encode prompts, sampler parameters, and hardware placement in version-controlled files.
- **Local-first**: all integrations target open-source projects that support on-device inference (Diffusers, Transformers, Imageio, OpenCV, etc.).
- **CLI** for ad-hoc experimentation (`aster run text-to-text ...`).
- **Python API** for composing tasks programmatically.

The repository intentionally ships with minimal stub implementations so that it works without heavyweight dependencies.  When you install the optional extras (`pip install -e .[image,text,video]`) the tasks automatically switch to the real model-backed implementations.

## Project Layout

```
├── configs/
│   └── sample.yaml        # Example configuration file showing each task type
├── src/
│   └── aster_multimedia/
│       ├── cli.py         # Click-based command line entry point
│       ├── config.py      # Config parsing + validation helpers
│       ├── registry.py    # Task discovery and registration
│       └── tasks/
│           ├── base.py            # Base class definitions
│           ├── text_to_text.py    # LLM task (Transformers pipeline)
│           ├── image_to_image.py  # Image to image diffusion
│           ├── image_to_video.py  # Uses latent consistency / video diffusers
│           └── video_to_image.py  # Frame extraction + img2img per frame
└── pyproject.toml
```

## Quick Start

1. Create and activate a Python 3.10+ virtual environment.
2. Install the core package:
   ```bash
   pip install -e .
   ```
3. For full functionality, include extras for the modalities you need:
   ```bash
   pip install -e .[text,image,video]
   ```
4. Run a task using the sample configuration:
   ```bash
   aster run --config configs/sample.yaml text-to-text
   ```

## Configuration

Each task section inside a YAML file mirrors the dataclass defined in `aster_multimedia.config`.  You can also override arguments on the CLI.  The sample configuration contains documented defaults for all supported pipelines.

## Extending

- Add a new task by subclassing `Task` and registering it via `@registry.register("your-task")`.
- Use dependency groups in `pyproject.toml` to capture optional libraries.
- Provide hardware-aware defaults by inspecting `torch.cuda.is_available()` and `torch.backends.mps.is_available()` before launching models.

## Disclaimer

The heavy-weight machine learning models are **not** bundled with this repository.  Refer to the docstrings in each task for recommended model identifiers from Hugging Face.
