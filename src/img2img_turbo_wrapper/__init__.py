"""
img2img_turbo_wrapper package.

Public API:
- run_inference: Main inference entry point
"""

# Apply CPU compatibility patches as soon as the package is imported
try:
    import img2img_turbo_wrapper.cpu_patch  # type: ignore
except ImportError:
    pass


from .api import run_inference

__all__ = ["run_inference"]
