"""
img2img_turbo_wrapper package.

Public API:
- run_inference_paired: Main inference paired entry point
"""

# Apply CPU compatibility patches as soon as the package is imported
try:
    import img2img_turbo.cpu_patch  # type: ignore
except ImportError:
    pass


# Apply CLIP patch (must be before any 'import clip' in code)
try:
    # Note: We didn't test this.
    import img2img_turbo.clip_patch
except ImportError:
    pass


from .api import run_inference_paired

__all__ = ["run_inference_paired"]
__version__ = "0.0.1"
