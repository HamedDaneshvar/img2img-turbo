from inference_paired import run_inference_paired as _run_inference_paired


def run_inference_paired(*args, **kwargs):
    """Public wrapper for paired inference."""
    return _run_inference_paired(*args, **kwargs)
