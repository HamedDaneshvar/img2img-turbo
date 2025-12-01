from inference_paired import run_inference as _run_inference


def run_inference(*args, **kwargs):
    """Public wrapper for paired inference."""
    return _run_inference(*args, **kwargs)
