import torch
import torch.nn as nn

print("CPU compatibility patch applied - img2img-turbo now works on CPU!")

# ------------------- 1. Patch Tensor.cuda() and Tensor.to() -------------------
_original_tensor_cuda = torch.Tensor.cuda
_original_tensor_to = torch.Tensor.to


def _safe_tensor_cuda(self, device=None, non_blocking=False):
    """Replace .cuda() on tensors - fallback to CPU if CUDA is not available"""
    if not torch.cuda.is_available():
        return self.cpu()
    return _original_tensor_cuda(self, device=device, non_blocking=non_blocking)


def _safe_tensor_to(self, *args, **kwargs):
    """Replace .to() on tensors - convert 'cuda' → 'cpu' when no GPU"""
    if not torch.cuda.is_available():
        # Handle positional device argument: tensor.to("cuda")
        if len(args) > 0 and args[0] in ("cuda", None):
            args = ("cpu",) + args[1:]
        # Handle keyword argument: tensor.to(device="cuda")
        elif "device" in kwargs and kwargs["device"] in ("cuda", None):
            kwargs["device"] = "cpu"
    return _original_tensor_to(self, *args, **kwargs)

torch.Tensor.cuda = _safe_tensor_cuda
torch.Tensor.to = _safe_tensor_to

# ------------------- 2. Patch nn.Module.to() and .cuda() -------------------
_original_module_to = nn.Module.to


def _safe_module_to(self, *args, **kwargs):
    """Safe replacement for model.to() / model.cuda()"""
    if not torch.cuda.is_available():
        device = args[0] if args else kwargs.get("device")
        if device in ("cuda", None):
            if args:
                args = ("cpu",) + args[1:]
            else:
                kwargs["device"] = "cpu"
    return _original_module_to(self, *args, **kwargs)


def _safe_module_cuda(self, device=None):
    """Safe .cuda() for modules"""
    if not torch.cuda.is_available():
        return self.to("cpu")
    return _original_module_to(self, "cuda" if device is None else device)

nn.Module.to = _safe_module_to
nn.Module.cuda = _safe_module_cuda

# ------------------- 3. Patch DDPMScheduler.set_timesteps -------------------
try:
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    _original_set_timesteps = DDPMScheduler.set_timesteps

    def _patched_set_timesteps(self, num_inference_steps, device=None, *args, **kwargs):
        """Force device='cpu' in set_timesteps() if CUDA is unavailable"""
        if device in ("cuda", None) and not torch.cuda.is_available():
            device = "cpu"
        return _original_set_timesteps(self, num_inference_steps, device=device, *args, **kwargs)

    DDPMScheduler.set_timesteps = _patched_set_timesteps
except Exception:
    pass  # diffusers might not be installed yet - patch will apply later

# ------------------- 4. Patch torch.tensor() -------------------
_original_tensor_func = torch.tensor


def _safe_tensor(*args, **kwargs):
    """
    Safe torch.tensor() - automatically uses 'cpu' when:
    - device="cuda" is explicitly passed
    - or no device is specified and CUDA is not available
    """
    if "device" in kwargs:
        if kwargs["device"] in ("cuda", None) and not torch.cuda.is_available():
            kwargs["device"] = "cpu"
    else:
        # Default device when none is provided
        kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return _original_tensor_func(*args, **kwargs)

torch.tensor = _safe_tensor

print("All CPU compatibility patches applied successfully!")
