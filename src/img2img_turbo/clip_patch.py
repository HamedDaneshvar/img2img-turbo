# CLIP compatibility patch
#
# The original img2img-turbo training script imports the official OpenAI CLIP implementation
# (`import clip`). However, PyPI (and TestPyPI) do not allow direct git dependencies in
# pyproject.toml metadata. To make the package uploadable while keeping the original code
# untouched, we monkey-patch the `clip` module to use `open-clip-torch` (which provides
# near-identical functionality and can load the original OpenAI weights via pretrained='openai').
# This removes the need for `clip @ git+https://github.com/openai/CLIP.git` in dependencies.

import sys
import torch
import open_clip


# Fake clip module to mimic openai/clip API using open_clip
class FakeClipModule:
    @staticmethod
    def load(name, device=None, jit=False, download_root='~/.cache/clip', T=8, verbose=False):
        # Map to open_clip equivalent - use 'openai' pretrained for exact match to original CLIP
        model_name = name.replace('/', '-')  # e.g., "ViT-B/32" → "ViT-B-32"
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained='openai',  # Use original OpenAI weights for best compatibility
            device=device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # Mimic clip's return: (model, preprocess)
        return model, preprocess

    @staticmethod
    def tokenize(texts, context_length=77, truncate=True):
        return open_clip.tokenize(texts, context_length=context_length, truncate=truncate)

# Override sys.modules['clip'] with our fake module
sys.modules['clip'] = FakeClipModule()

print("CLIP patch applied: Using open_clip-torch as backend for 'clip' imports.")
