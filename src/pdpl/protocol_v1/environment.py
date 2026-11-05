from __future__ import annotations
import platform
import sys
from importlib import metadata

PACKAGES = ["torch", "transformers", "accelerate", "huggingface_hub", "bitsandbytes", "langchain",
            "langchain-community", "langchain-text-splitters", "faiss-cpu", "python-docx", "numpy",
            "PyYAML", "scikit-learn", "pandas", "openpyxl", "scipy", "matplotlib"]


def package_versions() -> dict:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def environment_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "packages": package_versions(), "cuda_available": None, "gpu": None}
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except Exception:       # torch not installed
        pass
    return info
