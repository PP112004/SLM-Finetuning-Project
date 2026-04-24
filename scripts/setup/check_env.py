from __future__ import annotations

import importlib
import os
import platform
import sys


CORE_PACKAGES = [
    "torch",
    "transformers",
    "datasets",
    "evaluate",
    "accelerate",
    "peft",
    "trl",
]

APPLE_SILICON_PACKAGES = [
    "mlx",
    "mlx_lm",
]


def main() -> int:
    mps_available = False

    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Headless/session: {os.environ.get('TERM_PROGRAM', 'unknown')}")
    print()

    for name in CORE_PACKAGES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "unknown")
            print(f"[ok] {name} {version}")
        except Exception as exc:
            print(f"[missing] {name}: {exc}")

    print()
    try:
        import torch

        mps_available = torch.backends.mps.is_available()
        print(f"torch.backends.mps.is_available(): {mps_available}")
        print(f"torch.backends.mps.is_built(): {torch.backends.mps.is_built()}")
    except Exception as exc:
        print(f"torch check failed: {exc}")

    print()
    print("Apple Silicon optional packages:")
    for name in APPLE_SILICON_PACKAGES:
        if name == "mlx_lm" and not mps_available:
            print("[optional-skipped] mlx_lm: Metal/MPS is unavailable in this session")
            continue
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "unknown")
            print(f"[ok] {name} {version}")
        except Exception as exc:
            print(f"[optional-unavailable] {name}: {exc}")

    print()
    print("Note: MLX/MPS may be unavailable inside headless or sandboxed sessions.")
    print("Run this script from your normal macOS Terminal for the real local GPU check.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
