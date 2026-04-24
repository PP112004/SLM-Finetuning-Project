"""GPU environment sanity check for PS06 training.

Run this as the very first thing on a fresh H100/A100 rental to confirm:
- CUDA + bf16 are available
- transformers / trl / peft / accelerate / datasets versions are compatible
- HF cache path is writable
- optional: unsloth + bitsandbytes import cleanly

Exit code 0 if all mandatory checks pass, 1 otherwise.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


MANDATORY = [
    ("torch", None),
    ("transformers", "4.44"),
    ("accelerate", "0.34"),
    ("datasets", "2.21"),
    ("peft", "0.13"),
    ("trl", "0.12"),
    ("safetensors", None),
    ("sentencepiece", None),
    ("lm_eval", None),
]

OPTIONAL = [
    ("unsloth", None),
    ("bitsandbytes", None),
    ("wandb", None),
]


def _ver(mod) -> str:
    return getattr(mod, "__version__", "?")


def _version_ge(actual: str, required: str) -> bool:
    def _parts(v: str) -> list[int]:
        out: list[int] = []
        for part in v.split("+")[0].split("."):
            try:
                out.append(int(part))
            except ValueError:
                break
        return out

    a, r = _parts(actual), _parts(required)
    for i in range(max(len(a), len(r))):
        ai = a[i] if i < len(a) else 0
        ri = r[i] if i < len(r) else 0
        if ai != ri:
            return ai > ri
    return True


def check_module(name: str, min_version: str | None, required: bool) -> bool:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        tag = "FAIL" if required else "skip"
        print(f"  [{tag}] {name}: import error -> {type(exc).__name__}: {exc}")
        return not required
    actual = _ver(mod)
    ok = True
    if min_version is not None:
        ok = _version_ge(actual, min_version)
    tag = "ok  " if ok else "FAIL"
    suffix = f" (need >= {min_version})" if min_version and not ok else ""
    print(f"  [{tag}] {name} {actual}{suffix}")
    return ok or not required


def check_cuda() -> bool:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] torch import: {exc}")
        return False

    print(f"  torch {torch.__version__}  cuda_build={torch.version.cuda}")
    if not torch.cuda.is_available():
        print("  [FAIL] torch.cuda.is_available() is False")
        return False
    n = torch.cuda.device_count()
    print(f"  [ok  ] CUDA available, devices={n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / (1024**3)
        print(f"         [{i}] {props.name}  {mem_gb:.1f} GB  sm={props.major}.{props.minor}")

    bf16_ok = torch.cuda.is_bf16_supported()
    print(f"  [{'ok  ' if bf16_ok else 'warn'}] bf16 supported = {bf16_ok}")
    return True


def check_hf_cache() -> bool:
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    try:
        cache.mkdir(parents=True, exist_ok=True)
        test = cache / ".writecheck"
        test.write_text("ok")
        test.unlink()
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] HF cache not writable at {cache}: {exc}")
        return False
    print(f"  [ok  ] HF cache writable: {cache}")
    return True


def main() -> int:
    print("=== GPU env check ===")
    ok = True

    print("\n[CUDA / torch]")
    ok &= check_cuda()

    print("\n[HF cache]")
    ok &= check_hf_cache()

    print("\n[Mandatory libraries]")
    for name, min_ver in MANDATORY:
        ok &= check_module(name, min_ver, required=True)

    print("\n[Optional libraries]")
    for name, min_ver in OPTIONAL:
        check_module(name, min_ver, required=False)

    print("\n=== Summary ===")
    print("PASS" if ok else "FAIL — fix the FAIL items above before training")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
