"""Helpers for REASAN-specific Omniverse Kit defaults."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path


def apply_default_reasan_kit_args(args_cli: Namespace, script_file: str) -> None:
    """Add REASAN's local Isaac asset mirror and texture cache to ``--kit_args``.

    Explicit user-provided Kit settings are preserved. The defaults are appended only
    when the same Kit setting key is not already present.
    """

    training_dir = Path(script_file).resolve().parents[1]
    asset_root = training_dir / "assets" / "omniverse" / "Assets" / "Isaac" / "4.5"
    texture_cache = training_dir / "assets" / "cache" / "texturecache"
    texture_cache.mkdir(parents=True, exist_ok=True)

    defaults = {
        "/persistent/isaac/asset_root/cloud": asset_root,
        "/rtx-transient/resourcemanager/localTextureCachePath": texture_cache,
    }

    kit_args = (args_cli.kit_args or "").strip()
    kit_arg_parts = [kit_args] if kit_args else []
    for key, value in defaults.items():
        setting_prefix = f"--{key}="
        if setting_prefix not in kit_args:
            kit_arg_parts.append(f"{setting_prefix}{value}")

    args_cli.kit_args = " ".join(kit_arg_parts)
