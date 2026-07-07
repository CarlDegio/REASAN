#!/usr/bin/env python3
"""Download REASAN runtime assets from the Isaac Sim public asset bucket.

The script mirrors files under:

    training/assets/omniverse/Assets/Isaac/4.5

That path can then be used as Isaac's local asset root:

    --kit_args "--/persistent/isaac/asset_root/cloud=/abs/path/to/training/assets/omniverse/Assets/Isaac/4.5"
"""

from __future__ import annotations

import argparse
import os
import posixpath
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path


DEFAULT_REMOTE_ROOT = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5"
DEFAULT_LOCAL_ROOT = Path(__file__).resolve().parent / "omniverse" / "Assets" / "Isaac" / "4.5"

DEFAULT_ASSETS = (
    # Unitree Go2 is used by all REASAN Go2 tasks through UNITREE_GO2_CFG.
    "Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd",
    # REASAN play locomotion environment sky texture.
    "Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
    # Terrain/object materials referenced by REASAN env configs.
    "NVIDIA/Materials/Base/Architecture/Shingles_01.mdl",
    "NVIDIA/Materials/Base/Wood/Ash_Planks.mdl",
    # Debug visualization markers used by IsaacLab/REASAN play environments.
    "Isaac/Props/UIElements/arrow_x.usd",
    "Isaac/Props/UIElements/frame_prim.usd",
)

REFERENCE_EXTENSIONS = (
    "usd",
    "usda",
    "usdc",
    "mdl",
    "mtlx",
    "png",
    "jpg",
    "jpeg",
    "hdr",
    "exr",
    "tif",
    "tiff",
    "bmp",
    "obj",
    "stl",
    "dae",
    "fbx",
    "bin",
    "pt",
    "onnx",
)

LOCAL_PATH_PREFIXES = (
    "home/",
    "mnt/",
    "opt/",
    "tmp/",
    "var/",
    "Users/",
    "Volumes/",
)

REFERENCE_RE = re.compile(
    rb"""(?P<path>(?:[A-Za-z0-9_.%+\-]+/|\./|\../)+[A-Za-z0-9_.%+\-]+"""
    + rb"""\.(?:"""
    + b"|".join(ext.encode("ascii") for ext in REFERENCE_EXTENSIONS)
    + rb"""))"""
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download REASAN online assets into a local Isaac asset mirror.")
    parser.add_argument(
        "--remote-root",
        default=DEFAULT_REMOTE_ROOT,
        help=f"Remote Isaac asset root. Default: {DEFAULT_REMOTE_ROOT}",
    )
    parser.add_argument(
        "--local-root",
        type=Path,
        default=DEFAULT_LOCAL_ROOT,
        help=f"Local Isaac asset root. Default: {DEFAULT_LOCAL_ROOT}",
    )
    parser.add_argument(
        "--asset",
        action="append",
        default=[],
        help=(
            "Extra asset path relative to the asset root, for example "
            "'Isaac/Robots/ANYbotics/anymal_instanceable.usd'. Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--proxy",
        default=None,
        help="HTTP proxy URL for downloads. If omitted, curl/urllib use http_proxy/https_proxy from the shell.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist.")
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only download the manifest entries, without scanning downloaded files for relative references.",
    )
    parser.add_argument(
        "--use-urllib",
        action="store_true",
        help="Use Python urllib instead of curl. By default curl is used when available.",
    )
    parser.add_argument(
        "--print-kit-args",
        action="store_true",
        help="Print the --kit_args value needed to point Isaac/IsaacLab at the local mirror.",
    )
    return parser.parse_args()


def to_remote_url(remote_root: str, asset_path: str) -> str:
    return f"{remote_root.rstrip('/')}/{asset_path.lstrip('/')}"


def to_local_path(local_root: Path, asset_path: str) -> Path:
    normalized = asset_path.lstrip("/")
    if normalized.startswith("../") or "/../" in normalized:
        raise ValueError(f"Refusing to write outside local root: {asset_path}")
    return local_root / normalized


def normalize_asset_path(path: str) -> str:
    decoded = urllib.parse.unquote(path)
    decoded = decoded.replace("\\", "/")
    while decoded.startswith("./"):
        decoded = decoded[2:]
    normalized = posixpath.normpath(decoded).lstrip("/")
    if normalized == ".":
        return ""
    return normalized


def remote_url_to_asset_path(remote_root: str, url: str) -> str | None:
    remote_root = remote_root.rstrip("/") + "/"
    parsed_url = urllib.parse.urlparse(url)
    parsed_root = urllib.parse.urlparse(remote_root)
    if parsed_url.scheme and parsed_url.netloc:
        if parsed_url.scheme != parsed_root.scheme or parsed_url.netloc != parsed_root.netloc:
            return None
        root_path = parsed_root.path.rstrip("/") + "/"
        if not parsed_url.path.startswith(root_path):
            return None
        return normalize_asset_path(parsed_url.path[len(root_path) :])
    return None


def resolve_reference(current_asset: str, reference: str, remote_root: str) -> str | None:
    reference = reference.strip().strip("@\"'<>[](){};,")
    if not reference:
        return None
    if reference.startswith(LOCAL_PATH_PREFIXES):
        return None
    if reference.startswith(("omniverse://", "omniverse:")):
        return None
    if reference.startswith(("http://", "https://")):
        return remote_url_to_asset_path(remote_root, reference)
    if reference.startswith("/"):
        return normalize_asset_path(reference)
    current_dir = str(Path(current_asset).parent)
    if current_dir == ".":
        current_dir = ""
    return normalize_asset_path(str(Path(current_dir) / reference))


def extract_references(path: Path, current_asset: str, remote_root: str) -> set[str]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        print(f"[WARN] Could not read {path}: {exc}", file=sys.stderr)
        return set()

    refs: set[str] = set()
    for match in REFERENCE_RE.finditer(data):
        raw = match.group("path").decode("utf-8", errors="ignore")
        resolved = resolve_reference(current_asset, raw, remote_root)
        if resolved and resolved != current_asset:
            refs.add(resolved)
    return refs


def curl_download(url: str, destination: Path, proxy: str | None, force: bool) -> bool:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        print(f"[SKIP] {destination}")
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    cmd = [
        "curl",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--retry",
        "3",
        "--connect-timeout",
        "20",
        "--max-time",
        "600",
        "--output",
        str(partial),
        url,
    ]
    if proxy:
        cmd[1:1] = ["--proxy", proxy]

    print(f"[GET ] {url}", flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        if partial.exists():
            partial.unlink()
        print(f"[FAIL] curl exited with {result.returncode}: {url}", file=sys.stderr)
        return False

    partial.replace(destination)
    return True


def urllib_download(url: str, destination: Path, proxy: str | None, force: bool) -> bool:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        print(f"[SKIP] {destination}")
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    opener = urllib.request.build_opener(*handlers)

    print(f"[GET ] {url}")
    try:
        with opener.open(url, timeout=60) as response, partial.open("wb") as out_file:
            shutil.copyfileobj(response, out_file)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if partial.exists():
            partial.unlink()
        print(f"[FAIL] {exc}: {url}", file=sys.stderr)
        return False

    partial.replace(destination)
    return True


def download_asset(
    asset_path: str,
    remote_root: str,
    local_root: Path,
    proxy: str | None,
    force: bool,
    use_urllib: bool,
) -> bool:
    asset_path = normalize_asset_path(asset_path)
    url = to_remote_url(remote_root, asset_path)
    destination = to_local_path(local_root, asset_path)

    if not use_urllib and shutil.which("curl"):
        return curl_download(url, destination, proxy, force)
    return urllib_download(url, destination, proxy, force)


def main() -> int:
    args = parse_args()
    remote_root = args.remote_root.rstrip("/")
    local_root = args.local_root.resolve()
    requested = [normalize_asset_path(path) for path in (*DEFAULT_ASSETS, *args.asset)]

    queue: deque[str] = deque(requested)
    seen: set[str] = set()
    failed: list[str] = []

    while queue:
        asset_path = queue.popleft()
        if asset_path in seen:
            continue
        seen.add(asset_path)

        ok = download_asset(
            asset_path=asset_path,
            remote_root=remote_root,
            local_root=local_root,
            proxy=args.proxy,
            force=args.force,
            use_urllib=args.use_urllib,
        )
        if not ok:
            failed.append(asset_path)
            continue

        if args.no_recursive:
            continue
        local_path = to_local_path(local_root, asset_path)
        for ref in sorted(extract_references(local_path, asset_path, remote_root)):
            if ref not in seen:
                queue.append(ref)

    print()
    print(f"Downloaded or verified {len(seen) - len(failed)} assets under: {local_root}")
    if args.print_kit_args:
        print()
        print(f'--kit_args "--/persistent/isaac/asset_root/cloud={local_root}"')

    if failed:
        print()
        print("Failed assets:", file=sys.stderr)
        for asset_path in failed:
            print(f"  {asset_path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
