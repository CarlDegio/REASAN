# REASAN Assets

This directory is for local mirrors of online Isaac Sim/IsaacLab assets used by REASAN.

Download the default REASAN asset set:

```bash
cd /home/ubuntu/PythonProjects/lzh/REASAN/training
python assets/download_reasan_assets.py --proxy http://127.0.0.1:7890 --print-kit-args
```

The downloader stores files under:

```text
training/assets/omniverse/Assets/Isaac/4.5
```

The play scripts under `training/scripts/` use that directory as Isaac's asset root by default.

```bash
python scripts/play_loco.py --load_run loco_1
```

They also redirect Kit's texture cache to:

```text
training/assets/cache/texturecache
```

If needed, these defaults can still be overridden with explicit `--kit_args`.

```bash
python scripts/play_loco.py --load_run loco_1 --kit_args "--/persistent/isaac/asset_root/cloud=/path/to/other/assets"
```

Additional assets can be added without editing the script:

```bash
python assets/download_reasan_assets.py \
  --asset Isaac/Robots/ANYbotics/anymal_instanceable.usd \
  --proxy http://127.0.0.1:7890
```
