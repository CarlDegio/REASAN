from __future__ import annotations

import ast
from pathlib import Path

import cv2
import numpy as np


TRAINING_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = TRAINING_DIR / "tests" / "manual_g1_loco_ppo_video.py"


def test_g1_loco_ppo_video_script_has_required_cli_and_runtime_wiring():
    source = SCRIPT_PATH.read_text()
    ast.parse(source)

    assert 'parser.add_argument("--checkpoint", type=Path, default=None' in source
    assert "if args_cli.checkpoint is None:" in source
    assert 'parser.add_argument("--episodes", type=int, default=10' in source
    assert 'parser.add_argument("--second_stage", action="store_true"' in source
    assert "args_cli.headless = True" in source
    assert "args_cli.enable_cameras = True" in source
    assert 'os.environ.setdefault("WANDB_MODE", "disabled")' in source
    assert 'os.environ.pop("DISPLAY", None)' in source
    assert 'os.environ.pop("WAYLAND_DISPLAY", None)' in source
    assert source.index('os.environ.pop("DISPLAY", None)') < source.index("from isaaclab.app import AppLauncher")
    assert 'TASK_NAME = "Unitree-G1-Locomotion"' in source
    assert "num_envs=1" in source
    assert 'render_mode="rgb_array"' in source

    assert "OnPolicyRunnerLoco" in source
    assert "ppo_runner.load(checkpoint, load_optimizer=False)" in source
    assert "get_inference_policy" in source
    assert "onnxruntime" not in source


def test_g1_loco_ppo_video_script_streams_one_annotated_video():
    source = SCRIPT_PATH.read_text()

    assert "def annotate_frame" in source
    assert 'f"Episode: {episode_id}"' in source
    assert 'f"Timestep: {timestep}"' in source
    assert "cv2.VideoWriter(" in source
    assert "writer.write(" in source
    assert "writer.release()" in source
    assert "traceback.print_exc()" in source
    assert "gym_env.unwrapped.render(recompute=True)" in source
    assert "frame = gym_env.unwrapped.render()" in source
    assert "force_render_after_reset" not in source
    assert "recompute=not" not in source
    assert "gym_env.render(recompute=" not in source
    assert "completed_episodes < args_cli.episodes" in source
    assert "env.close()" in source


def test_g1_loco_ppo_video_annotation_and_streaming_writer(tmp_path):
    source = SCRIPT_PATH.read_text()
    tree = ast.parse(source)
    helper_names = {"annotate_frame", "create_video_writer"}
    helper_nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in helper_names]
    namespace = {"Path": Path, "cv2": cv2, "np": np}
    exec(compile(ast.Module(body=helper_nodes, type_ignores=[]), str(SCRIPT_PATH), "exec"), namespace)

    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    annotated = namespace["annotate_frame"](frame, episode_id=3, timestep=42)

    assert annotated.shape == frame.shape
    assert annotated.dtype == np.uint8
    assert np.any(annotated[:100, :240] != 0)
    assert np.all(annotated[140:, 280:] == 0)

    output_path = tmp_path / "annotated.mp4"
    writer = namespace["create_video_writer"](output_path, annotated, fps=50)
    for _ in range(3):
        writer.write(annotated)
    writer.release()

    capture = cv2.VideoCapture(str(output_path))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
        ok, decoded_frame = capture.read()
        assert ok
        assert decoded_frame.shape == annotated.shape
    finally:
        capture.release()
