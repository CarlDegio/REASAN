from pathlib import Path


TRAINING_DIR = Path(__file__).resolve().parents[1]
PLAY_G1_FILTER_SCRIPT = TRAINING_DIR / "scripts" / "play_g1_filter.py"


def test_training_entrypoints_do_not_disable_cudnn():
    entrypoints = list((TRAINING_DIR / "scripts").glob("*.py"))
    entrypoints += list((TRAINING_DIR / "tests").glob("manual_*.py"))

    offenders = [
        path.relative_to(TRAINING_DIR)
        for path in entrypoints
        if "torch.backends.cudnn.enabled = False" in path.read_text()
    ]

    assert offenders == []


def test_g1_play_can_publish_safe_filter_velocity_at_configured_rate():
    source = PLAY_G1_FILTER_SCRIPT.read_text()

    assert 'parser.add_argument("--zmq_filter_endpoint"' in source
    assert "FilterVelocityPublisher(args_cli.zmq_filter_endpoint)" in source
    assert "env.unwrapped._high_actions[0]" in source
    assert "step % publish_interval_steps == 0" in source
