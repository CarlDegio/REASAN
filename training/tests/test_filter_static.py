from pathlib import Path


TRAINING_DIR = Path(__file__).resolve().parents[1]
PLAY_G1_FILTER_SCRIPT = TRAINING_DIR / "scripts" / "play_g1_filter.py"
FILTER_ENV = (
    TRAINING_DIR
    / "go2_lidar"
    / "go2_lidar"
    / "tasks"
    / "go2_filter_env.py"
)


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


def test_g1_play_can_replace_filter_input_from_navila_zmq():
    source = PLAY_G1_FILTER_SCRIPT.read_text()

    assert 'parser.add_argument("--navila_endpoint"' in source
    assert "NavilaVelocitySubscriber(args_cli.navila_endpoint)" in source
    assert "navila_subscriber.poll()" in source
    assert "normalized_action.copy_(navila_command" in source


def test_g1_play_suppresses_yaw_when_filter_input_is_zero():
    source = PLAY_G1_FILTER_SCRIPT.read_text()

    assert 'parser.add_argument("--suppress_output_on_zero_input"' in source
    assert "if args_cli.suppress_output_on_zero_input:" in source
    assert "suppress_output_for_zero_input(" in source
    assert "env.unwrapped._high_actions" in source


def test_g1_play_disables_ema_without_nonzero_input_deadband():
    source = PLAY_G1_FILTER_SCRIPT.read_text()

    assert "apply_linear_speed_deadband" not in source
    assert 'default=0.0,\n    help="EMA retention' in source


def test_play_uses_same_center_actor_rays_as_filter_training():
    source = FILTER_ENV.read_text()

    assert "if self.cfg.is_play_env:\n            actor_rays[:, :] = gt_rays[:, :]" not in source
