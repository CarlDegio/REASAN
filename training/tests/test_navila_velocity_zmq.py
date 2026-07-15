import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
MOCK_PUBLISHER = SCRIPTS / "mock_navila_velocity_publisher.py"
sys.path.insert(0, str(SCRIPTS))

from navila_velocity_zmq import (  # noqa: E402
    NavilaCommandTimeline,
    build_navila_message,
    parse_navila_command,
)


SAMPLE = {
    "action": "turn_right",
    "angle_rad": 0.7853981633974483,
    "distance_m": 0.0,
    "duration_s": 1.5,
    "raw_text": "The next action is turn right 45degree.",
    "segments": [{"duration_s": 1.5, "vx": 0.0, "vy": 0.0, "wz": -0.5235987755982988}],
    "source": "navila",
    "status": "ok",
    "type": "navila_reasan_velocity_command",
    "velocity": {"vx": 0.0, "vy": 0.0, "wz": -0.5235987755982988},
    "version": 1,
}


def test_parse_sample_command():
    segments = parse_navila_command(json.dumps(SAMPLE).encode())

    assert len(segments) == 1
    assert segments[0].duration_s == pytest.approx(1.5)
    assert segments[0].velocity == pytest.approx((0.0, 0.0, -0.5235987755982988))


def test_timeline_runs_segments_in_order_then_stops():
    timeline = NavilaCommandTimeline()
    command = dict(SAMPLE)
    command["segments"] = [
        {"duration_s": 0.5, "vx": 0.4, "vy": 0.0, "wz": 0.0},
        {"duration_s": 1.0, "vx": 0.0, "vy": 0.1, "wz": -0.2},
    ]

    timeline.replace(parse_navila_command(command), now=10.0)

    assert timeline.velocity(now=10.49) == pytest.approx((0.4, 0.0, 0.0))
    assert timeline.velocity(now=10.5) == pytest.approx((0.0, 0.1, -0.2))
    assert timeline.velocity(now=11.499) == pytest.approx((0.0, 0.1, -0.2))
    assert timeline.velocity(now=11.5) == (0.0, 0.0, 0.0)


def test_new_command_immediately_replaces_old_timeline():
    timeline = NavilaCommandTimeline()
    timeline.replace(parse_navila_command(SAMPLE), now=10.0)
    replacement = build_navila_message("forward", (0.6, 0.0, 0.0), duration_s=0.5)

    timeline.replace(parse_navila_command(replacement), now=10.2)

    assert timeline.velocity(now=10.2) == pytest.approx((0.6, 0.0, 0.0))
    assert timeline.velocity(now=10.7) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [("type", "wrong"), ("status", "error"), ("version", 2)],
)
def test_rejects_wrong_protocol_identity(field, value):
    command = dict(SAMPLE)
    command[field] = value

    with pytest.raises(ValueError):
        parse_navila_command(command)


def test_rejects_non_finite_velocity_and_non_positive_duration():
    command = dict(SAMPLE)
    command["segments"] = [{"duration_s": 0.0, "vx": float("nan"), "vy": 0.0, "wz": 0.0}]

    with pytest.raises(ValueError):
        parse_navila_command(command)


def test_mock_message_matches_navila_schema():
    message = json.loads(build_navila_message("turn_right", (0.0, 0.0, -0.5), 0.5))

    assert message["type"] == "navila_reasan_velocity_command"
    assert message["status"] == "ok"
    assert message["version"] == 1
    assert message["velocity"] == {"vx": 0.0, "vy": 0.0, "wz": -0.5}
    assert message["segments"] == [{"duration_s": 0.5, "vx": 0.0, "vy": 0.0, "wz": -0.5}]


def test_keyboard_mock_exposes_expected_motion_keys():
    source = MOCK_PUBLISHER.read_text()

    for key in ("w", "s", "a", "d", "q", "e", " ", "x"):
        assert f'"{key}"' in source
    assert "build_navila_message" in source
    assert 'default="tcp://*:5560"' in source
    assert "duration={args.duration" in source
    assert "duration_s={segments[0].duration_s" in (SCRIPTS / "navila_velocity_zmq.py").read_text()
