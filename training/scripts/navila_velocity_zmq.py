"""Receive Navila JSON commands and expose the active 3-D velocity segment."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any, Sequence

import zmq


MESSAGE_TYPE = "navila_reasan_velocity_command"
ZERO_VELOCITY = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class VelocitySegment:
    duration_s: float
    velocity: tuple[float, float, float]


def _decode_json(message: bytes | str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(message, dict):
        return message
    try:
        decoded = json.loads(message)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as error:
        raise ValueError(f"Invalid Navila JSON: {error}") from error
    if not isinstance(decoded, dict):
        raise ValueError("Navila message must be a JSON object")
    return decoded


def parse_navila_command(message: bytes | str | dict[str, Any]) -> tuple[VelocitySegment, ...]:
    command = _decode_json(message)
    if command.get("type") != MESSAGE_TYPE:
        raise ValueError(f"Unexpected Navila message type: {command.get('type')!r}")
    if command.get("status") != "ok":
        raise ValueError(f"Navila command status is not ok: {command.get('status')!r}")
    if command.get("version") != 1:
        raise ValueError(f"Unsupported Navila command version: {command.get('version')!r}")

    raw_segments = command.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        velocity = command.get("velocity")
        raw_segments = [dict(velocity, duration_s=command.get("duration_s"))] if isinstance(velocity, dict) else []
    if not raw_segments:
        raise ValueError("Navila command contains no velocity segments")

    segments = []
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, dict):
            raise ValueError(f"Navila segment {index} must be an object")
        try:
            duration = float(raw["duration_s"])
            velocity = (float(raw["vx"]), float(raw["vy"]), float(raw["wz"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid Navila segment {index}: {error}") from error
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError(f"Navila segment {index} duration must be finite and positive")
        if not all(math.isfinite(value) for value in velocity):
            raise ValueError(f"Navila segment {index} velocity must be finite")
        segments.append(VelocitySegment(duration, velocity))
    return tuple(segments)


class NavilaCommandTimeline:
    def __init__(self):
        self._segments: tuple[VelocitySegment, ...] = ()
        self._started_at = 0.0

    def replace(self, segments: Sequence[VelocitySegment], now: float) -> None:
        self._segments = tuple(segments)
        self._started_at = now

    def velocity(self, now: float) -> tuple[float, float, float]:
        elapsed = max(0.0, now - self._started_at)
        for segment in self._segments:
            if elapsed < segment.duration_s - 1.0e-9:
                return segment.velocity
            elapsed -= segment.duration_s
        return ZERO_VELOCITY


def build_navila_message(
    action: str, velocity: Sequence[float], duration_s: float, raw_text: str = "keyboard mock"
) -> str:
    vx, vy, wz = map(float, velocity)
    payload = {
        "action": action,
        "angle_rad": abs(wz) * duration_s,
        "distance_m": math.hypot(vx, vy) * duration_s,
        "duration_s": float(duration_s),
        "raw_text": raw_text,
        "segments": [{"duration_s": float(duration_s), "vx": vx, "vy": vy, "wz": wz}],
        "source": "keyboard_mock",
        "status": "ok",
        "type": MESSAGE_TYPE,
        "velocity": {"vx": vx, "vy": vy, "wz": wz},
        "version": 1,
    }
    return json.dumps(payload, separators=(",", ":"))


class NavilaVelocitySubscriber:
    def __init__(self, endpoint: str, context: zmq.Context | None = None):
        self._context = context or zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(endpoint)
        self._timeline = NavilaCommandTimeline()

    def poll(self, now: float | None = None) -> tuple[float, float, float]:
        now = time.monotonic() if now is None else now
        try:
            message = self._socket.recv(zmq.NOBLOCK)
        except zmq.Again:
            return self._timeline.velocity(now)
        try:
            segments = parse_navila_command(message)
        except ValueError as error:
            print(f"[NAVILA] Ignored command: {error}")
        else:
            self._timeline.replace(segments, now)
            print(
                f"[NAVILA] Command: {len(segments)} segment(s), "
                f"duration_s={segments[0].duration_s:g}, first={segments[0].velocity}"
            )
        return self._timeline.velocity(now)

    def close(self) -> None:
        self._socket.close()
