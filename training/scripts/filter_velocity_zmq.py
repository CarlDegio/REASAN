"""Small one-way publisher for G1 Filter velocity commands."""

from __future__ import annotations

import struct
import time
from collections.abc import Sequence

import zmq


MESSAGE_PREFIX = b"filter_velocity"
PACKET_VELOCITY = 0
PACKET_STOP = 1
_PACKET = struct.Struct("<BQfff")


def encode_filter_packet(kind: int, sequence: int, velocity: Sequence[float]) -> bytes:
    if kind not in (PACKET_VELOCITY, PACKET_STOP):
        raise ValueError(f"Unsupported Filter packet kind: {kind}")
    if len(velocity) != 3:
        raise ValueError("Filter velocity must contain vx, vy, and wz")
    return MESSAGE_PREFIX + _PACKET.pack(kind, sequence, *map(float, velocity))


def decode_filter_packet(packet: bytes) -> tuple[int, int, tuple[float, float, float]]:
    if not packet.startswith(MESSAGE_PREFIX) or len(packet) != len(MESSAGE_PREFIX) + _PACKET.size:
        raise ValueError("Malformed Filter velocity packet")
    kind, sequence, vx, vy, wz = _PACKET.unpack_from(packet, len(MESSAGE_PREFIX))
    if kind not in (PACKET_VELOCITY, PACKET_STOP):
        raise ValueError(f"Unsupported Filter packet kind: {kind}")
    return kind, sequence, (vx, vy, wz)


class FilterVelocityPublisher:
    def __init__(self, endpoint: str, context: zmq.Context | None = None):
        self._context = context or zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(endpoint)
        self._sequence = 0
        self._closed = False

    def publish(self, velocity: Sequence[float]) -> None:
        if self._closed:
            return
        self._socket.send(encode_filter_packet(PACKET_VELOCITY, self._sequence, velocity))
        self._sequence += 1

    def close(self) -> None:
        if self._closed:
            return
        stop = encode_filter_packet(PACKET_STOP, self._sequence, (0.0, 0.0, 0.0))
        for _ in range(3):
            self._socket.send(stop)
            time.sleep(0.02)
        self._socket.close()
        self._closed = True
