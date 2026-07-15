import struct
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from filter_velocity_zmq import (  # noqa: E402
    MESSAGE_PREFIX,
    PACKET_STOP,
    PACKET_VELOCITY,
    decode_filter_packet,
    encode_filter_packet,
)


def test_velocity_packet_round_trip():
    packet = encode_filter_packet(PACKET_VELOCITY, 17, (0.4, -0.2, 0.7))

    kind, sequence, velocity = decode_filter_packet(packet)

    assert packet.startswith(MESSAGE_PREFIX)
    assert kind == PACKET_VELOCITY
    assert sequence == 17
    assert velocity == pytest.approx((0.4, -0.2, 0.7))


def test_stop_packet_is_explicit_and_fixed_size():
    packet = encode_filter_packet(PACKET_STOP, 18, (0.0, 0.0, 0.0))

    kind, sequence, velocity = decode_filter_packet(packet)

    assert len(packet) == len(MESSAGE_PREFIX) + struct.calcsize("<BQfff")
    assert (kind, sequence, velocity) == (PACKET_STOP, 18, (0.0, 0.0, 0.0))


import pytest  # noqa: E402
