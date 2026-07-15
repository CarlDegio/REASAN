#!/usr/bin/env python3
"""Publish Navila-format velocity commands from a raw keyboard terminal."""

from __future__ import annotations

import argparse
import sys
import termios
import time
import tty

import zmq

from navila_velocity_zmq import build_navila_message


KEY_COMMANDS = {
    "w": ("forward", (0.5, 0.0, 0.0)),
    "s": ("backward", (-0.3, 0.0, 0.0)),
    "a": ("move_left", (0.0, 0.15, 0.0)),
    "d": ("move_right", (0.0, -0.15, 0.0)),
    "q": ("turn_left", (0.0, 0.0, 0.5)),
    "e": ("turn_right", (0.0, 0.0, -0.5)),
    " ": ("stop", (0.0, 0.0, 0.0)),
}
QUIT_KEY = "x"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://*:5560")
    parser.add_argument("--duration", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration <= 0.0:
        raise ValueError("duration must be positive")
    if not sys.stdin.isatty():
        raise RuntimeError("The keyboard mock requires an interactive terminal")

    context = zmq.Context.instance()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(args.endpoint)
    original_terminal = termios.tcgetattr(sys.stdin.fileno())
    stop_message = build_navila_message("stop", (0.0, 0.0, 0.0), args.duration)
    print(f"Navila keyboard mock publishing on {args.endpoint}, duration={args.duration:g}s")
    print("W/S forward/back | A/D lateral | Q/E yaw | Space stop | X exit")
    time.sleep(0.3)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            key = sys.stdin.read(1).lower()
            if key == QUIT_KEY:
                break
            command = KEY_COMMANDS.get(key)
            if command is None:
                continue
            action, velocity = command
            message = build_navila_message(action, velocity, args.duration)
            socket.send_string(message)
            print(
                f"\r{action:>10}: vx={velocity[0]:+.2f} vy={velocity[1]:+.2f} "
                f"wz={velocity[2]:+.2f} duration={args.duration:g}s   ",
                end="",
                flush=True,
            )
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, original_terminal)
        for _ in range(3):
            socket.send_string(stop_message)
            time.sleep(0.02)
        socket.close()
        print("\nStopped")


if __name__ == "__main__":
    main()
