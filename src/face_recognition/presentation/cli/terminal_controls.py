"""Nonblocking capture controls for a Windows console or Linux terminal."""

import os
import select
import sys
from typing import Any


class TerminalControls:
    def __init__(self) -> None:
        self._fd: int | None = None
        self._saved: Any = None

    def __enter__(self) -> "TerminalControls":
        if not sys.stdin.isatty():
            raise RuntimeError("Dataset capture requires an interactive terminal for SPACE/q controls")
        if os.name != "nt":
            import termios
            import tty

            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            try:
                tty.setcbreak(self._fd)
            except BaseException:
                termios.tcsetattr(self._fd, termios.TCSANOW, self._saved)
                raise
        return self

    def poll(self) -> str | None:
        if os.name == "nt":
            import msvcrt
            if not msvcrt.kbhit():
                return None
            key = msvcrt.getwch()
            if key in ("\x00", "\xe0"):
                if msvcrt.kbhit():
                    msvcrt.getwch()
                return None
            if key == "\x03":
                raise KeyboardInterrupt
            return key.lower()
        if self._fd is None or not select.select([self._fd], [], [], 0)[0]:
            return None
        key = os.read(self._fd, 1).decode("ascii", errors="ignore").lower()
        if key == "\x03":
            raise KeyboardInterrupt
        return key

    def __exit__(self, *args: object) -> None:
        if self._saved is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSANOW, self._saved)
            self._saved = None
