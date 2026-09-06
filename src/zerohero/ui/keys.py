"""Cross-platform non-blocking single-key reader for the terminal UI.

POSIX: cbreak mode (termios/tty) + select() with zero timeout so poll() never
blocks. Windows: msvcrt.kbhit()/getwch(), which is already non-blocking by
design. Both sides normalise to the same vocabulary: single printable chars,
"esc", "up"/"down"/"left"/"right", " ".
"""

from __future__ import annotations

import sys

try:
    import termios
    import tty

    _POSIX = True
except ImportError:  # Windows
    _POSIX = False

try:
    import msvcrt
except ImportError:
    msvcrt = None  # type: ignore[assignment]


class KeyReader:
    """Context manager. Construction/exit never raises, even off a real tty."""

    def __init__(self) -> None:
        self._fd: int | None = None
        self._old_settings: list | None = None
        self._is_tty = False

    def __enter__(self) -> KeyReader:
        if _POSIX:
            try:
                self._fd = sys.stdin.fileno()
                self._is_tty = sys.stdin.isatty()
                if self._is_tty:
                    self._old_settings = termios.tcgetattr(self._fd)
                    tty.setcbreak(self._fd)
            except (OSError, ValueError, termios.error):
                self._is_tty = False
        else:
            try:
                self._is_tty = sys.stdin.isatty()
            except (OSError, ValueError):
                self._is_tty = False
        return self

    def __exit__(self, *exc: object) -> None:
        if _POSIX and self._fd is not None and self._old_settings is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            except (OSError, ValueError, termios.error):
                pass

    def poll(self) -> str | None:
        """Return one normalised keypress, or None if nothing is waiting."""
        if not self._is_tty:
            return None
        if _POSIX:
            return self._poll_posix()
        return self._poll_windows()

    def _poll_posix(self) -> str | None:
        import select

        assert self._fd is not None
        ready, _, _ = select.select([self._fd], [], [], 0)
        if not ready:
            return None
        try:
            ch = sys.stdin.read(1)
        except (OSError, ValueError):
            return None
        if not ch:
            return None
        if ch == "\x1b":
            # Arrow keys arrive as ESC [ A/B/C/D. If nothing follows within
            # this instant, it's a bare ESC -- peek with another zero-timeout
            # select rather than blocking for the rest of the sequence.
            ready, _, _ = select.select([self._fd], [], [], 0)
            if not ready:
                return "esc"
            ch2 = sys.stdin.read(1)
            if ch2 != "[":
                return "esc"
            ready, _, _ = select.select([self._fd], [], [], 0)
            if not ready:
                return "esc"
            ch3 = sys.stdin.read(1)
            return {"A": "up", "B": "down", "C": "right", "D": "left"}.get(ch3)
        if ch == " ":
            return " "
        return ch

    def _poll_windows(self) -> str | None:
        # typeshed's msvcrt stub is win32-only, so these are unresolved on
        # every other platform's mypy run -- real and harmless.
        if msvcrt is None or not msvcrt.kbhit():  # type: ignore[attr-defined]
            return None
        ch = msvcrt.getwch()  # type: ignore[attr-defined]
        if ch in ("\x00", "\xe0"):  # arrow/function key prefix
            if not msvcrt.kbhit():  # type: ignore[attr-defined]
                return None
            ch2 = msvcrt.getwch()  # type: ignore[attr-defined]
            return {"H": "up", "P": "down", "M": "right", "K": "left"}.get(ch2)
        if ch == "\x1b":
            return "esc"
        if ch == " ":
            return " "
        return ch


def enable_vt() -> None:
    """Turn on ANSI escape processing for the Windows console. No-op elsewhere."""
    if _POSIX:
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        STD_OUTPUT_HANDLE = -11
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return
        kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
    except Exception:
        pass
