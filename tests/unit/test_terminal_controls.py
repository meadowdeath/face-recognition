from unittest.mock import Mock, patch
import unittest

from face_recognition.presentation.cli import terminal_controls as controls
from face_recognition.presentation.cli import validate_normalization


class TerminalControlsTests(unittest.TestCase):
    def test_redirected_input_fails_clearly(self) -> None:
        with patch.object(controls.sys.stdin, "isatty", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "interactive terminal"):
                with controls.TerminalControls():
                    pass

    def test_windows_reuses_existing_nonblocking_poll(self) -> None:
        with patch.object(controls.sys.stdin, "isatty", return_value=True), \
             patch.object(controls.os, "name", "nt"), \
             patch.object(validate_normalization, "terminal_key", return_value=" ") as poll:
            with controls.TerminalControls() as terminal:
                self.assertEqual(terminal.poll(), " ")
        poll.assert_called_once()

    def test_linux_reads_without_enter_and_restores_terminal_on_failure(self) -> None:
        termios = Mock(TCSANOW=0)
        termios.tcgetattr.return_value = ["original terminal state"]
        tty = Mock()
        stdin = Mock()
        stdin.isatty.return_value = True
        stdin.fileno.return_value = 7
        with patch.object(controls.os, "name", "posix"), patch.object(controls.sys, "stdin", stdin), \
             patch.dict("sys.modules", {"termios": termios, "tty": tty}), \
             patch.object(controls.select, "select", return_value=([7], [], [])), \
             patch.object(controls.os, "read", return_value=b" "):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                with controls.TerminalControls() as terminal:
                    self.assertEqual(terminal.poll(), " ")
                    raise RuntimeError("test failure")
        tty.setcbreak.assert_called_once_with(7)
        termios.tcsetattr.assert_called_once_with(7, 0, ["original terminal state"])

    def test_linux_poll_does_not_block_or_read_when_no_key_is_available(self) -> None:
        terminal = controls.TerminalControls()
        terminal._fd = 7
        with patch.object(controls.os, "name", "posix"), \
             patch.object(controls.select, "select", return_value=([], [], [])) as select, \
             patch.object(controls.os, "read") as read:
            self.assertIsNone(terminal.poll())
        select.assert_called_once_with([7], [], [], 0)
        read.assert_not_called()
