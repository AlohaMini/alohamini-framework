"""One nonblocking terminal reader for both hotkeys and confirmed 0/1 rewards."""

import os
import sys
import time


class ConsoleInput:
    ESCAPE_TIMEOUT_S = 0.15

    def __init__(self, stream=None, *, hotkeys=("h", " ", "t")):
        self.hotkeys = hotkeys
        self.stream = sys.stdin if stream is None else stream
        self.reward_text = ""
        self._rating = False
        self._saved = None
        self._escape = ""
        self._escape_started_s = 0.0
        self._windows_extended = False
        self._after_cr = False
        self._echo = ""

    def __enter__(self):
        if os.name != "nt" and self.stream.isatty():
            import termios
            import tty

            self._saved = termios.tcgetattr(self.stream.fileno())
            tty.setcbreak(self.stream.fileno())
        return self

    def __exit__(self, *_exc):
        if self._saved is not None:
            import termios

            termios.tcsetattr(self.stream.fileno(), termios.TCSADRAIN, self._saved)

    def _keys(self, text):
        if (
            not text
            and self._escape == "\x1b"
            and time.monotonic() - self._escape_started_s >= self.ESCAPE_TIMEOUT_S
        ):
            self._escape = ""
            yield "q"
        for char in text:
            skip_lf = self._after_cr and char == "\n"
            self._after_cr = char == "\r"
            if skip_lf:
                continue
            if char in ("q", "Q", "\x03"):
                self._escape = ""
                yield char
                continue
            if char == "\x1b":
                if self._escape == "\x1b":
                    yield "q"
                self._escape = char
                self._escape_started_s = time.monotonic()
                continue
            if self._escape == "\x1b":
                self._escape = ""
                if char in ("[", "O"):
                    self._escape = "\x1b" + char
                    continue
                yield "q"
            elif self._escape:
                # CSI/SS3 parameters may span multiple nonblocking reads.
                if "@" <= char <= "~" and char != "[":
                    self._escape = ""
                    if char == "C":
                        yield "right"
                elif not " " <= char <= "~":
                    self._escape = ""
                continue
            yield char

    def feed(self, text, *, awaiting_reward=False, returning=False, confirm_reset=False):
        if awaiting_reward != self._rating:
            self.reward_text = ""
            self._rating = awaiting_reward
        events = []
        self._echo = ""
        for char in self._keys(text):
            if char == "right":
                events.append(("key", "right"))
            elif char in ("q", "Q", "\x03"):
                events.append(("key", "q"))
            elif returning and char == " ":
                events.append(("key", " "))
            elif confirm_reset and char in ("\r", "\n"):
                events.append(("key", "enter"))
            elif awaiting_reward:
                self._echo += char
                if char in ("\r", "\n"):
                    if self.reward_text in ("0", "1"):
                        events.append(("reward", int(self.reward_text)))
                    self.reward_text = ""
                elif char in ("\x7f", "\b"):
                    self.reward_text = self.reward_text[:-1]
                else:
                    self.reward_text += char
            elif char.lower() in self.hotkeys:
                events.append(("key", char.lower()))
        return events

    def poll(self, *, awaiting_reward=False, returning=False, confirm_reset=False):
        if os.name == "nt":
            import msvcrt

            text = ""
            while msvcrt.kbhit():
                char = msvcrt.getwch()
                if self._windows_extended:
                    self._windows_extended = False
                    if char == "M":
                        text += "\x1b[C"
                elif char in ("\x00", "\xe0"):
                    self._windows_extended = True
                else:
                    text += char
        else:
            import select

            text = ""
            while select.select([self.stream], [], [], 0)[0]:
                data = os.read(self.stream.fileno(), 1)
                if not data:
                    return [("key", "q")]
                text += data.decode("ascii", errors="ignore")
        events = self.feed(
            text, awaiting_reward=awaiting_reward, returning=returning, confirm_reset=confirm_reset
        )
        if self._echo:
            print(self._echo.replace("\r", "\n"), end="", flush=True)
        return events
