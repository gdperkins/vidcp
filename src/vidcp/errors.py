"""User-facing error type.

``VidcpError`` is a :class:`click.ClickException` subclass so that Click's
standalone handling shows it nicely (message in red, hint in dim) and exits
with code 1, without a traceback — no extra wiring needed at the entrypoint.
"""

from __future__ import annotations

from typing import IO

import click


class VidcpError(click.ClickException):
    """A recoverable, user-facing error.

    Parameters
    ----------
    message:
        The primary error message, shown in red.
    hint:
        Optional remediation hint, shown in dim below the message.
    """

    exit_code = 1

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint

    def show(self, file: IO[str] | None = None) -> None:
        click.echo(f"Error: {self.message}", err=True, file=file)
        if self.hint:
            click.echo(f"{self.hint}", err=True, file=file)
