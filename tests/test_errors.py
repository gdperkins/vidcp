import click

from vidcp.errors import VidcpError


def test_vidcp_error_is_click_exception():
    assert isinstance(VidcpError("boom"), click.ClickException)


def test_vidcp_error_carries_message_hint_and_exit_code():
    err = VidcpError("boom", hint="do X")
    assert err.message == "boom"
    assert err.hint == "do X"
    assert err.exit_code == 1


def test_vidcp_error_without_hint():
    err = VidcpError("boom")
    assert err.hint is None


def test_vidcp_error_show_outputs_message_and_hint(capsys):
    VidcpError("boom", hint="do X").show()
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "boom" in combined
    assert "do X" in combined
