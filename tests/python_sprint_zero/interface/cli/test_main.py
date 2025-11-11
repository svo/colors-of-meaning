from assertpy import assert_that
from click.testing import CliRunner

from python_sprint_zero.interface.cli.main import run


def test_should_print_coconuts():
    runner = CliRunner()
    result = runner.invoke(run)
    assert_that(result.output).is_equal_to("coconuts\n")


def test_should_print_custom_message():
    runner = CliRunner()
    result = runner.invoke(run, ["--message", "custom"])
    assert_that(result.output).is_equal_to("custom\n")
