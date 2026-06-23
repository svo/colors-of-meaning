from unittest.mock import patch

from argon2 import PasswordHasher
from assertpy import assert_that

from colors_of_meaning.interface.cli.hash_password import main

ENTERED_PASSWORD = "correct-horse-battery"


class TestHashPasswordCli:
    @patch("colors_of_meaning.interface.cli.hash_password.getpass.getpass", return_value=ENTERED_PASSWORD)
    @patch("builtins.print")
    def test_should_print_argon2_hash_for_entered_password(self, mock_print, mock_getpass):
        main()

        assert_that(mock_print.call_args.args[0]).starts_with("$argon2")

    @patch("colors_of_meaning.interface.cli.hash_password.getpass.getpass", return_value=ENTERED_PASSWORD)
    @patch("builtins.print")
    def test_should_print_hash_that_verifies_for_entered_password(self, mock_print, mock_getpass):
        main()

        printed_hash = mock_print.call_args.args[0]
        assert PasswordHasher().verify(printed_hash, ENTERED_PASSWORD) is True
