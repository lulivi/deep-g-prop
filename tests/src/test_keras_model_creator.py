"""Test the :mod:`src.keras_model_creator` module."""
import unittest

from unittest import mock

from click.testing import CliRunner

from src.keras_model_creator import cli


class TestKerasModelCreatorCli(unittest.TestCase):
    """Tests for the MLPIndividual class."""

    def test_cli_missing_required(self):
        """Required arguments are not provided."""
        runner = CliRunner()
        with mock.patch("src.deep_g_prop.keras"):
            result = runner.invoke(cli, [])

        self.assertEqual(result.exit_code, 2, result.stdout)

    def test_cli_dataset_not_found(self):
        """Dataset could not be found."""
        model_name = "example1"
        dataset_name = "nonexistent"
        hidden_layers = "5 5"
        runner = CliRunner()
        with mock.patch("src.deep_g_prop.keras"):
            result = runner.invoke(
                cli, [model_name, dataset_name, hidden_layers]
            )

        self.assertEqual(result.exit_code, 2, result.stdout)

    def test_cli_hiden_layers_format_error(self):
        """Model path could not be found."""
        model_name = "example1"
        dataset_name = "cancer1"
        hidden_layers = "incorrect sequence"
        runner = CliRunner()
        with mock.patch("src.deep_g_prop.keras"):
            result = runner.invoke(
                cli, [model_name, dataset_name, hidden_layers]
            )

        self.assertEqual(result.exit_code, 2, result.stdout)

    def test_cli_ok(self):
        """Model path could not be found."""
        model_name = "example1"
        dataset_name = "cancer1"
        hidden_layers = "5 6 7"
        runner = CliRunner()
        with mock.patch("src.deep_g_prop.keras"):
            result = runner.invoke(
                cli, [model_name, dataset_name, hidden_layers]
            )

        self.assertEqual(result.exit_code, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
