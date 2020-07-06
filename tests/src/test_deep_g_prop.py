"""Test the :mod:`src.deep_g_prop` module."""
import unittest

from unittest import mock

from click.testing import CliRunner

from src.deep_g_prop import cli


class TestDeepGPropCli(unittest.TestCase):
    """Tests for the DeepGProp CLI."""

    @mock.patch("src.deep_g_prop.DGPLOGGER")
    def test_cli_dataset_not_found(self, mock_dgplogger):
        """Non existen dataset."""
        dataset_name = "nonexistent"
        runner = CliRunner()
        with mock.patch("src.deep_g_prop.genetic_algorithm"):
            result = runner.invoke(cli, ["--dataset-name", dataset_name])

        self.assertEqual(result.exit_code, 2, result.stdout)
        mock_dgplogger.configure_dgp_logger.assert_not_called()

    @mock.patch("src.deep_g_prop.DGPLOGGER")
    def test_cli_wrong_neurons_range(self, mock_dgplogger):
        """Non valid neurons range."""
        test_neurons_range = -2, 7
        runner = CliRunner()
        with mock.patch("src.deep_g_prop.genetic_algorithm"):
            result = runner.invoke(
                cli, ["--neurons-range", *test_neurons_range]
            )

        self.assertEqual(result.exit_code, 2, result.stdout)
        mock_dgplogger.configure_dgp_logger.assert_not_called()

    @mock.patch("src.deep_g_prop.DGPLOGGER")
    def test_cli_wrong_layers_range(self, mock_dgplogger):
        """Non valid layers range."""
        test_layers_range = 2, 1
        runner = CliRunner()
        with mock.patch("src.deep_g_prop.genetic_algorithm"):
            result = runner.invoke(cli, ["--layers-range", *test_layers_range])

        self.assertEqual(result.exit_code, 2, result.stdout)
        mock_dgplogger.configure_dgp_logger.assert_not_called()

    @mock.patch("src.deep_g_prop.DGPLOGGER")
    def test_cli_ok(self, mock_dgplogger):
        """Everything ok."""
        argv = [
            "-d",
            "cancer1",
            "-ip",
            20,
            "-mg",
            10,
            "-nr",
            2,
            2,
            "-lr",
            1,
            1,
            "-cx",
            0.5,
            "-b",
            0.5,
            "-w",
            0.5,
            "-n",
            0.5,
            "-l",
            0.5,
            "-c",
            "-v",
            "info",
        ]
        runner = CliRunner()
        with mock.patch("src.deep_g_prop.genetic_algorithm"):
            result = runner.invoke(cli, argv)

        mock_dgplogger.configure_dgp_logger.assert_called_once()
        self.assertEqual(result.exit_code, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
