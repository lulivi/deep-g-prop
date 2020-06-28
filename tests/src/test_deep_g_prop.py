"""Test the :mod:`src.deep_g_prop` module."""
import unittest

from unittest import mock

from click.testing import CliRunner

from src.deep_g_prop import cli


class TestDeepGPropCli(unittest.TestCase):
    """Tests for the MLPIndividual class."""

    @mock.patch("src.deep_g_prop.DGPLOGGER")
    def test_cli_dataset_not_found(self, mock_dgplogger):
        """Model path could not be found."""
        dataset_name = "nonexistent"
        runner = CliRunner()
        with mock.patch("src.deep_g_prop.genetic_algorithm"):
            result = runner.invoke(cli, ["--dataset-name", dataset_name])

        self.assertEqual(result.exit_code, 2, result.stdout)
        mock_dgplogger.configure_dgp_logger.assert_not_called()

    @mock.patch("src.deep_g_prop.DGPLOGGER")
    def test_cli_wrong_hidden_sequence(self, mock_dgplogger):
        """Model path could not be found."""
        test_hidden_sequence = "wrong This"
        runner = CliRunner()
        with mock.patch("src.deep_g_prop.genetic_algorithm"):
            result = runner.invoke(
                cli, ["--hidden-sequence", test_hidden_sequence]
            )

        self.assertEqual(result.exit_code, 2, result.stdout)
        mock_dgplogger.configure_dgp_logger.assert_not_called()

    @mock.patch("src.deep_g_prop.DGPLOGGER")
    def test_cli_ok(self, mock_dgplogger):
        """Model is found."""
        argv = [
            "--init-population-size",
            "1",
            "--max-generations",
            "1",
            "--cx-prob",
            "1.1",
            "--mut-bias-prob",
            "1.1",
            "--mut-weights-prob",
            "1.1",
            "--mut-neuron-prob",
            "1.1",
            "--mut-layer-prob",
            "1.1",
            "--fit-train-prob",
            "0.5",
            "--verbosity",
            "DEBUG",
        ]
        runner = CliRunner()
        with mock.patch("src.deep_g_prop.genetic_algorithm"):
            result = runner.invoke(cli, argv)

        mock_dgplogger.configure_dgp_logger.assert_called_once()
        self.assertEqual(result.exit_code, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
