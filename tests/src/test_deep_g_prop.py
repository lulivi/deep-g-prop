"""Test the :mod:`src.deep_g_prop` module."""
import unittest

from unittest import mock

from click.testing import CliRunner

from settings import TESTS_DIR_PATH
from src.deep_g_prop import cli


class TestDeepGPropCli(unittest.TestCase):
    """Tests for the MLPIndividual class."""

    def test_cli_missing_required(self):
        """Required argument is not provided."""
        runner = CliRunner()
        with mock.patch.multiple(
            "src.deep_g_prop",
            keras=mock.DEFAULT,
            genetic_algorithm=mock.DEFAULT,
        ):
            result = runner.invoke(cli, [])

        self.assertEqual(result.exit_code, 2, result.stdout)

    def test_cli_dataset_not_found(self):
        """Model path could not be found."""
        model_path = "path/to/model.h5"
        dataset_name = "nonexistent"
        runner = CliRunner()
        with mock.patch.multiple(
            "src.deep_g_prop",
            keras=mock.DEFAULT,
            genetic_algorithm=mock.DEFAULT,
        ):
            result = runner.invoke(
                cli, [model_path, "--dataset-name", dataset_name]
            )

        self.assertEqual(result.exit_code, 2, result.stdout)

    def test_cli_model_not_found(self):
        """Model path could not be found."""
        model_path = "path/to/model.h5"
        runner = CliRunner()
        with mock.patch.multiple(
            "src.deep_g_prop",
            keras=mock.DEFAULT,
            genetic_algorithm=mock.DEFAULT,
        ):
            result = runner.invoke(cli, [model_path])

        self.assertEqual(result.exit_code, 2, result.stdout)

    def test_cli_model_found(self):
        """Model is found."""
        model_path = str(
            TESTS_DIR_PATH / "test_files" / "cancer_test_model.h5"
        )
        runner = CliRunner()
        with mock.patch.multiple(
            "src.deep_g_prop",
            keras=mock.DEFAULT,
            genetic_algorithm=mock.DEFAULT,
        ):
            result = runner.invoke(cli, [model_path])

        self.assertEqual(result.exit_code, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
