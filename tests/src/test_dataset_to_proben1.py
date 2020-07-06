"""Test the :mod:`src.dataset_to_proben1` module."""
import unittest

from pathlib import Path
from unittest import mock

from click.testing import CliRunner

from settings import PROBEN1_DIR_PATH
from src.dataset_to_proben1 import cli


class TestDatasetToProben1(unittest.TestCase):
    """Tests for the dataset to proben1 converter CLI."""

    def test_cli_dataset_not_found(self):
        """Dataset path could not be found."""
        file_path = Path("non/existent/path")
        runner = CliRunner()
        result = runner.invoke(cli, [str(file_path)])

        self.assertEqual(result.exit_code, 2, result.stdout)

    @mock.patch("src.dataset_to_proben1.np")
    def test_cli_ok(self, mock_np):
        """Dataset is found."""
        mock_np.split.return_value = ("train", "validation", "test")
        runner = CliRunner()
        with mock.patch("src.dataset_to_proben1.pd"):
            with mock.patch("src.dataset_to_proben1.shuffle"):
                result = runner.invoke(
                    cli, [str(PROBEN1_DIR_PATH / "spambase.csv")]
                )

        self.assertEqual(result.exit_code, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
