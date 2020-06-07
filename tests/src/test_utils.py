"""Test the :mod:`src.utils` module."""
import logging
import sys
import tempfile
import unittest

from pathlib import Path

import pandas as pd  # type: ignore

from src import utils

ROOT = Path(__file__).parents[2].resolve()
try:
    SOURCE_DIR = ROOT.joinpath("src").resolve(strict=True)
    DATASET_DIR = SOURCE_DIR.joinpath("datasets").resolve(strict=True)
except FileNotFoundError as error:
    sys.exit(f"{error.strerror}: {error.filename}")


class Proben1Tests(unittest.TestCase):
    """Test proben1 functions."""

    def test_read_proben1_partition(self):
        """Test read partition function."""
        example_partition_name = "cancer1"
        partition_dict = utils.read_proben1_partition(
            DATASET_DIR, example_partition_name
        )

        self.assertEqual(partition_dict["name"], example_partition_name)
        self.assertIsInstance(partition_dict["trn"]["X"], pd.DataFrame)
        self.assertIsInstance(partition_dict["trn"]["y"], pd.Series)
        self.assertIsInstance(partition_dict["val"]["X"], pd.DataFrame)
        self.assertIsInstance(partition_dict["val"]["y"], pd.Series)
        self.assertIsInstance(partition_dict["tst"]["X"], pd.DataFrame)
        self.assertIsInstance(partition_dict["tst"]["y"], pd.Series)

        with self.assertRaises(utils.DatasetNotFoundError):
            utils.read_proben1_partition(DATASET_DIR, "non_existent_dataset")

    def test_read_all_proben1_partitions(self):
        """Test read all partitions function."""
        example_dataset_name = "cancer"
        partition_list = utils.read_all_proben1_partitions(
            DATASET_DIR, example_dataset_name
        )
        number_cancer_partitions = len(
            {
                part.stem
                for part in DATASET_DIR.glob(f"{example_dataset_name}*")
                if part.stem[-1].isnumeric()
            }
        )

        self.assertEqual(len(partition_list), number_cancer_partitions)
        for part in partition_list:
            self.assertIsInstance(part, dict)
            self.assertIn(example_dataset_name, part["name"])
            self.assertIsInstance(part["trn"]["X"], pd.DataFrame)
            self.assertIsInstance(part["trn"]["y"], pd.Series)
            self.assertIsInstance(part["val"]["X"], pd.DataFrame)
            self.assertIsInstance(part["val"]["y"], pd.Series)
            self.assertIsInstance(part["tst"]["X"], pd.DataFrame)
            self.assertIsInstance(part["tst"]["y"], pd.Series)

        with self.assertRaises(utils.PartitionsNotFoundError):
            utils.read_all_proben1_partitions(
                DATASET_DIR, "non_existent_partition_name"
            )

    def test_validate_level(self):
        """Test the validate level utility function."""
        obtained_level = utils.validate_level("DEBUG")
        self.assertEqual(obtained_level, logging.DEBUG)
        obtained_level = utils.validate_level("NOLEVEL")
        self.assertEqual(obtained_level, logging.INFO)

    def test_configure_logger(self):
        """Test the logger configuration."""
        logger = utils.configure_logger(
            "test_logger", Path(tempfile.gettempdir()), "DEBUG"
        )
        expected_handlers_types = [logging.StreamHandler, logging.FileHandler]

        self.assertListEqual(
            expected_handlers_types, list(map(type, logger.handlers)),
        )


if __name__ == "__main__":
    unittest.main()
