"""Test the :mod:`src.utils` module."""
import unittest

from unittest import mock

import numpy as np

from settings import PROBEN1_DIR_PATH
from src import utils


class Proben1Tests(unittest.TestCase):
    """Test proben1 functions."""

    def test_read_proben1_partition(self):
        """Test read partition function."""
        example_partition_name = "cancer1"
        part = utils.read_proben1_partition(
            example_partition_name, PROBEN1_DIR_PATH
        )

        self.assertEqual(part.name, example_partition_name)
        self.assertIsInstance(part.nin, int)
        self.assertIsInstance(part.nout, int)
        self.assertIsInstance(part.trn.X, np.ndarray)
        self.assertIsInstance(part.trn.y, np.ndarray)
        self.assertIsInstance(part.val.X, np.ndarray)
        self.assertIsInstance(part.val.y, np.ndarray)
        self.assertIsInstance(part.tst.X, np.ndarray)
        self.assertIsInstance(part.tst.y, np.ndarray)

        with self.assertRaises(utils.DatasetNotFoundError):
            utils.read_proben1_partition(
                "non_existent_dataset", PROBEN1_DIR_PATH
            )

    def test_read_all_proben1_partitions(self):
        """Test read all partitions function."""
        example_dataset_name = "cancer"
        partition_list = utils.read_all_proben1_partitions(
            example_dataset_name, PROBEN1_DIR_PATH
        )
        number_cancer_partitions = len(
            {
                part.stem
                for part in PROBEN1_DIR_PATH.glob(f"{example_dataset_name}*")
                if part.stem[-1].isnumeric()
            }
        )

        self.assertEqual(len(partition_list), number_cancer_partitions)
        for part in partition_list:
            self.assertIsInstance(part, utils.Proben1Partition)
            self.assertIn(example_dataset_name, part.name)
            self.assertIsInstance(part.nin, int)
            self.assertIsInstance(part.nout, int)
            self.assertIsInstance(part.trn.X, np.ndarray)
            self.assertIsInstance(part.trn.y, np.ndarray)
            self.assertIsInstance(part.val.X, np.ndarray)
            self.assertIsInstance(part.val.y, np.ndarray)
            self.assertIsInstance(part.tst.X, np.ndarray)
            self.assertIsInstance(part.tst.y, np.ndarray)

        with self.assertRaises(utils.PartitionsNotFoundError):
            utils.read_all_proben1_partitions(
                "non_existent_partition_name", PROBEN1_DIR_PATH
            )

    @staticmethod
    @mock.patch("src.utils.tabulate")
    def test_print_table(mock_tabulate):
        """Test the table printer."""
        table_data = [
            ["data1", "data2"],
            ["data3", "data4"],
        ]
        table_attributes = {
            "headers": ["header1", "header2"],
            "colalign": ("left", "right"),
        }
        utils.print_table(table_data, mock.Mock(), **table_attributes)

        mock_tabulate.assert_called_with(
            tabular_data=table_data,
            headers=table_attributes["headers"],
            tablefmt="simple",
            colalign=table_attributes["colalign"],
        )

    def test_print_data_summary(self):
        """Print the chosen data summary."""
        data = np.array(
            [
                [0.1, 0.1, 0.1],
                [0.2, 0.2, 0.2],
                [0.3, 0.1, 0.2],
                [0.4, 0.4, 0.4],
            ]
        )
        labels = np.array([0, 0, 1, 0])
        mock_print = mock.Mock()

        utils.print_data_summary(data, labels, "test", mock_print)

        self.assertEqual(mock_print.call_count, 6)


if __name__ == "__main__":
    unittest.main()
