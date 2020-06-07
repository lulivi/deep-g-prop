"""Test :mod:`src.mlp_frameworks` module common functions and variables."""
import unittest

from unittest import TestCase, mock

from src.mlp_frameworks import common


class TestCommon(TestCase):
    """Test the common module."""

    @mock.patch("src.mlp_frameworks.common.time.perf_counter")
    def test_cross_validation(self, mock_perf_counter):
        """Test the cross validation method."""
        common.N_SPLITS = 2
        mock_perf_counter.side_effect = (1.0, 2.0, 1.0, 2.0)
        mock_estimator = mock.Mock()
        mock_estimator_name = "estimator"
        mock_x = mock.MagicMock()
        mock_y = mock.MagicMock()
        mock_metric_1 = mock.Mock(side_effect=(0.1, 0.2, 0.1, 0.2))
        mock_metric_2 = mock.Mock(side_effect=(0.3, 0.4, 0.3, 0.4))
        mock_metrics = {
            "metric_1": (mock_metric_1, {}),
            "metric_2": (mock_metric_2, {"attribute": "attribute_value"}),
        }
        with mock.patch("src.mlp_frameworks.common.K_FOLD") as mock_k_fold:
            mock_k_fold.split.return_value = [
                ("train_idx", "test_idx"),
                ("train_idx", "test_idx"),
            ]
            obtained_result = common.cross_validation(
                mock_estimator,
                mock_estimator_name,
                mock_x,
                mock_y,
                mock_metrics,
            )

        expected_output = [
            "estimator",
            "1.00",
            "0.100000",
            "0.000000",
            "0.200000",
            "0.000000",
            "0.300000",
            "0.000000",
            "0.400000",
            "0.000000",
        ]
        self.assertListEqual(obtained_result, expected_output)
        self.assertEqual(mock_estimator.fit.call_count, 2)

    @staticmethod
    @mock.patch("src.mlp_frameworks.common.tabulate")
    def test_show_and_save_result(mock_tabulate):
        """Test the result showing and saving."""
        mock_result = mock.Mock()
        common.save_result = mock.Mock()
        common.show_and_save_result(mock_result)

        mock_tabulate.assert_called_with(
            tabular_data=[mock_result],
            headers=common.result_header,
            tablefmt="pretty",
        )
        common.save_result.assert_called_once()

    @staticmethod
    @mock.patch("src.mlp_frameworks.common.csv")
    @mock.patch("src.mlp_frameworks.common.open")
    def test_save_result_ok(mock_open, mock_csv):
        """Test the result save."""
        common.FRAMEWORK_RESULTS_CSV = mock.Mock()
        mock_file_descriptor = mock.Mock()
        mock_open.__enter__ = mock.Mock(return_value=mock_file_descriptor)
        mock_row = mock.Mock()
        mock_csv.reader.return_value = [common.result_header, mock_row]
        mock_result = mock.Mock()

        common.save_result(mock_result)

    @mock.patch("src.mlp_frameworks.common.csv")
    @mock.patch("src.mlp_frameworks.common.open")
    def test_save_result_different_headers(self, mock_open, mock_csv):
        """Test the result save."""
        common.FRAMEWORK_RESULTS_CSV = mock.Mock()
        mock_file_descriptor = mock.Mock()
        mock_open.__enter__ = mock.Mock(return_value=mock_file_descriptor)
        mock_row = mock.Mock()
        mock_csv.reader.return_value = [
            ["not", "the", "same", "header"],
            mock_row,
        ]
        mock_result = mock.Mock()

        with self.assertRaises(TypeError):
            common.save_result(mock_result)


if __name__ == "__main__":
    unittest.main()
