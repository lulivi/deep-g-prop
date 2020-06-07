"""Test :mod:`src.hp_optimization` module common functions and variables."""
import unittest

from unittest import TestCase, mock

from src.hp_optimization import common


class TestCommon(TestCase):
    """Test the common module."""

    @mock.patch("src.hp_optimization.common.time.perf_counter")
    def test_cross_validate(self, mock_perf_counter):
        """Test the cross validate method."""
        sample_name = "estimator"
        mock_cv = mock.Mock()
        mock_cv.best_score_ = 0.99999
        mock_perf_counter.side_effect = [0, 5]
        output_result = common.cross_validate(sample_name, mock_cv)

        self.assertEqual(output_result[0], sample_name)
        self.assertEqual(output_result[1], str(mock_cv.best_score_))
        self.assertEqual(output_result[2], "5.00000")

    @staticmethod
    @mock.patch("src.hp_optimization.common.csv")
    @mock.patch("src.hp_optimization.common.open")
    def test_save_result_ok(mock_open, mock_csv):
        """Test the result save."""
        common.HP_OPTIMIZATION_CSV = mock.Mock()
        mock_file_descriptor = mock.Mock()
        mock_open.__enter__ = mock.Mock(return_value=mock_file_descriptor)
        mock_row = mock.Mock()
        mock_csv.reader.return_value = [common.table_header, mock_row]
        mock_result = mock.Mock()

        common.save_result(mock_result)

    @mock.patch("src.hp_optimization.common.csv")
    @mock.patch("src.hp_optimization.common.open")
    def test_save_result_different_headers(self, mock_open, mock_csv):
        """Test the result save."""
        common.HP_OPTIMIZATION_CSV = mock.Mock()
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
