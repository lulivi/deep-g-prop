import unittest

from unittest import TestCase, mock

from src.hp_optimization.common import cross_validate, save_result


class TestCommon(TestCase):
    """Test the common module."""

    @mock.patch("src.hp_optimization.common.time.perf_counter")
    def test_cross_validate(self, mock_perf_counter):
        """Test the cross validate method."""
        sample_name = "estimator"
        mock_cv = mock.Mock()
        mock_cv.best_score_ = 0.99999
        mock_perf_counter.side_effect = [0, 5]
        output_result = cross_validate(sample_name, mock_cv)

        self.assertEqual(output_result[0], sample_name)
        self.assertEqual(output_result[1], str(mock_cv.best_score_))
        self.assertEqual(output_result[2], "5.00000")

    @mock.patch("src.hp_optimization.common.open")
    @mock.patch("src.hp_optimization.common.csv")
    def test_save_result(self, mock_csv, mock_open):
        """Test the result saving."""
        mock_file_descriptor = mock.Mock()
        mock_open.__enter__ = mock.Mock(return_value=mock_file_descriptor)
        mock_result = mock.Mock()

        save_result(mock_result)


if __name__ == "__main__":
    unittest.main()
