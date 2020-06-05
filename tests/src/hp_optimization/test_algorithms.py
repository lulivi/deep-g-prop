import unittest

from unittest import TestCase, mock

from src.hp_optimization import ga, grid, rand


class TestOptimizers(TestCase):
    """Test all the optimizers module."""

    @mock.patch("src.hp_optimization.rand.common")
    def test_rand(self, mock_common):
        """Test the cross run_search."""
        mock_output_cv = mock.Mock()
        mock_common.cross_validate.return_value = mock_output_cv
        rand.run_search()

        mock_common.RandomizedSearchCV.assert_called_once()
        mock_common.cross_validate.assert_called_once()
        mock_common.save_result.assert_called_with(mock_output_cv)

    @mock.patch("src.hp_optimization.grid.common")
    def test_grid(self, mock_common):
        """Test the cross run_search."""
        mock_output_cv = mock.Mock()
        mock_common.cross_validate.return_value = mock_output_cv
        grid.run_search()

        mock_common.GridSearchCV.assert_called_once()
        mock_common.cross_validate.assert_called_once()
        mock_common.save_result.assert_called_with(mock_output_cv)

    @mock.patch("src.hp_optimization.ga.common")
    def test_ga(self, mock_common):
        """Test the cross run_search."""
        mock_output_cv = mock.Mock()
        mock_common.cross_validate.return_value = mock_output_cv
        ga.run_search()

        mock_common.EvolutionaryAlgorithmSearchCV.assert_called_once()
        mock_common.cross_validate.assert_called_once()
        mock_common.save_result.assert_called_with(mock_output_cv)

if __name__ == "__main__":
    unittest.main()
