"""Test :mod:`src.hp_optimization` module algorithms."""
import unittest

from unittest import TestCase, mock

from src.hp_optimization import ga, grid, rand


class TestRand(TestCase):
    """Test the rand optimizer module."""

    @staticmethod
    @mock.patch("src.hp_optimization.rand.common")
    def test_run_search(mock_common):
        """Test the search."""
        mock_output_cv = mock.Mock()
        mock_common.cross_validate.return_value = mock_output_cv
        rand.run_search()

        mock_common.RandomizedSearchCV.assert_called_once()
        mock_common.cross_validate.assert_called_once()
        mock_common.save_result.assert_called_with(mock_output_cv)

    @staticmethod
    def test_main():
        """Test the main function."""
        with mock.patch(
            "src.hp_optimization.rand.run_search"
        ) as mock_run_search:
            rand.main()

        mock_run_search.assert_called_once()


class TestGrid(TestCase):
    """Test the grid optimizer module."""

    @staticmethod
    @mock.patch("src.hp_optimization.grid.common")
    def test_run_search(mock_common):
        """Test the search."""
        mock_output_cv = mock.Mock()
        mock_common.cross_validate.return_value = mock_output_cv
        grid.run_search()

        mock_common.GridSearchCV.assert_called_once()
        mock_common.cross_validate.assert_called_once()
        mock_common.save_result.assert_called_with(mock_output_cv)

    @staticmethod
    def test_main():
        """Test the main function."""
        with mock.patch(
            "src.hp_optimization.grid.run_search"
        ) as mock_run_search:
            grid.main()

        mock_run_search.assert_called_once()


class TestGa(TestCase):
    """Test the ga optimizer module."""

    @staticmethod
    @mock.patch("src.hp_optimization.ga.common")
    def test_run_search(mock_common):
        """Test the search."""
        mock_output_cv = mock.Mock()
        mock_common.cross_validate.return_value = mock_output_cv
        ga.run_search()

        mock_common.EvolutionaryAlgorithmSearchCV.assert_called_once()
        mock_common.cross_validate.assert_called_once()
        mock_common.save_result.assert_called_with(mock_output_cv)

    @staticmethod
    def test_main():
        """Test the main function."""
        with mock.patch(
            "src.hp_optimization.ga.run_search"
        ) as mock_run_search:
            ga.main()

        mock_run_search.assert_called_once()


if __name__ == "__main__":
    unittest.main()
