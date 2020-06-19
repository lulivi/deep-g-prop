"""Test the :mod:`src.ga_optimizer` module."""
import unittest

from unittest import mock

import keras
import numpy as np

from settings import TESTS_DIR_PATH
from src import ga_optimizer
from src.utils import read_proben1_partition

TEST_MODEL = TESTS_DIR_PATH / "test_files" / "cancer_test_model.h5"


class TestMLPIndividual(unittest.TestCase):
    """Tests for the MLPIndividual class."""

    def test_init(self):
        """Test the constructor."""
        individual = ga_optimizer.MLPIndividual()

        self.assertEqual(individual.bias, [])
        self.assertEqual(individual.weights, [])

    def test_len(self):
        """Test the len magic method."""
        individual = ga_optimizer.MLPIndividual()
        mock_weights = mock.Mock(spec=np.ndarray)
        mock_bias = mock.Mock(spec=np.ndarray)
        individual.add_layer(mock_weights, mock_bias)

        self.assertEqual(len(individual), 1)

    def test_add_layer(self):
        """Test the layer adder."""
        individual = ga_optimizer.MLPIndividual()
        mock_weights = mock.Mock(spec=np.ndarray)
        mock_bias = mock.Mock(spec=np.ndarray)

        self.assertEqual(individual.bias, [])

        individual.add_layer(mock_weights, mock_bias)

        self.assertListEqual(individual.bias, [mock_bias])
        self.assertListEqual(individual.weights, [mock_weights])

    def test_get_all(self):
        """Test the structure getter method."""
        individual = ga_optimizer.MLPIndividual()
        mock_weights = mock.Mock(spec=np.ndarray)
        mock_bias = mock.Mock(spec=np.ndarray)

        individual.add_layer(mock_weights, mock_bias)
        individual_structure = individual.get_all()

        self.assertEqual(len(individual_structure), 2)
        self.assertListEqual(individual_structure, [mock_weights, mock_bias])

    def test_can_mate_ok(self):
        """Check if two individuals can mate."""
        individual_1 = ga_optimizer.MLPIndividual()
        individual_1.add_layer(
            mock.Mock(spec=np.ndarray, shape=(2, 3)),
            mock.Mock(spec=np.ndarray, shape=(3,)),
        )
        individual_2 = ga_optimizer.MLPIndividual()
        individual_2.add_layer(
            mock.Mock(spec=np.ndarray, shape=(2, 3)),
            mock.Mock(spec=np.ndarray, shape=(3,)),
        )

        self.assertTrue(individual_1.can_mate(individual_2))

    def test_can_mate_diff_shape(self):
        """Check if two individuals can mate."""
        individual_1 = ga_optimizer.MLPIndividual()
        individual_1.add_layer(
            mock.Mock(spec=np.ndarray, shape=(2, 3)),
            mock.Mock(spec=np.ndarray, shape=(3,)),
        )
        individual_2 = ga_optimizer.MLPIndividual()
        individual_2.add_layer(
            mock.Mock(spec=np.ndarray, shape=(2, 2)),
            mock.Mock(spec=np.ndarray, shape=(2,)),
        )

        self.assertFalse(individual_1.can_mate(individual_2))

    def test_can_mate_diff_len(self):
        """Check if two individuals can mate."""
        individual_1 = ga_optimizer.MLPIndividual()
        individual_1.add_layer(
            mock.Mock(spec=np.ndarray, shape=(2, 3)),
            mock.Mock(spec=np.ndarray, shape=(3,)),
        )
        individual_2 = ga_optimizer.MLPIndividual()
        individual_2.add_layer(
            mock.Mock(spec=np.ndarray, shape=(2, 2)),
            mock.Mock(spec=np.ndarray, shape=(3,)),
        )
        individual_2.add_layer(
            mock.Mock(spec=np.ndarray, shape=(2, 2)),
            mock.Mock(spec=np.ndarray, shape=(3,)),
        )

        self.assertFalse(individual_1.can_mate(individual_2))

    def test_str(self):
        """Test the representation methods."""
        individual = ga_optimizer.MLPIndividual()
        mock_weights = mock.Mock(spec=np.ndarray)
        mock_bias = mock.Mock(spec=np.ndarray)
        individual.add_layer(mock_weights, mock_bias)

        self.assertEqual(repr(individual), str(individual))


class TestGeneticAlgorithm(unittest.TestCase):
    """Test the genetic_algorithm."""

    def test_genetic_algorithm_no_train(self):
        """Run the genetic algorithm a few times and check if it works."""
        first_weights, best_weights = ga_optimizer.genetic_algorithm(
            self.model, self.dataset, 10, 10, fit_train=False
        )
        first_score = ga_optimizer.test_individual(
            first_weights, self.model, self.dataset, fit_train=False
        )
        best_score = ga_optimizer.test_individual(
            best_weights, self.model, self.dataset, fit_train=False
        )
        self.assertGreater(best_score, first_score)

    def test_genetic_algorithm_train(self):
        """Run the genetic algorithm a few times and check if it works."""
        first_weights, best_weights = ga_optimizer.genetic_algorithm(
            self.model, self.dataset, 5, 3, fit_train=True
        )
        first_score = ga_optimizer.test_individual(
            first_weights, self.model, self.dataset, fit_train=True
        )
        best_score = ga_optimizer.test_individual(
            best_weights, self.model, self.dataset, fit_train=True
        )
        self.assertGreater(best_score, first_score)

    def setUp(self):
        """Setup the model to run the algorithm."""
        self.dataset = read_proben1_partition("cancer1")
        self.model = keras.models.load_model(TEST_MODEL)


if __name__ == "__main__":
    unittest.main()
