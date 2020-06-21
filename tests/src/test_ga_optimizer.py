"""Test the :mod:`src.ga_optimizer` module."""
import unittest

from unittest import mock

import keras
import numpy as np

from settings import TESTS_DIR_PATH
from src import ga_optimizer
from src.utils import read_proben1_partition

TEST_MODEL = TESTS_DIR_PATH / "test_files" / "cancer_test_model.h5"


class TestLayer(unittest.TestCase):
    """Tests for the Layer class."""

    def test_init(self):
        """Test the class constructor."""
        test_config = {"name": "layer name", "units": 3}
        test_weights = mock.Mock(spec=np.ndarray)
        test_bias = mock.Mock(spec=np.ndarray)
        layer = ga_optimizer.Layer(test_config, test_weights, test_bias)

        self.assertDictEqual(layer.config, test_config)
        self.assertEqual(layer.weights, test_weights)
        self.assertEqual(layer.bias, test_bias)

    def test_get_weights(self):
        """Test the get_weights method."""
        test_config = {"name": "layer name", "units": 3}
        test_weights = mock.Mock(spec=np.ndarray)
        test_bias = mock.Mock(spec=np.ndarray)
        layer = ga_optimizer.Layer(test_config, test_weights, test_bias)

        result = layer.get_weights()
        self.assertListEqual(result, [test_weights, test_bias])

    def test_set_weights(self):
        """Test the set_weights method."""
        test_config = {"name": "layer name", "units": 3}
        test_weights = mock.Mock(spec=np.ndarray)
        test_bias = mock.Mock(spec=np.ndarray)
        layer = ga_optimizer.Layer(test_config, test_weights, test_bias)

        new_weights = mock.Mock(spec=np.ndarray)
        new_bias = mock.Mock(spec=np.ndarray)
        layer.set_weights([new_weights, new_bias])

        self.assertEqual(layer.weights, new_weights)
        self.assertEqual(layer.bias, new_bias)


class TestMLPIndividual(unittest.TestCase):
    """Tests for the MLPIndividual class."""

    def test_init(self):
        """Test the constructor."""
        mock_model = mock.Mock()
        mock_layer = mock.Mock()
        mock_layer.get_weights.return_value = [mock.Mock(), mock.Mock()]
        mock_layer.get_config.return_value = mock.Mock()
        mock_model.layers = [mock_layer]
        individual = ga_optimizer.MLPIndividual(mock_model)

        self.assertEqual(
            individual.layers[0].config,
            ga_optimizer.Layer(
                mock_layer.get_config(), *mock_layer.get_weights()
            ).config,
        )
        self.assertEqual(
            individual.layers[0].weights,
            ga_optimizer.Layer(
                mock_layer.get_config(), *mock_layer.get_weights()
            ).weights,
        )
        self.assertEqual(
            individual.layers[0].bias,
            ga_optimizer.Layer(
                mock_layer.get_config(), *mock_layer.get_weights()
            ).bias,
        )

    def test_len(self):
        """Test the len magic method."""
        mock_model = mock.Mock()
        mock_layer = mock.Mock()
        mock_layer.get_weights.return_value = [mock.Mock(), mock.Mock()]
        mock_layer.get_config.return_value = mock.Mock()
        mock_model.layers = [mock_layer]
        individual = ga_optimizer.MLPIndividual(mock_model)

        self.assertEqual(len(individual), 1)

    def test_append(self):
        """Test the layer adder."""
        mock_model = mock.Mock()
        mock_layer = mock.Mock()
        mock_layer.get_weights.return_value = [mock.Mock(), mock.Mock()]
        mock_layer.get_config.return_value = mock.Mock()
        mock_model.layers = [mock_layer]
        individual = ga_optimizer.MLPIndividual(mock_model)
        individual.append(mock_layer.get_config(), *mock_layer.get_weights())

        self.assertEqual(len(individual), 2)

    def test_can_mate_ok(self):
        """Check if two individuals can mate."""
        mock_model = mock.Mock()
        mock_layer = mock.Mock()
        mock_weights = mock.Mock(shape=(9, 4))
        mock_bias = mock.Mock(shape=(4,))
        mock_layer.get_weights.return_value = [mock_weights, mock_bias]
        mock_layer.get_config.return_value = mock.Mock()
        mock_model.layers = [mock_layer]
        individual_1 = ga_optimizer.MLPIndividual(mock_model)

        individual_2 = ga_optimizer.MLPIndividual(mock_model)

        self.assertTrue(individual_1.can_mate(individual_2))

    def test_can_mate_diff_len(self):
        """Check if two individuals can mate."""
        mock_model1 = mock.Mock()
        mock_layer1 = mock.Mock()
        mock_weights1 = mock.Mock(shape=(9, 4))
        mock_bias1 = mock.Mock(shape=(4,))
        mock_layer1.get_weights.return_value = [mock_weights1, mock_bias1]
        mock_layer1.get_config.return_value = mock.Mock()

        mock_model2 = mock.Mock()
        mock_layer2 = mock.Mock()
        mock_weights2 = mock.Mock(shape=(9, 4))
        mock_bias2 = mock.Mock(shape=(4,))
        mock_layer2.get_weights.return_value = [mock_weights2, mock_bias2]
        mock_layer2.get_config.return_value = mock.Mock()

        mock_model1.layers = [mock_layer1]
        individual_1 = ga_optimizer.MLPIndividual(mock_model1)
        mock_model2.layers = [mock_layer1, mock_layer2]
        individual_2 = ga_optimizer.MLPIndividual(mock_model2)

        self.assertFalse(individual_1.can_mate(individual_2))

    def test_can_mate_diff_shape(self):
        """Check if two individuals can mate."""
        mock_model1 = mock.Mock()
        mock_layer1 = mock.Mock()
        mock_weights1 = mock.Mock(shape=(9, 4))
        mock_bias1 = mock.Mock(shape=(4,))
        mock_layer1.get_weights.return_value = [mock_weights1, mock_bias1]
        mock_layer1.get_config.return_value = mock.Mock()

        mock_model2 = mock.Mock()
        mock_layer2 = mock.Mock()
        mock_weights2 = mock.Mock(shape=(8, 4))
        mock_bias2 = mock.Mock(shape=(4,))
        mock_layer2.get_weights.return_value = [mock_weights2, mock_bias2]
        mock_layer2.get_config.return_value = mock.Mock()

        mock_model1.layers = [mock_layer1]
        individual_1 = ga_optimizer.MLPIndividual(mock_model1)
        mock_model2.layers = [mock_layer2]
        individual_2 = ga_optimizer.MLPIndividual(mock_model2)

        self.assertFalse(individual_1.can_mate(individual_2))

    def test_str(self):
        """Test the representation methods."""
        mock_model = mock.Mock()
        mock_layer = mock.Mock()
        mock_layer.get_weights.return_value = [mock.Mock(), mock.Mock()]
        mock_layer.get_config.return_value = mock.Mock()
        mock_model.layers = [mock_layer]
        individual = ga_optimizer.MLPIndividual(mock_model)

        self.assertEqual(repr(individual), str(individual))


class TestGeneticAlgorithm(unittest.TestCase):
    """Test the genetic_algorithm."""

    def test_genetic_algorithm_no_train(self):
        """Run the genetic algorithm a few times and check if it works."""
        first_weights, best_weights = ga_optimizer.genetic_algorithm(
            self.model, self.dataset, 10, 10, fit_train=False
        )
        first_score = ga_optimizer.test_individual(
            first_weights, self.dataset, fit_train=False
        )
        best_score = ga_optimizer.test_individual(
            best_weights, self.dataset, fit_train=False
        )
        self.assertGreater(best_score, first_score)

    def test_genetic_algorithm_train(self):
        """Run the genetic algorithm a few times and check if it works."""
        first_weights, best_weights = ga_optimizer.genetic_algorithm(
            self.model, self.dataset, 5, 3, fit_train=True
        )
        first_score = ga_optimizer.test_individual(
            first_weights, self.dataset, fit_train=True
        )
        best_score = ga_optimizer.test_individual(
            best_weights, self.dataset, fit_train=True
        )
        self.assertGreater(best_score, first_score)

    def setUp(self):
        """Setup the model to run the algorithm."""
        self.dataset = read_proben1_partition("cancer1")
        self.model = keras.models.load_model(TEST_MODEL)


if __name__ == "__main__":
    unittest.main()
