"""Test the :mod:`src.ga_optimizer` module."""
import unittest

from unittest import mock

import numpy as np

from src import ga_optimizer
from src.utils import read_proben1_partition


class TestLayer(unittest.TestCase):
    """Tests for the Layer class."""

    def test_init(self):
        """Test the class constructor."""
        test_config = {"name": "layer name", "units": 3}
        test_weights = mock.Mock(spec=np.ndarray)
        test_bias = mock.Mock(spec=np.ndarray)
        layer = ga_optimizer.types.Layer(test_config, test_weights, test_bias)

        self.assertDictEqual(layer.config, test_config)
        self.assertEqual(layer.weights, test_weights)
        self.assertEqual(layer.bias, test_bias)

    def test_uniform(self):
        """Test the class constructor with uniform distribution."""
        test_name = "layer1"
        test_in = 2
        test_out = 1
        test_trainable = True
        layer = ga_optimizer.types.Layer.uniform(
            test_name, test_in, test_out, test_trainable
        )

        self.assertDictContainsSubset(
            {
                "name": test_name,
                "trainable": test_trainable,
                "batch_input_shape": [None, test_in],
                "units": test_out,
            },
            layer.config,
        )
        self.assertTupleEqual(layer.weights.shape, (test_in, test_out))
        self.assertTupleEqual(layer.bias.shape, (test_out,))
        self.assertGreater(np.count_nonzero(layer.weights), 0)

    def test_zeros(self):
        """Test the class constructor zeros for weights."""
        test_name = "layer1"
        test_in = 2
        test_out = 1
        test_trainable = True
        layer = ga_optimizer.types.Layer.zeros(
            test_name, test_in, test_out, test_trainable
        )

        self.assertDictContainsSubset(
            {
                "name": test_name,
                "trainable": test_trainable,
                "batch_input_shape": [None, test_in],
                "units": test_out,
            },
            layer.config,
        )
        self.assertTupleEqual(layer.weights.shape, (test_in, test_out))
        self.assertTupleEqual(layer.bias.shape, (test_out,))
        self.assertEqual(np.count_nonzero(layer.weights), 0)

    def test_get_weights(self):
        """Test the get_weights method."""
        test_config = {"name": "layer name", "units": 3}
        test_weights = mock.Mock(spec=np.ndarray)
        test_bias = mock.Mock(spec=np.ndarray)
        layer = ga_optimizer.types.Layer(test_config, test_weights, test_bias)

        result = layer.get_weights()
        self.assertListEqual(result, [test_weights, test_bias])

    def test_set_weights(self):
        """Test the set_weights method."""
        test_config = {"name": "layer name", "units": 3}
        test_weights = mock.Mock(spec=np.ndarray)
        test_bias = mock.Mock(spec=np.ndarray)
        layer = ga_optimizer.types.Layer(test_config, test_weights, test_bias)

        new_weights = mock.Mock(spec=np.ndarray)
        new_bias = mock.Mock(spec=np.ndarray)
        layer.set_weights([new_weights, new_bias])

        self.assertEqual(layer.weights, new_weights)
        self.assertEqual(layer.bias, new_bias)


class TestMLPIndividual(unittest.TestCase):
    """Tests for the MLPIndividual class."""

    @mock.patch("src.ga_optimizer.types.Layer")
    def test_init(self, mock_layer):
        """Test the constructor."""
        test_in = 3
        test_hidden = [ga_optimizer.types.HiddenLayerInfo(2, True)]
        test_out = 2
        individual = ga_optimizer.types.MLPIndividual(
            test_in, test_hidden, test_out
        )

        self.assertEqual(
            individual.layers[0].config, mock_layer.uniform.return_value.config
        )
        self.assertEqual(
            individual.layers[1].weights,
            mock_layer.uniform.return_value.weights,
        )

    def test_len(self):
        """Test the len magic method."""
        test_in = 3
        test_hidden = [ga_optimizer.types.HiddenLayerInfo(2, True)]
        test_out = 2
        individual = ga_optimizer.types.MLPIndividual(
            test_in, test_hidden, test_out
        )

        self.assertEqual(len(individual.layers), 2)

    def test_append(self):
        """Test the layer adder."""
        test_in = 3
        test_hidden = [ga_optimizer.types.HiddenLayerInfo(2, True)]
        test_out = 2
        individual = ga_optimizer.types.MLPIndividual(
            test_in, test_hidden, test_out
        )

        self.assertEqual(len(individual.layers), 2)

        test_layer_name = "hehe"
        test_layer_in = 2
        test_layer_out = 5
        test_layer_trainable = False
        individual.append_hidden(
            ga_optimizer.types.Layer.zeros(
                test_layer_name,
                test_layer_in,
                test_layer_out,
                test_layer_trainable,
            )
        )

        self.assertEqual(len(individual.layers), 3)

    def test_can_mate_ok(self):
        """Check if two individuals can mate."""
        test_in_1 = 3
        test_hidden_1 = [ga_optimizer.types.HiddenLayerInfo(2, True)]
        test_out_1 = 2
        individual_1 = ga_optimizer.types.MLPIndividual(
            test_in_1, test_hidden_1, test_out_1
        )

        test_in_2 = 3
        test_hidden_2 = [ga_optimizer.types.HiddenLayerInfo(2, True)]
        test_out_2 = 2
        individual_2 = ga_optimizer.types.MLPIndividual(
            test_in_2, test_hidden_2, test_out_2
        )

        self.assertTrue(individual_1.can_mate(individual_2))

    def test_can_mate_diff_len(self):
        """Check if two individuals can mate."""
        test_in_1 = 3
        test_hidden_1 = [
            ga_optimizer.types.HiddenLayerInfo(2, True),
            ga_optimizer.types.HiddenLayerInfo(3, True),
        ]
        test_out_1 = 2
        individual_1 = ga_optimizer.types.MLPIndividual(
            test_in_1, test_hidden_1, test_out_1
        )

        test_in_2 = 3
        test_hidden_2 = [ga_optimizer.types.HiddenLayerInfo(2, True)]
        test_out_2 = 2
        individual_2 = ga_optimizer.types.MLPIndividual(
            test_in_2, test_hidden_2, test_out_2
        )

        self.assertFalse(individual_1.can_mate(individual_2))

    def test_can_mate_diff_shape(self):
        """Check if two individuals can mate."""
        test_in_1 = 3
        test_hidden_1 = [
            ga_optimizer.types.HiddenLayerInfo(4, True),
        ]
        test_out_1 = 2
        individual_1 = ga_optimizer.types.MLPIndividual(
            test_in_1, test_hidden_1, test_out_1
        )

        test_in_2 = 3
        test_hidden_2 = [ga_optimizer.types.HiddenLayerInfo(2, True)]
        test_out_2 = 2
        individual_2 = ga_optimizer.types.MLPIndividual(
            test_in_2, test_hidden_2, test_out_2
        )

        self.assertFalse(individual_1.can_mate(individual_2))

    def test_str(self):
        """Test the representation methods."""
        test_in = 3
        test_hidden = [
            ga_optimizer.types.HiddenLayerInfo(4, True),
        ]
        test_out = 2
        individual = ga_optimizer.types.MLPIndividual(
            test_in, test_hidden, test_out
        )

        self.assertEqual(repr(individual), str(individual))


class TestGeneticAlgorithm(unittest.TestCase):
    """Test the genetic_algorithm."""

    def test_genetic_algorithm(self):
        """Run the genetic algorithm a few times and check if it works."""
        first_weights, best_weights = ga_optimizer.genetic_algorithm(
            [ga_optimizer.types.HiddenLayerInfo(5, True)],
            self.dataset,
            15,
            5,
            0.5,
            0.2,
            0.75,
            0.3,
            0.3,
            0.1,
        )
        first_score = ga_optimizer.utils.test_individual(
            first_weights, self.dataset
        )
        best_score = ga_optimizer.utils.test_individual(
            best_weights, self.dataset
        )
        self.assertGreater(best_score[0], first_score[0])
        self.assertLess(best_score[1], first_score[1])

    def setUp(self):
        """Setup the model to run the algorithm."""
        self.dataset = read_proben1_partition("cancer1")


if __name__ == "__main__":
    unittest.main()
