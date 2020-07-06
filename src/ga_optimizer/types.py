"""GA individual and misc object classes."""
from typing import Any, Dict, List

import numpy as np


class Layer:
    """Keras information to create a :class:`Dense` layer.

    :ivar config: layer configuration.
    :ivar weights: layer weights array.
    :ivar bias: layer bias array.

    """

    def __init__(
        self, config: Dict[str, Any], weights: np.ndarray, bias: np.ndarray,
    ) -> None:
        """Create a new Layer object.

        :param config: layer configuration.
        :param weights: layer weights array.
        :param bias: layer bias array.

        """
        self.config: Dict[str, Any] = config
        self.weights: np.ndarray = weights
        self.bias: np.ndarray = bias

    # pylint: disable=too-many-arguments
    @classmethod
    def uniform(
        cls,
        name: str,
        input_neurons: int,
        output_neurons: int,
        activation: str = "relu",
    ):
        """Create a new layer using a :func:`np.random.uniform` distribution.

        :param name: hidden layer identifier.
        :param input_neurons: number of input neurons.
        :param output_neurons: number of output neurons.
        :param activation: layer activation.
        :returns: the newly created class instance.

        """
        weights = np.random.uniform(
            -1.0, 1.0, size=(input_neurons, output_neurons)
        )
        bias = np.random.uniform(-1.0, 1.0, size=(output_neurons,))
        config = cls._get_layer_config(
            name, input_neurons, output_neurons, activation
        )
        return cls(config, weights, bias)

    # pylint: disable=too-many-arguments
    @classmethod
    def zeros(
        cls,
        name: str,
        input_neurons: int,
        output_neurons: int,
        activation: str = "relu",
    ):
        """Create a new layer using the :func:`np.zeros` function.

        :param name: hidden layer identifier.
        :param input_neurons: number of input neurons.
        :param output_neurons: number of output neurons.
        :param activation: layer activation.
        :returns: the newly created class instance.

        """
        weights = np.zeros((input_neurons, output_neurons))
        bias = np.zeros((output_neurons,))
        config = cls._get_layer_config(
            name, input_neurons, output_neurons, activation
        )
        return cls(config, weights, bias)

    @staticmethod
    def _get_layer_config(
        name: str, input_neurons: int, output_neurons: int, activation: str,
    ):
        """Obtain a :class:`Dense` class configuration.

        :param name: hidden layer identifier.
        :param input_neurons: number of input neurons.
        :param output_neurons: number of output neurons.
        :returns: the created configuration.

        """
        return {
            "name": name,
            "trainable": True,
            "batch_input_shape": [None, input_neurons],
            "dtype": "float32",
            "units": output_neurons,
            "activation": activation,
            "use_bias": True,
            "kernel_initializer": {"class_name": "Zeros", "config": {}},
            "bias_initializer": {"class_name": "Zeros", "config": {}},
            "kernel_regularizer": None,
            "bias_regularizer": None,
            "activity_regularizer": None,
            "kernel_constraint": None,
            "bias_constraint": None,
        }

    def get_weights(self):
        """Simulate the :class:`Dense` method.

        :returns: weights and bias as a list.

        """
        return [self.weights, self.bias]

    def set_weights(self, layer_weights: List[np.ndarray]):
        """Simulate the :class:`Dense` method.

        :param layer_weights: list with weights and bias.

        """
        self.weights, self.bias = layer_weights


class MLPIndividual:
    """Basic structure of a simple MLP weights."""

    def __init__(
        self,
        model_input: int,
        hidden_layers: List[int],
        constant_hidden_layers: bool,
        model_output: int,
    ) -> None:
        """Define the base model.

        :param model_input: number of caracteristic the data has.
        :param hidden_layer_sequence: a list of layers sizes.
        :param constant_hidden_layers: ``True`` if the hidden layers wouldn't
            be changed.
        :param model_output: number of output neurons.

        """
        self._layers: List[Layer] = []
        next_input = model_input

        for layer_index, neurons_number in enumerate(hidden_layers):
            self._layers.append(
                Layer.uniform(
                    name=f"Hidden{layer_index}",
                    input_neurons=next_input,
                    output_neurons=neurons_number,
                    activation="relu",
                )
            )
            next_input = neurons_number

        self._constant_hidden_layers = constant_hidden_layers

        self._layers.append(
            Layer.uniform(
                name="OutputLayer",
                input_neurons=next_input,
                output_neurons=model_output,
                activation="softmax",
            )
        )

    def __len__(self) -> int:
        """Calculate the number of layers the model has.

        :returns: the number of layers.

        """
        return len(self._layers)

    def append_hidden(self, layer: Layer) -> None:
        """Append a new hidden layer.

        :param layer: hidden layer to append.

        """
        self._layers.insert(len(self) - 1, layer)

    @property
    def layers(self) -> List[Layer]:
        """Obtain the individual list of :class:`Layer`.

        :returns: the list of layers.

        """
        return self._layers

    @property
    def constant_hidden_layers(self) -> bool:
        """Tell if the hidden layers are mutable.

        :returns: ``True`` if the hidden layers can change, ``False``
            elsewhere.

        """
        return self._constant_hidden_layers

    def can_mate(self, other: "MLPIndividual") -> bool:
        """Check if the current individual can mate with ``other``.

        :param other: other individual.
        :returns: ``True`` if both individuals can mate.

        """
        if len(self) != len(other):
            return False

        for layer, other_layer in zip(self.layers, other.layers):
            if (
                layer.weights.shape != other_layer.weights.shape
                or layer.bias.shape != other_layer.bias.shape
            ):
                return False

        return True

    def __str__(self) -> str:
        """Serialize as a string the object."""
        str_representation = (
            f"{self.__class__.__name__}:"
            "\n-- Constant hidden layers --"
            f"\n{str(self._constant_hidden_layers)}"
        )

        for index, layer in enumerate(self.layers):
            str_representation += (
                f"\nLayer {index}:"
                f"\n-- Config --\n{str(layer.config)}"
                f"\n-- Weights --\n{str(layer.weights)}"
                f"\n-- Bias --\n{str(layer.bias)}"
            )

        return str_representation

    def __repr__(self) -> str:
        """Serialize as a string the object."""
        return str(self)
