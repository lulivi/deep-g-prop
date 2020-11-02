"""DEAP toolbox configuration.

Setup the initializer, evaluator, crossover and mutation operators.

"""
import random
import time

from os import environ
from typing import Callable, Tuple, Union

import numpy as np

from deap import base, creator, tools
from sklearn.metrics import accuracy_score, fbeta_score

from src.dgp_logger import DGPLOGGER
from src.ga_optimizer.types import Layer, MLPIndividual
from src.utils import Proben1Partition, Proben1Split

environ["KERAS_BACKEND"] = "theano"

# pylint: disable=C0411,C0413
from keras.layers import Dense  # noqa: E402  # isort:skip

# pylint: disable=C0411,C0413
from keras.losses import (  # noqa: E402  # isort:skip
    BinaryCrossentropy,
    CategoricalCrossentropy,
)

# pylint: disable=C0411,C0413
from keras.models import Sequential  # noqa: E402  # isort:skip

# pylint: disable=C0411,C0413
from keras.optimizers import SGD  # noqa: E402  # isort:skip


def individual_initializer(
    individual_class: Callable,
    nin_nout: Tuple[int, int],
    neuron_layer_ranges: Tuple[Tuple[int, int], Tuple[int, int]],
    constant_hidden_layers: int,
):
    """Initialize an individual with uniform.

    :param individual_class: individual class.
    :param model_input: number of neurons in the input layer.
    :param max_neurons: top limit for the random neurons generator.
    :param max_layers: top limit for the random layers generator.
    :param model_output: number of classes to predict.
    layer configuration and weights..

    """
    hidden_layers = np.random.randint(
        neuron_layer_ranges[0][0],
        neuron_layer_ranges[0][1] + 1,
        random.randint(*neuron_layer_ranges[1]),
    ).tolist()
    return individual_class(
        nin_nout[0], hidden_layers, constant_hidden_layers, nin_nout[1]
    )


def individual_evaluator(
    individual: MLPIndividual, trn: Proben1Split, tst: Proben1Split, **kwargs
):
    """Evaluate an individual.

    :param individual: current individual to evaluate.
    :param trn: training data and labels.
    :param tst: validation data and labels.
    :param multi_class: ``True`` if the dataset is for multiclass
        classification.
    :returns: the fitness values.

    """
    multi_class = kwargs.get("multi_class", False)
    start_time = time.perf_counter()
    units_size_list = [
        layer.config["units"] for layer in individual.layers[:-1]
    ]
    DGPLOGGER.debug(
        f"    Evaluating individual with neuron number: {units_size_list}"
    )
    # Create the model with the individual configuration
    model = Sequential()

    for layer_index, layer in enumerate(individual.layers):
        model.add(Dense.from_config(layer.config))
        model.layers[layer_index].set_weights([layer.weights, layer.bias])

    model.compile(
        optimizer=SGD(),
        loss=CategoricalCrossentropy()
        if multi_class
        else BinaryCrossentropy(),
    )

    model.fit(trn.X, trn.y_cat, epochs=100, batch_size=16, verbose=0)

    # Predict the scores
    predicted_y = model.predict_classes(tst.X)
    f2_score = fbeta_score(
        tst.y,
        predicted_y,
        beta=2,
        average="micro" if multi_class else "binary",
    )
    error_perc = (
        1.0 - accuracy_score(tst.y, predicted_y, normalize=True)
    ) * 100
    neuron_layer_score = sum(units_size_list) * len(units_size_list)
    DGPLOGGER.debug(
        f"        error%={error_perc:.2f}\n"
        f"        neuron/layer-score={neuron_layer_score:.2f}\n"
        f"        f2-score={f2_score:.5f}\n"
        f"        evaluation time={time.perf_counter() - start_time: .2f} sec"
    )

    return (error_perc, neuron_layer_score, f2_score)


def crossover_operator(ind1: MLPIndividual, ind2: MLPIndividual):
    """Apply crossover betweent two individuals.

    This method will swap neurons with two random points from a random layer.
    The neurons associated bias and weights are swapped.

    :param ind1: the first individual.
    :param ind2: the second individual.
    :returns: a tuple with  the cross points and the crossed layer.

    """
    # Choose randomly the layer index to swap. If the hidden layers of any of
    # the two individuals are constant, swap neurons from the output layer
    # neuron in the output layer.
    layer_index = (
        len(ind1) - 1
        if ind1.constant_hidden_layers or ind2.constant_hidden_layers
        else random.randint(0, len(ind1) - 1)
    )
    cx_pts = random.sample(range(len(ind1.layers[layer_index].bias)), 2)

    (
        ind1.layers[layer_index].weights[:, cx_pts[0] : cx_pts[1]],
        ind2.layers[layer_index].weights[:, cx_pts[0] : cx_pts[1]],
    ) = (
        ind2.layers[layer_index].weights[:, cx_pts[0] : cx_pts[1]].copy(),
        ind1.layers[layer_index].weights[:, cx_pts[0] : cx_pts[1]].copy(),
    )
    (
        ind1.layers[layer_index].bias[cx_pts[0] : cx_pts[1]],
        ind2.layers[layer_index].bias[cx_pts[0] : cx_pts[1]],
    ) = (
        ind2.layers[layer_index].bias[cx_pts[0] : cx_pts[1]].copy(),
        ind1.layers[layer_index].bias[cx_pts[0] : cx_pts[1]].copy(),
    )

    return cx_pts, layer_index


def layer_mutator(individual: MLPIndividual) -> int:
    """Add/remove one layer to the model.

    Compute whether to append a new hidden layer or pop the last one.

    :param individual: individual to mutate.
    :return: wether the layer was added or removed.

    """
    # Choose randomly to add or delete a layer. Ensure there are 2 or more
    # layers in the model before deleting one. The output layer is included in
    # the count.
    choice = 1 if len(individual) <= 2 else random.choice((-1, 1))

    difference = 0

    if choice > 0:
        # Choose a random number of neurons
        new_layer_output_neurons = random.randint(2, 5)
        # Obtain current last hidden layer neuron number
        previous_layer_output = individual.layers[-2].config["units"]
        # Insert a new hidden layer into the individual
        individual.append_hidden(
            Layer.uniform(
                name=f"Hidden{len(individual)}",
                input_neurons=previous_layer_output,
                output_neurons=new_layer_output_neurons,
            )
        )

        # Obtain the differences between the new layer neurons and the output
        # layer input neurons and apply necessary changes to this last one
        output_layer_input_neurons = individual.layers[-1].weights.shape[0]
        difference = new_layer_output_neurons - output_layer_input_neurons

        # Add input neuron entries
        if difference > 0:
            next_layer_neurons = len(individual.layers[-1].bias)
            individual.layers[-1].weights = np.append(
                individual.layers[-1].weights,
                np.random.uniform(-1.0, 1.0, (difference, next_layer_neurons)),
                axis=0,
            )
        # Remove input neuron entries
        elif difference < 0:
            individual.layers[-1].weights = np.delete(
                individual.layers[-1].weights,
                slice(
                    output_layer_input_neurons + difference,
                    output_layer_input_neurons,
                ),
                axis=0,
            )
    else:
        # Obtain the predecessor output units and delte the chosen layer
        removed_predecessor_units = individual.layers[-3].config["units"]
        del individual.layers[-2]

        # Calculate the difference between the predecesor layer and the output
        # layer
        output_layer_input_len = individual.layers[-1].weights.shape[0]
        difference = removed_predecessor_units - output_layer_input_len

        # Append the neccesary input neuron entries
        if difference > 0:
            next_layer_neurons = len(individual.layers[-1].bias)
            individual.layers[-1].weights = np.append(
                individual.layers[-1].weights,
                np.random.uniform(-0.5, 0.5, (difference, next_layer_neurons)),
                axis=0,
            )
        # Remove the leftovers
        elif difference < 0:
            individual.layers[-1].weights = np.delete(
                individual.layers[-1].weights,
                slice(
                    output_layer_input_len + difference, output_layer_input_len
                ),
                axis=0,
            )

    # Update output layer input neurons
    individual.layers[-1].config["batch_input_shape"][1] += difference

    return choice


def neuron_mutator(individual: MLPIndividual) -> int:
    """Add/remove one neuron from a random hidden layer.

    For a random layer append a new neuron or pop the last one.

    :param individual: individual to mutate.
    :returns: whether the neuron was added or removed.

    """
    # We want to ignore output layer so it only adds/pops from a hidden layer
    layer_index = random.randint(0, len(individual) - 2)

    # Choose randomly to add or delete a neuron. If the number of neurons is
    # two, just add a new one.
    choice = (
        1
        if len(individual.layers[layer_index].bias) <= 2
        else random.choice((-1, 1))
    )

    if choice > 0:
        # Get previous layer neurons as a reference for creating a new neuron
        # for this layer
        previous_layer_neurons = individual.layers[layer_index].weights.shape[
            0
        ]
        # Append a new neuron to the weights and bias of the chosen layer
        individual.layers[layer_index].weights = np.append(
            individual.layers[layer_index].weights,
            np.random.uniform(-0.5, 0.5, (previous_layer_neurons, 1)),
            axis=1,
        )
        individual.layers[layer_index].bias = np.append(
            individual.layers[layer_index].bias,
            [random.uniform(-0.5, 0.5)],
            axis=0,
        )
        # Append a new input entry for the chosen layer in the following layer
        next_layer_neurons = len(individual.layers[layer_index + 1].bias)
        individual.layers[layer_index + 1].weights = np.append(
            individual.layers[layer_index + 1].weights,
            np.random.uniform(-0.5, 0.5, (1, next_layer_neurons)),
            axis=0,
        )
    else:
        # Remove last neuron weights and bias from the chosen layer
        individual.layers[layer_index].weights = np.delete(
            individual.layers[layer_index].weights, -1, axis=1
        )
        individual.layers[layer_index].bias = np.delete(
            individual.layers[layer_index].bias, -1, axis=0
        )
        # Remove the input neuron from the next layer
        individual.layers[layer_index + 1].weights = np.delete(
            individual.layers[layer_index + 1].weights, -1, axis=0
        )

    # Update the units in the chosen and next layer config
    individual.layers[layer_index].config["units"] += choice
    individual.layers[layer_index + 1].config["batch_input_shape"][1] += choice

    return choice


def weights_mutator(
    individual: MLPIndividual, attribute: str, gen_prob: float
) -> int:
    """Mutate some individual weights genes.

    For each layer weights or bias, obtain a random :class:`np.ndarray`(with
    values in the range ``[0.0 and 1.0]``) with the same shape as the selected
    attribute and mutate the genes that satisfy the ``gen_prob`` probability
    with a value in the range ``[-0.5, 0.5]``

    :param individual: individual to mutate.
    :param attribute: attribute to mutate. Must be either ``weights`` or
        ``bias``.
    :param gen_prob: probability of a gen to mutate.
    :returns: number of genes mutated.

    """
    mutated_genes = 0

    layer_list = (
        [individual.layers[-1]]
        if individual.constant_hidden_layers
        else individual.layers
    )

    for layer in layer_list:
        weights = getattr(layer, attribute)
        weights_shape = weights.shape

        mask = np.random.rand(*weights_shape) < gen_prob
        mutated_genes += np.count_nonzero(mask)
        mutations = np.random.uniform(-0.5, 0.5, weights_shape)
        mutations[~mask] = 0
        weights += mutations

    return mutated_genes


# pylint: disable=no-member
def configure_toolbox(
    dataset: Proben1Partition, **config: Union[float, bool, tuple]
):
    r"""Register all neccesary objects and functions.

    :param dataset: data to work with.
    :param \**config: diversed configuration parameters.

    :Keyword Arguments:
        - *neurons_range* --  max and min values given for the neurons random
            generator for each layer.
        - *layers_range* --  max and min values given for the layers random
            generator.
        - *mut_bias_prob* --  probability to mutate the individual bias
            genes.
        - *mut_weights_prob* --  probability to mutate the individual weights
            genes.
        - *const_hidden_layers* --  ``True`` if the no crossover or mutation
            can be applied to the hidden layers.

    :returns: the toolbox with the registered functions.

    """
    # --------------------------------
    # Individual registration
    # --------------------------------
    DGPLOGGER.debug("-- Register necessary functions and elements")
    DGPLOGGER.debug("Register the fitness measure...")
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -0.5, 0.5))

    DGPLOGGER.debug("Register the individual...")
    creator.create("Individual", MLPIndividual, fitness=creator.FitnessMulti)
    toolbox = base.Toolbox()

    DGPLOGGER.debug("Register the individual initializer...")
    toolbox.register(
        "individual",
        individual_initializer,
        creator.Individual,
        (dataset.nin, dataset.nout),
        (config["neurons_range"], config["layers_range"]),
        config["const_hidden_layers"],
    )

    # define the population to be a list of individuals
    DGPLOGGER.debug("Register the population initializer...")
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # --------------------------------
    # Operator registration
    # --------------------------------
    DGPLOGGER.debug("Register the evaluator function...")
    toolbox.register(
        "evaluate",
        individual_evaluator,
        trn=dataset.trn,
        tst=dataset.val,
        multi_class=dataset.nout > 2,
    )

    DGPLOGGER.debug("Register the crossover operator...")
    toolbox.register("crossover", crossover_operator)

    DGPLOGGER.debug("Register the bias mutate operator...")
    toolbox.register(
        "mutate_bias",
        weights_mutator,
        attribute="bias",
        gen_prob=config["mut_bias_prob"],
    )

    DGPLOGGER.debug("Register the weights mutate operator...")
    toolbox.register(
        "mutate_weights",
        weights_mutator,
        attribute="weights",
        gen_prob=config["mut_weights_prob"],
    )

    DGPLOGGER.debug("register the neuron mutator operator")
    toolbox.register("mutate_neuron", neuron_mutator)

    DGPLOGGER.debug("register the layer mutator operator")
    toolbox.register("mutate_layer", layer_mutator)

    return toolbox
