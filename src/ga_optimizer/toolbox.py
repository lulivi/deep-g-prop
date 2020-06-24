"""DEAP toolbox configuration.

Setup the initializer, evaluator, crossover and mutation operators.

"""
import random
import time

from typing import Callable, Dict, List

import numpy as np

from deap import base, creator, tools
from keras.layers import Dense
from keras.losses import CategoricalCrossentropy
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.metrics import accuracy_score, fbeta_score

from src.dgp_logger import DGPLOGGER
from src.ga_optimizer.types import HiddenLayerInfo, Layer, MLPIndividual
from src.utils import Proben1Partition, Proben1Split


def individual_initializer(
    individual_class: Callable,
    model_input: int,
    hidden_layer_sequence: List[int],
    model_output: int,
):
    """Initialize an individual with uniform.

    :func:`np.random.uniform` function is used to obtain the new individual
    weights and bias.

    :param individual_class: individual class.
    :param model: :class:`Sequential` model from which obtain the
    layer configuration and weights..

    """
    return individual_class(model_input, hidden_layer_sequence, model_output)


def individual_evaluator(
    individual: MLPIndividual, trn: Proben1Split, tst: Proben1Split, **kwargs,
):
    """Evaluate an individual.

    :param individual: current individual to evaluate.
    :param trn: training data and labels.
    :param tst: validation data and labels.
    :param **kwargs: See below.
    :Keyword Arguments:
        - fit_train: whether to fit the train data with some forward pases
            before predicting.
        - average: average approach for :func:`fbeta_score` scorer.
    :returns: a tuple with the scores.

    """
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
        optimizer=SGD(), loss=CategoricalCrossentropy(),
    )

    # Train if choosen
    if kwargs.pop("fit_train", False):
        model.fit(
            trn.X, trn.y_cat, epochs=100, batch_size=16, verbose=0,
        )

    # Predict the scores
    predicted_y = model.predict_classes(tst.X)
    f2_score = fbeta_score(
        tst.y,
        predicted_y,
        beta=2,
        average=kwargs.pop(
            "average", "micro" if trn.y_cat.shape[1] > 2 else "binary"
        ),
    )
    accuracy = accuracy_score(tst.y, predicted_y, normalize=True)
    error_perc = (1.0 - accuracy) * 100
    end_time = time.perf_counter()
    DGPLOGGER.debug(
        f"        f2-score={f2_score:.5f}\n"
        f"        error%={error_perc:.2f}\n"
        f"        eval time={end_time - start_time: .2f} sec"
    )

    return (f2_score, error_perc)


def crossover_operator(ind1: MLPIndividual, ind2: MLPIndividual):
    """Apply crossover betweent two individuals.

    This method will swap a random neuron from a random layer. The neuron
    associated bias and weights are swapped.

    :param ind1: the first individual.
    :param ind2: the second individual.

    """
    layer_index = random.randint(0, len(ind1) - 1)
    neuron_index = random.randint(0, len(ind1.layers[layer_index].bias) - 1)

    (
        ind1.layers[layer_index].weights[:, neuron_index],
        ind2.layers[layer_index].weights[:, neuron_index],
    ) = (
        ind2.layers[layer_index].weights[:, neuron_index].copy(),
        ind1.layers[layer_index].weights[:, neuron_index].copy(),
    )
    (
        ind1.layers[layer_index].bias[neuron_index],
        ind2.layers[layer_index].bias[neuron_index],
    ) = (
        ind2.layers[layer_index].bias[neuron_index],
        ind1.layers[layer_index].bias[neuron_index],
    )


def layer_mutator(individual: MLPIndividual) -> int:
    """Add/remove one layer to the model.

    :param individual: individual to mutate.
    :return: wether the layer was added or removed.

    """
    # Choose a random number of neurons
    new_layer_output_neurons = random.randint(2, 5)

    # Choose randomly to add or delete a layer
    choice = random.choice((-1, 1))
    difference = 0

    if choice > 0:
        previous_layer_output = individual.layers[-2].config["units"]
        # Insert a new hidden layer into the individual
        individual.append_hidden(
            Layer.uniform(
                name=f"Hidden{len(individual)}",
                input_neurons=previous_layer_output,
                output_neurons=new_layer_output_neurons,
                trainable=True,
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
    # Ensure there are 2 or more layers in the model before deleting one
    elif len(individual) > 2:
        # Obtain the predecessor output units and delte the choosen layer
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
    else:
        choice = 0

    # Update output layer input neurons
    individual.layers[-1].config["batch_input_shape"][1] += difference

    return choice


def neuron_mutator(individual: MLPIndividual) -> int:
    """Add/remove one neuron from a random hidden layer.

    Randomly choose whether to add or remove a neuron.

    :param individual: individual to mutate.
    :returns: whether the neuron was added or removed.

    """
    # We want to ignore output layer so it only adds/pops from a hidden layer
    layer_index = random.randint(0, len(individual) - 2)

    # Choose randomly to add or delete a neuron
    choice = random.choice((-1, 1))

    if choice > 0:
        # Get previous layer neurons as a reference for creating a new neuron
        # for this layer
        previous_layer_neurons = individual.layers[layer_index].weights.shape[
            0
        ]
        # Append a new neuron to the weights and bias of the choosen layer
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
        # Append a new input entry for the choosen layer in the following layer
        next_layer_neurons = len(individual.layers[layer_index + 1].bias)
        individual.layers[layer_index + 1].weights = np.append(
            individual.layers[layer_index + 1].weights,
            np.random.uniform(-0.5, 0.5, (1, next_layer_neurons)),
            axis=0,
        )
    # Ensure there are 2 or more neurons in the hidden layer
    elif len(individual.layers[layer_index].bias) > 2:
        # Remove last neuron weights and bias from the choosen layer
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
    else:
        choice = 0

    # Update the units in the choosen and next layer config
    individual.layers[layer_index].config["units"] += choice
    individual.layers[layer_index + 1].config["batch_input_shape"][1] += choice

    return choice


def bias_mutator(individual: MLPIndividual, gen_prob: float) -> int:
    """Mutate some individual bias genes.

    For each layer bias, obtain a random :class:`np.ndarray`(with values in the
    range ``[0.0 and 1.0]``) with the same shape as the bias (in this case, a
    1D numpy array) and mutate the genes that satisfy the ``gen_prob``
    probability with a value in the range ``[-0.5, 0.5]``

    :param individual: individual to mutate.
    :param gen_prob: probability of a gen to mutate.
    :returns: number of genes mutated.

    """
    mutated_genes = 0

    for layer in individual.layers:
        if not layer.config["trainable"]:
            continue
        mask = np.random.rand(*layer.bias.shape) < gen_prob
        mutated_genes += np.count_nonzero(mask)
        mutations = np.random.uniform(-0.5, 0.5, layer.bias.shape)
        mutations[~mask] = 0
        layer.bias += mutations

    return mutated_genes


def weights_mutator(individual: MLPIndividual, gen_prob: float) -> int:
    """Mutate some individual weights genes.

    For each layer weights, obtain a random :class:`np.ndarray`(with values in
    the range ``[0.0 and 1.0]``) with the same shape as the weights and mutate
    the genes that satisfy the ``gen_prob`` probability with a value in the
    range ``[-0.5, 0.5]``

    :param individual: individual to mutate.
    :param gen_prob: probability of a gen to mutate.
    :returns: number of genes mutated.

    """
    mutated_genes = 0

    for layer in individual.layers:
        if not layer.config["trainable"]:
            continue
        mask = np.random.rand(*layer.weights.shape) < gen_prob
        mutated_genes += np.count_nonzero(mask)
        mutations = np.random.uniform(-0.5, 0.5, layer.weights.shape)
        mutations[~mask] = 0
        layer.weights += mutations

    return mutated_genes


# pylint: disable=no-member
def configure_toolbox(
    hidden_layers_info: List[HiddenLayerInfo],
    dataset: Proben1Partition,
    probabilities: Dict[str, float],
    fit_train: bool,
):
    """Register all neccesary objects and functions.

    :param hidden_layers_info: list of hidden layers basic configuration.
    :param dataset: data to work with.
    :param probabilities: list of mutation probabilities for weights and genes.
    :param fit_train: whether to fit the training data in each evaluation.
    :returns: the toolbox with the registered functions.

    """
    # --------------------------------
    # Individual registration
    # --------------------------------
    DGPLOGGER.debug("-- Register necessary functions and elements")
    DGPLOGGER.debug("Register the fitness measure...")
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -0.5))

    DGPLOGGER.debug("Register the individual...")
    creator.create("Individual", MLPIndividual, fitness=creator.FitnessMulti)
    toolbox = base.Toolbox()

    DGPLOGGER.debug("Register the individual initializer...")
    toolbox.register(
        "individual",
        individual_initializer,
        creator.Individual,
        dataset.nin,
        hidden_layers_info,
        dataset.nout,
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
        average="micro" if dataset.nout > 2 else "binary",
        fit_train=fit_train,
    )

    DGPLOGGER.debug("Register the crossover operator...")
    toolbox.register("crossover", crossover_operator)

    DGPLOGGER.debug("Register the weights mutate operator...")
    toolbox.register(
        "mutate_weights", weights_mutator, gen_prob=probabilities["weights"]
    )

    DGPLOGGER.debug("Register the bias mutate operator...")
    toolbox.register(
        "mutate_bias", bias_mutator, gen_prob=probabilities["bias"]
    )

    DGPLOGGER.debug("register the neuron mutator operator")
    toolbox.register("mutate_neuron", neuron_mutator)

    DGPLOGGER.debug("register the layer mutator operator")
    toolbox.register("mutate_layer", layer_mutator)

    DGPLOGGER.debug("Register the selector function...")
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox