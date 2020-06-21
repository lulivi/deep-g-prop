"""Multilayer perceptron optimization via genetic algorithms."""
import random
import time

from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from deap import base, creator, tools
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import balanced_accuracy_score, fbeta_score

from src.common import SEED
from src.dgp_logger import DGPLOGGER
from src.types import Proben1Partition, Proben1Split
from src.utils import print_table

ModelStructure = List[Tuple[tuple, np.dtype]]
np.set_printoptions(precision=5, floatmode="fixed")


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

    def get_weights(self):
        """Simulate the :class:`Dense` method.

        :returns: weights and bias as a list.

        """
        return [self.weights, self.bias]

    def set_weights(self, layer_weights: List[np.ndarray]):
        """Simulate the :class:`Dense` method.

        :param layer_weights: list with weights and bias.

        """
        self.weights = layer_weights[0]
        self.bias = layer_weights[1]


class MLPIndividual:
    """Basic structure of a simple MLP weights."""

    def __init__(self, model: Sequential) -> None:
        """Define the base model.

        :param model: base mode from which to obtain the configuration and
            weights

        """
        self._layers: List[Layer] = []

        for layer in model.layers:
            layer_weights = layer.get_weights()
            self._layers.append(Layer(layer.get_config(), *layer_weights))

    def __len__(self) -> int:
        """Calculate the number of layers the model has.

        :returns: the number of layers.

        """
        return len(self._layers)

    def append(
        self, config: Dict[str, Any], weights: np.ndarray, bias: np.ndarray,
    ) -> None:
        """Append a new layer.

        :param config: layer configuration (name, units, initializers, ...).
        :param weights: layer weights.
        :param bias: layer bias.

        """
        self._layers.append(Layer(config, weights, bias))

    @property
    def layers(self) -> List[Layer]:
        """Obtain the individual list of :class:`Layer`.

        :returns: the list of layers.

        """
        return self._layers

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
        str_representation = f"{self.__class__.__name__}:"

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


###############################################################################
#
# Toolbox functions
#
###############################################################################


def individual_initializer(
    individual_class: Callable, model: Sequential,
):
    """Initialize an individual with uniform.

    :func:`np.random.uniform` function is used to obtain the new individual
    weights and bias.

    :param individual_class: individual class.
    :param model: :class:`Sequential` model from which obtain the
    layer configuration and weights..

    """
    individual = individual_class(model)

    for layer in individual.layers:
        layer.weights = np.random.uniform(
            low=-1.0, high=1.0, size=layer.weights.shape
        )
        layer.bias = np.random.uniform(
            low=-1.0, high=1.0, size=layer.bias.shape
        )

    return individual


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
    units_size_list = [layer.config["units"] for layer in individual.layers]
    DGPLOGGER.debug(
        f"Evaluating individual with neuron number: {units_size_list}"
    )
    # Create the model with the individual configuration
    model = Sequential()

    for layer_index, layer in enumerate(individual.layers):
        model.add(Dense.from_config(layer.config))
        model.layers[layer_index].set_weights([layer.weights, layer.bias])

    model.compile(optimizer="sgd", loss="binary_crossentropy")

    # Train if choosen
    if kwargs.pop("fit_train", False):
        model.fit(
            trn.X, trn.y_cat, epochs=20, batch_size=8, verbose=0,
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
    bal_accuracy_score = balanced_accuracy_score(tst.y, predicted_y)
    end_time = time.perf_counter()
    DGPLOGGER.debug(
        f"    Obtained scores: f2-score={f2_score:.5f}, "
        f"balanced acc score={bal_accuracy_score:.5f}, in "
        f"{end_time - start_time: .2f} sec"
    )

    return (f2_score, bal_accuracy_score)


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
    neurons_number = random.randint(2, 5)

    # Choose randomly to add or delete a layer
    choice = random.choice((-1, 1))

    if choice > 0:
        # Create a new layer config and weights
        previous_layer_output = individual.layers[-2].config["units"]
        new_layer_config = {
            "name": f"HiddenLayer{len(individual)+1}",
            "trainable": True,
            "dtype": "float32",
            "units": neurons_number,
            "activation": "relu",
            "use_bias": True,
            "kernel_initializer": {"class_name": "Zeros", "config": {}},
            "bias_initializer": {"class_name": "Zeros", "config": {}},
            "kernel_regularizer": None,
            "bias_regularizer": None,
            "activity_regularizer": None,
            "kernel_constraint": None,
            "bias_constraint": None,
        }
        new_layer_weights = np.random.uniform(
            -1, 1, size=(previous_layer_output, neurons_number)
        )
        new_layer_bias = np.random.uniform(-1, 1, size=(neurons_number,))

        # Insert the recently created configuration and weights into our
        # individual
        individual.layers.insert(
            len(individual.layers) - 1,
            Layer(new_layer_config, new_layer_weights, new_layer_bias,),
        )

        # Obtain the differences between the new layer neurons and the output
        # layer input neurons and apply necessary changes to this last one
        next_layer_input_len = individual.layers[-1].weights.shape[0]
        difference = neurons_number - next_layer_input_len

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
                slice(next_layer_input_len + difference, next_layer_input_len),
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

    # Update the units in the layer config
    individual.layers[layer_index].config["units"] += choice

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

    for layer_index in range(len(individual)):
        mask = (
            np.random.rand(*individual.layers[layer_index].bias.shape)
            < gen_prob
        )
        mutated_genes += np.count_nonzero(mask)
        individual.layers[layer_index].bias[mask] += random.uniform(-0.5, 0.5)

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

    for layer_index in range(len(individual)):
        mask = (
            np.random.rand(*individual.layers[layer_index].weights.shape)
            < gen_prob
        )
        mutated_genes += np.count_nonzero(mask)
        individual.layers[layer_index].weights[mask] += random.uniform(
            -0.5, 0.5
        )

    return mutated_genes


# pylint: disable=no-member
def configure_toolbox(
    model: Sequential,
    dataset: Proben1Partition,
    probabilities: Dict[str, float],
    fit_train: bool,
):
    """Register all neccesary objects and functions.

    :param model: base model, from whom obtain the initial individual shape.
    :param dataset: data to work with.
    :param mut_bias_prob: probability to mutate the individual bias genes.
    :param mut_weights_prob: probability to mutate the individual weights
        genes.
    :param fit_train: whether to fit the training data in each evaluation.
    :returns: the toolbox with the registered functions.

    """
    # --------------------------------
    # Individual registration
    # --------------------------------
    DGPLOGGER.debug("-- Register necessary functions and elements")
    DGPLOGGER.debug("Register the fitness measure...")
    creator.create("FitnessMax", base.Fitness, weights=(1.0, 0.7))

    DGPLOGGER.debug("Register the individual...")
    creator.create("Individual", MLPIndividual, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    DGPLOGGER.debug("Register the individual initializer...")
    toolbox.register(
        "individual", individual_initializer, creator.Individual, model,
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


###############################################################################
#
# Util functions
#
###############################################################################


def evaluate_population(population: list, evaluate_fn: Callable) -> None:
    """Evaluate the population.

    :param population: list of individuals.
    :param evaluate_fn: function to evaluate the population.

    """
    fitnesses = map(evaluate_fn, population)

    for ind, fit in zip(population, fitnesses):

        ind.fitness.values = fit


def apply_crossover(
    population: list, cx_prob: float, crossover_fn: Callable
) -> int:
    """Crossover the population by 2 with a probability.

    :param population: list of individuals.
    :param cx_prob: mate probability.
    :param crossover_fn: function to mate two individuals.
    :returns: number of individual pairs mated.

    """
    crossed_individuals = 0

    for crossover_index, (child1, child2) in enumerate(
        zip(population[::2], population[1::2])
    ):
        if random.random() < cx_prob and child1.can_mate(child2):
            crossed_individuals += 1
            DGPLOGGER.debug(
                f"Applying crossover for the {crossover_index} couple "
                f"({crossover_index * 2}, {crossover_index * 2+1}))."
            )
            crossover_fn(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    return crossed_individuals


def apply_mutation(
    population: list,
    toolbox: base.Toolbox,
    mut_neuron_prob: float,
    mut_layer_prob: float,
) -> int:
    """Mutate the population with probabilities.

    :param population: list of individuals.
    :param toolbox: object where the mutation operators are defined.
    :param mut_neuron_prob: probability to mute a neuron (append/pop).
    :param mut_layer_prob: probability to mute a layer (append/pop).
    :returns: the number of individuals mutated.

    """
    mutated_individuals = 0

    for index, mutant in enumerate(population):
        mut_bias_genes = mut_weights_genes = neuron_diff = layer_diff = 0

        mut_bias_genes = toolbox.mutate_bias(mutant)
        mut_weights_genes = toolbox.mutate_weights(mutant)
        del mutant.fitness.values

        if random.random() < mut_neuron_prob:
            neuron_diff = toolbox.mutate_neuron(mutant)

        if random.random() < mut_layer_prob:
            layer_diff = toolbox.mutate_layer(mutant)

        if mut_bias_genes or mut_weights_genes:
            mutated_individuals += 1
            DGPLOGGER.debug(
                f"    For individual {index}:\n"
                f"        mutated {mut_bias_genes} bias genes\n"
                f"        mutated {mut_weights_genes} weights genes\n"
                f"        changed neuron number {neuron_diff}\n"
                f"        changed layer number {layer_diff}\n"
            )

    return mutated_individuals


def finished_generation_summary(current_generation: int, population: list):
    """Print a summary of the current generation.

    :param current_generation: index of the generation.
    :param population: list of individuals of the current generation.

    """
    fits = np.array([ind.fitness.values for ind in population])
    table = [
        ["Statistic", "F2 score", "Balanced acc score"],
        ["Max", fits[:, 0].max(), fits[:, 1].max()],
        ["Avg", fits[:, 0].mean(), fits[:, 1].mean()],
        ["Min", fits[:, 0].min(), fits[:, 1].min()],
        ["Std", fits[:, 0].std(), fits[:, 1].std()],
    ]
    DGPLOGGER.info(f"    Sumary of generation {current_generation}:")
    print_table(table, DGPLOGGER.info, floatfmt=(None, ".5f", ".5f"))


def finished_algorithm_summary(
    initial_population, final_population, best_individual_selector
):
    """Print a summary of the hole process.

    :param initial_population: list of initial individuals.
    :param final_population: list of final individuals.
    :param best_individual_selector: function to select the best individual.

    """
    initial_pop_table = []
    final_pop_table = []
    table_common_attributes = {
        "headers": ["Index", "F2 score", "Balanced acc score"],
        "print_fn": DGPLOGGER.debug,
        "floatfmt": ("i", ".5f", ".5f"),
    }

    for index, individual in enumerate(
        best_individual_selector(initial_population, len(initial_population))
    ):
        initial_pop_table.append(
            [
                str(index),
                str(individual.fitness.values[0]),
                str(individual.fitness.values[1]),
            ]
        )

    DGPLOGGER.debug("Initial population fitness values:")
    print_table(initial_pop_table, **table_common_attributes)

    for index, individual in enumerate(
        best_individual_selector(final_population, len(final_population))
    ):
        final_pop_table.append(
            [
                str(index),
                str(individual.fitness.values[0]),
                str(individual.fitness.values[1]),
            ]
        )

    DGPLOGGER.debug("Final population fitness values:")
    print_table(final_pop_table, **table_common_attributes)


def test_individual(
    individual: MLPIndividual,
    dataset: Proben1Partition,
    text: str = "Individual",
    fit_train: bool = False,
) -> Tuple[float, float]:
    """Test an individual with the validation and test data.

    :param individual: current individual to evaluate.
    :param model: base model, from whom obtain the initial individual shape.
    :param dataset: data to work with.
    :param fit_train: whether to fit the training data in each evaluation.

    """
    DGPLOGGER.info(f"{text} weights")
    DGPLOGGER.info(str(individual))
    DGPLOGGER.info(
        "Predicting the validation and test data with the best individual."
    )
    val_scores = individual_evaluator(
        individual,
        dataset.trn,
        dataset.val,
        fit_train=fit_train,
        average="micro" if dataset.nout > 2 else "binary",
    )
    tst_scores = individual_evaluator(
        individual,
        dataset.trn,
        dataset.tst,
        fit_train=fit_train,
        average="micro" if dataset.nout > 2 else "binary",
    )
    individual_table = [
        ["Validation", str(val_scores[0]), str(val_scores[1])],
        ["Test", str(tst_scores[0]), str(tst_scores[1])],
    ]
    print_table(
        individual_table,
        print_fn=DGPLOGGER.info,
        floatfmt=(None, ".5f", ".5f"),
        headers=["Partition", "F2 score", "Balanced acc score"],
    )
    return tst_scores


###############################################################################
#
# Genetic Algofithm
#
###############################################################################


# pylint: disable=no-member,too-many-arguments,too-many-locals
def genetic_algorithm(
    model: Sequential,
    dataset: Proben1Partition,
    init_population_size: int = 50,
    max_generations: int = 1000,
    cx_prob: float = 0.5,
    mut_bias_prob: float = 0.3,
    mut_weights_prob: float = 0.4,
    mut_neuron_prob: float = 0.3,
    mut_layer_prob: float = 0.3,
    fit_train: bool = False,
) -> Tuple[MLPIndividual, MLPIndividual]:
    """Perform optimization with a genetic algorithm.

    :param model: base model, from whom obtain the initial individual shape.
    :param dataset: data to work with.
    :param init_population_size: initial population size.
    :param max_generations: maximun number of generations to run.
    :param cx_prob: probability to mate two individuals.
    :param mut_bias_prob: probability to mutate the individual bias genes.
    :param mut_weights_prob: probability to mutate the individual weights
        genes.
    :param mut_neuron_prob: probability to add/remove a neuron from the model.
    :param mut_layer_prob: probability to add/remove a layer from the model.
    :param fit_train: whether to fit the training data in each evaluation.
    :returns: the toolbox with the registered functions.

    """
    np.random.seed(SEED)
    random.seed(SEED)

    # Configure dataset related variables
    toolbox = configure_toolbox(
        model,
        dataset,
        {"bias": mut_bias_prob, "weights": mut_weights_prob},
        fit_train,
    )

    # --------------------------------
    # Algorithm start
    # --------------------------------
    DGPLOGGER.title(msg="Start the algorithm")
    time_start = time.perf_counter()
    population = toolbox.population(n=init_population_size)

    DGPLOGGER.debug("Evaluate the initial population")
    evaluate_population(population, toolbox.evaluate)

    population.sort(key=lambda ind: ind.fitness.values, reverse=True)
    test_individual(
        population[0], dataset, "Best initial individual", fit_train
    )

    initial_population = population[:]
    DGPLOGGER.debug(f"    -- Evaluated {len(population)} individuals")

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in population]

    # Variable keeping track of the number of generations
    current_generation = 0

    try:
        # Begin the evolution
        while max(fits) < 1 and current_generation < max_generations:

            # A new generation
            current_generation = current_generation + 1
            DGPLOGGER.info(f"-- Generation {current_generation} --")

            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # --------------------------------
            # Operators
            # --------------------------------

            DGPLOGGER.debug("    -- Crossing some individuals")
            crossed_individuals = apply_crossover(
                offspring, cx_prob, toolbox.crossover
            )
            DGPLOGGER.info(
                f"    -- Crossed {crossed_individuals} individual pairs."
            )
            DGPLOGGER.debug("    -- Mutating some individuals.")
            mutated_individuals = apply_mutation(
                offspring, toolbox, mut_neuron_prob, mut_layer_prob
            )
            DGPLOGGER.info(
                f"    -- Mutated {mutated_individuals} individuals."
            )

            # --------------------------------
            # Evaluation
            # --------------------------------

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            DGPLOGGER.debug(
                f"    -- Evaluating {len(invalid_ind)} individuals."
            )
            evaluate_population(invalid_ind, toolbox.evaluate)
            DGPLOGGER.info(f"    -- Evaluated {len(invalid_ind)} individuals.")

            population[:] = offspring
            fits = [ind.fitness.values[0] for ind in population]
            finished_generation_summary(current_generation, population)
    except KeyboardInterrupt:
        DGPLOGGER.info("Stopping the algorithm...")

    elapsed_time = time.perf_counter() - time_start
    DGPLOGGER.debug(f"-- End of evolution in {elapsed_time} seconds.")
    population.sort(key=lambda ind: ind.fitness.values, reverse=True)
    finished_algorithm_summary(initial_population, population, tools.selBest)
    test_individual(population[0], dataset, "Best final individual", fit_train)

    return initial_population[0], population[0]
