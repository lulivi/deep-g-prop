import csv
import random
import time

from pathlib import Path
from typing import List, Union
from warnings import simplefilter

import numpy as np
import sklearn.datasets

from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.neural_network import MLPClassifier

# Set global path variables
HP_OPTIMIZATION_DIR = Path(__file__).parent
LOGS_DIR = HP_OPTIMIZATION_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
HP_OPTIMIZATION_CSV = HP_OPTIMIZATION_DIR / "hp_optimization.csv"

random.seed(12345)
np.random.seed(12345)

table_header = ["optimizator", "best_score", "time"]

data = sklearn.datasets.load_digits()
X = data["data"]
y = data["target"]

paramgrid = {
    "solver": ["lbfgs", "sgd", "adam"],
    "hidden_layer_sizes": [(16, 16), (100, 100), (16,)],
    "activation": ["logistic", "tanh", "relu"],
    "batch_size": [3, 5],
    "learning_rate": ["constant", "invscaling", "adaptive"],
    "learning_rate_init": [0.001, 0.01],
    "random_state": [12345],
}

cross_validation = StratifiedKFold(n_splits=4)

model = MLPClassifier()

simplefilter(action="ignore", category=ConvergenceWarning)


def cross_validate(
    name: str,
    cv: Union[EvolutionaryAlgorithmSearchCV, GridSearchCV, RandomizedSearchCV],
) -> List[str]:
    """Run the fit function to obtain the optimal parameters.

    :param name: name of the optimizer.
    :param cv: optimizer cross validator.
    :returns: the cv results.

    """
    time_start = time.perf_counter()
    cv.fit(X, y)
    time_stop = time.perf_counter()

    return [
        name,
        f"{cv.best_score_:.5f}",
        f"{time_stop - time_start:.5f}",
    ]


def save_result(result: List[str]) -> None:
    """Save the results to a csv.

    :param result: list containing algorithm name, best score and fit time.

    """
    HP_OPTIMIZATION_CSV.touch(exist_ok=True)

    with open(HP_OPTIMIZATION_CSV, "r") as f:
        csv_content = list(csv.reader(f))

    result_headers = table_header
    contents = []
    if csv_content:
        headers = csv_content[0]
        if len(csv_content) > 1:
            contents = csv_content[1:]

        if headers != result_headers:
            raise TypeError(
                "Headers from the csv and the result does not match."
            )

    contents.append(result)

    with open(HP_OPTIMIZATION_CSV, "w") as f:
        writer = csv.writer(f)
        writer.writerow(result_headers)
        writer.writerows(contents)
