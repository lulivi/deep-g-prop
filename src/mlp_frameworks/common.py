"""Common functions and variables used by the MLP classifiers."""
import csv
import time

from pathlib import Path
from typing import List

import numpy as np
import sklearn.datasets

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold
from tabulate import tabulate

from src.common import SEED
from src.dgp_logger import DGPLOGGER

# Set global path variables
CURRENT_FILE_PATH = Path(__file__).resolve()
CURRENT_FILE_DIR = CURRENT_FILE_PATH.parent
DGPLOGGER.configure_dgp_logger(
    "DEBUG", CURRENT_FILE_PATH.stem, CURRENT_FILE_DIR
)
FRAMEWORK_RESULTS_CSV = CURRENT_FILE_DIR / "framework_results.csv"

# Load the dataset
dataset = sklearn.datasets.load_digits()
X = dataset["data"].astype(np.float32)
y = dataset["target"].astype(np.int64)

# Define some common estimator attributes
FEATURE_NUMBER = X.shape[1]
DGPLOGGER.info("Dataset shape: '%s'", X.shape)
DGPLOGGER.info("Target data shape: '%s'", y.shape)
CLASS_NUMBER = len(np.unique(y))
DGPLOGGER.info("Number of classes: '%d'", CLASS_NUMBER)
DGPLOGGER.info(
    "Class distribution: <%s>",
    np.asarray(np.unique(y, return_counts=True)).T.tolist(),
)
DGPLOGGER.info("Seed: '%d'", SEED)
np.random.seed(SEED)
SGD_LR = 0.001
EPOCHS = 200
BATCH_SIZE = 5
ACTIVATION = "relu"
SOLVER = "sgd"
N_SPLITS = 5
METRICS = {
    "accuracy": (accuracy_score, {}),
    "f1_score": (f1_score, {"average": "macro"}),
    "precision": (precision_score, {"average": "macro"}),
    "recall": (recall_score, {"average": "macro"}),
}
# K-Fold strategy
K_FOLD = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

result_header = [
    "estimator",
    "fit_time",
    "train_accuracy_mean",
    "train_accuracy_std",
    "test_accuracy_mean",
    "test_accuracy_std",
    "train_f1_score_mean",
    "train_f1_score_std",
    "test_f1_score_mean",
    "test_f1_score_std",
    "train_precision_mean",
    "train_precision_std",
    "test_precision_mean",
    "test_precision_std",
    "train_recall_mean",
    "train_recall_std",
    "test_recall_mean",
    "test_recall_std",
]


def cross_validation(estimator, estimator_name, X, y, metrics,) -> List[str]:
    """Run cross validation to the chosen estimator.

    :param estimator: object to which apply fit and predict, to obtain results.
    :param estimator_name: name of the chosen estimator.
    :param X: dataset values without the labels.
    :param y: dataset labels.
    :param metrics: metrics which will be applied to the estimator.
    :returns: a dictionary with the obtained results.

    """
    # pylint: disable=too-many-locals,invalid-name,redefined-outer-name
    DGPLOGGER.debug("Training %s...", estimator_name)
    DGPLOGGER.debug("Estimator: %s", str(estimator))
    result = {"estimator": estimator_name}
    result["fit_times"] = np.zeros((N_SPLITS,))
    for metric_name in metrics.keys():
        result[f"train_{metric_name}"] = np.zeros((N_SPLITS,))
        result[f"test_{metric_name}"] = np.zeros((N_SPLITS,))

    for fold, (train_idx, test_idx) in enumerate(K_FOLD.split(X=X, y=y)):
        DGPLOGGER.info("Fold %d/%d", fold + 1, N_SPLITS)

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        DGPLOGGER.debug("Estimator fit...")
        start_time = time.perf_counter()
        estimator.fit(X_train, y_train)
        fit_time = time.perf_counter() - start_time
        DGPLOGGER.debug("Estimator fit... Done in %.5f seconds", fit_time)

        result["fit_times"][fold] = fit_time
        DGPLOGGER.debug("Predicting train labels...")
        y_pred_train = estimator.predict(X_train)
        DGPLOGGER.debug("Predicting test labels...")
        y_pred_test = estimator.predict(X_test)

        for metric_name, (func, kwargs) in metrics.items():
            DGPLOGGER.debug("Obtaining train metric: %s...", metric_name)
            result[f"train_{metric_name}"][fold] = func(
                y_train, y_pred_train, **kwargs
            )
            DGPLOGGER.debug(
                "Obtaining train metric: %s... Done: %s",
                metric_name,
                result[f"train_{metric_name}"][fold],
            )
            DGPLOGGER.debug("Obtaining test metric: %s...", metric_name)
            result[f"test_{metric_name}"][fold] = func(
                y_test, y_pred_test, **kwargs
            )
            DGPLOGGER.debug(
                "Obtaining test metric: %s... Done: %s",
                metric_name,
                result[f"test_{metric_name}"][fold],
            )

    DGPLOGGER.debug("Training %s... Done", estimator_name)

    output_list = [
        result.pop("estimator"),
        np.format_float_positional(
            result.pop("fit_times").mean(), precision=2, unique=False
        ),
    ]
    for metric_name in sorted(metrics.keys()):
        output_list.append(
            np.format_float_positional(
                result[f"train_{metric_name}"].mean(),
                precision=6,
                unique=False,
            )
        )
        output_list.append(
            np.format_float_positional(
                result[f"train_{metric_name}"].std(), precision=6, unique=False
            )
        )
        output_list.append(
            np.format_float_positional(
                result[f"test_{metric_name}"].mean(), precision=6, unique=False
            )
        )
        output_list.append(
            np.format_float_positional(
                result[f"test_{metric_name}"].std(), precision=6, unique=False
            )
        )

    return output_list


def show_and_save_result(result: List[str]) -> None:
    """Show and save the results in a table format.

    :param result: list with the cross validation results of one estimator.

    """
    data = [result]

    print(
        tabulate(tabular_data=data, headers=result_header, tablefmt="pretty",)
    )
    save_result(result)


def save_result(result: List[str]) -> None:
    """Save the results in a csv.

    :param result: list with the cross validation results of one estimator.

    """
    DGPLOGGER.info("Saving results to %s...", str(FRAMEWORK_RESULTS_CSV))
    FRAMEWORK_RESULTS_CSV.touch(exist_ok=True)

    with open(FRAMEWORK_RESULTS_CSV, "r") as file_descriptor:
        csv_content = list(csv.reader(file_descriptor))

    contents = []
    if csv_content:
        headers = csv_content[0]
        if len(csv_content) > 1:
            contents = csv_content[1:]

        if headers != result_header:
            raise TypeError(
                "Headers from the csv and the result does not match."
            )

    contents.append(result)

    with open(FRAMEWORK_RESULTS_CSV, "w") as file_descriptor:
        writer = csv.writer(file_descriptor)
        writer.writerow(result_header)
        writer.writerows(contents)

    DGPLOGGER.info("Saving results to %s... Done", str(FRAMEWORK_RESULTS_CSV))
