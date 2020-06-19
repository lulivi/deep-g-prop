"""Multilayer perceptron classification with scikit-learn."""
from sklearn.neural_network import MLPClassifier

from src.mlp_frameworks import common


def cross_validation() -> None:
    """Create the estimator, run cross validation and save the results."""
    print("Running cross validation for scikit-learn MLP estimator.")
    # For output layer solver uses softmax
    estimator = MLPClassifier(
        hidden_layer_sizes=(100, 100),
        activation=common.ACTIVATION,
        solver=common.SOLVER,
        learning_rate="constant",
        learning_rate_init=common.SGD_LR,
        max_iter=common.EPOCHS,
        batch_size=common.BATCH_SIZE,
        random_state=common.SEED,
        verbose=0,
    )
    results = common.cross_validation(
        estimator=estimator,
        estimator_name="Scikit-learn",
        X=common.X,
        y=common.y,
        metrics=common.METRICS,
    )

    common.show_and_save_result(results)


def main():
    """Call the cross validation."""
    cross_validation()


if __name__ == "__main__":
    main()
