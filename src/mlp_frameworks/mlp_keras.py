"""Multilayer perceptron classification with Keras."""
from keras import backend as K
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier

from src.mlp_frameworks import common


def baseline_model():
    """Create the model."""
    model = Sequential(
        [
            Dense(
                input_dim=common.FEATURE_NUMBER,
                units=100,
                activation=common.ACTIVATION,
                use_bias=True,
            ),
            Dense(units=100, activation=common.ACTIVATION, use_bias=True,),
            Dense(
                units=common.CLASS_NUMBER, activation="softmax", use_bias=True,
            ),
        ]
    )
    model.summary()
    model.compile(
        optimizer=common.SOLVER,
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

    return model


def cross_validation() -> None:
    """Create the estimator, run cross validation and save the results."""
    print(f"Running cross validation for Keras estimator ({K.backend()}).")
    estimator = KerasClassifier(
        build_fn=baseline_model,
        epochs=common.EPOCHS,
        batch_size=common.BATCH_SIZE,
        verbose=3,
    )
    results = common.cross_validation(
        estimator=estimator,
        estimator_name=f"Keras ({K.backend()})",
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
