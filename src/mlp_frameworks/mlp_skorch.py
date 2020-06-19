"""Multilayer perceptron classification with skorch."""
import torch
import torch.nn.functional as F

from skorch import NeuralNetClassifier
from torch import nn
from torch.optim import SGD

from src.mlp_frameworks import common


class MyModule(nn.Module):
    """Main module for creating an estimator."""

    def __init__(self):
        """Create the module object."""
        super(MyModule, self).__init__()

        self.dense0 = nn.Linear(
            in_features=common.FEATURE_NUMBER, out_features=100, bias=True
        )
        self.nonlin = F.relu
        self.solver = F.softmax
        self.dense1 = nn.Linear(in_features=100, out_features=100, bias=True)
        self.output = nn.Linear(
            in_features=100, out_features=common.CLASS_NUMBER, bias=True
        )

    def forward(self, X, **kwargs):
        """Run a forward operation."""
        # pylint: disable=unused-argument,arguments-differ
        X = self.nonlin(self.dense0(X))
        X = self.nonlin(self.dense1(X))
        X = self.solver(self.output(X), dim=-1)

        return X


def cross_validation() -> None:
    """Create the estimator, run cross validation and save the results."""
    print("Running cross validation for skorch (torch) MLP estimator.")
    torch.manual_seed(common.SEED)
    estimator = NeuralNetClassifier(
        module=MyModule,
        max_epochs=common.EPOCHS,
        lr=common.SGD_LR,
        batch_size=common.BATCH_SIZE,
        optimizer=SGD,
        train_split=None,
        verbose=0,
    )
    results = common.cross_validation(
        estimator=estimator,
        estimator_name="Skorch (Torch)",
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
