"""Test :mod:`src.mlp_frameworks` module frameworks."""
import unittest

from unittest import TestCase, mock

from src.mlp_frameworks import mlp_keras, mlp_sklearn, mlp_skorch


class TestSklearn(TestCase):
    """Test the Sklearn neural network classifier module."""

    @staticmethod
    @mock.patch("src.mlp_frameworks.mlp_sklearn.common")
    def test_cross_validation(mock_common):
        """Test the search."""
        mock_output_cv = mock.Mock()
        mock_common.cross_validation.return_value = mock_output_cv
        with mock.patch(
            "src.mlp_frameworks.mlp_sklearn.MLPClassifier"
        ) as mock_classifier:
            mlp_sklearn.cross_validation()

        mock_classifier.assert_called_once()
        mock_common.cross_validation.assert_called_once()
        mock_common.show_and_save_result.assert_called_with(mock_output_cv)

    @staticmethod
    def test_main():
        """Test the main function."""
        with mock.patch(
            "src.mlp_frameworks.mlp_sklearn.cross_validation"
        ) as mock_cross_validation:
            mlp_sklearn.main()

        mock_cross_validation.assert_called_once()


class TestKeras(TestCase):
    """Test the Keras neural network classifier module."""

    @staticmethod
    def test_baseline_model():
        """Test the obtention of a baseline model."""
        with mock.patch.multiple(
            mlp_keras, Dense=mock.DEFAULT, Sequential=mock.DEFAULT,
        ) as values:
            mock_sequential = values["Sequential"]
            mock_model = mock.Mock()
            mock_sequential.return_value = mock_model

            mlp_keras.baseline_model()

        mock_model.summary.assert_called_once()
        mock_model.compile.assert_called_once()

    @staticmethod
    @mock.patch("src.mlp_frameworks.mlp_keras.common")
    def test_cross_validation(mock_common):
        """Test the search."""
        mock_output_cv = mock.Mock()
        mock_common.cross_validation.return_value = mock_output_cv
        with mock.patch(
            "src.mlp_frameworks.mlp_keras.KerasClassifier"
        ) as mock_classifier:
            mlp_keras.cross_validation()

        mock_classifier.assert_called_once()
        mock_common.cross_validation.assert_called_once()
        mock_common.show_and_save_result.assert_called_with(mock_output_cv)

    @staticmethod
    def test_main():
        """Test the main function."""
        with mock.patch(
            "src.mlp_frameworks.mlp_keras.cross_validation"
        ) as mock_cross_validation:
            mlp_keras.main()

        mock_cross_validation.assert_called_once()


class TestSkorch(TestCase):
    """Test the Skorch neural network classifier module."""

    @mock.patch("src.mlp_frameworks.mlp_skorch.F")
    @mock.patch("src.mlp_frameworks.mlp_skorch.nn")
    def test_skorch_mymodule(self, mock_nn, mock_f):
        """Test the skorch module constructor."""
        module = mlp_skorch.MyModule()

        self.assertEqual(module.dense0, mock_nn.Linear())
        self.assertEqual(module.dense1, mock_nn.Linear())
        self.assertEqual(module.output, mock_nn.Linear())
        self.assertEqual(module.nonlin, mock_f.relu)
        self.assertEqual(module.solver, mock_f.softmax)

        x_data = module.forward(mock.Mock())
        self.assertEqual(x_data, module.solver(mock.Mock()))

    @staticmethod
    @mock.patch("src.mlp_frameworks.mlp_skorch.common")
    def test_cross_validation(mock_common):
        """Test the search."""
        mock_output_cv = mock.Mock()
        mock_common.cross_validation.return_value = mock_output_cv
        with mock.patch(
            "src.mlp_frameworks.mlp_skorch.NeuralNetClassifier"
        ) as mock_classifier:
            mlp_skorch.cross_validation()

        mock_classifier.assert_called_once()
        mock_common.cross_validation.assert_called_once()
        mock_common.show_and_save_result.assert_called_with(mock_output_cv)

    @staticmethod
    def test_main():
        """Test the main function."""
        with mock.patch(
            "src.mlp_frameworks.mlp_skorch.cross_validation"
        ) as mock_cross_validation:
            mlp_skorch.main()

        mock_cross_validation.assert_called_once()


if __name__ == "__main__":
    unittest.main()
