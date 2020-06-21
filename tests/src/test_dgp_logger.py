"""Test the :mod:`src.dgp_logger` module."""
import unittest

from unittest import mock

from src.dgp_logger import DGPLogger


class TestDGPLogger(unittest.TestCase):
    """Test DGPLogger class."""

    # pylint: disable=protected-access
    def test_log(self):
        """Test log function override."""
        custom_logger = DGPLogger("test")
        self.assertListEqual(custom_logger.handlers, [])

        with mock.patch("src.dgp_logger.super") as mock_super:
            custom_logger._log(
                level="level", msg="this is a multiline\nstring", args=[]
            )

        self.assertEqual(mock_super.return_value._log.call_count, 2)

    @staticmethod
    def test_sep():
        """Test the sep printer method."""
        custom_logger = DGPLogger("test")
        custom_logger.log = mock.Mock()
        custom_logger.sep("level")

        custom_logger.log.assert_called_with("level", "*" * 79)

    def test_title(self):
        """Test the thitle printer method."""
        custom_logger = DGPLogger("test")
        custom_logger.log = mock.Mock()
        custom_logger.title("level", "message")

        custom_logger.log.assert_any_call("level", "message")
        self.assertEqual(custom_logger.log.call_count, 3)

    @mock.patch("src.dgp_logger.datetime")
    @mock.patch("src.dgp_logger.logging")
    def test_configure_dgp_logger(self, mock_logging, mock_datetime):
        """Test the configuration method."""
        custom_logger = DGPLogger("test")
        custom_logger.addHandler = mock.Mock()
        mock_logging.getLevelName.return_value = 28
        mock_datetime.now.return_value.strftime.return_value = "fake_date"
        stem_sufix = "stem_sufix"

        mock_dir_path = mock.MagicMock()
        mock_dir_path.__truediv__ = mock.Mock(return_value="composed_path")
        custom_logger.configure_dgp_logger("INFO", mock_dir_path, stem_sufix)

        mock_logging.FileHandler.assert_called_with("composed_path")
        mock_logging.StreamHandler.assert_called_once()
        self.assertEqual(custom_logger.addHandler.call_count, 2)


if __name__ == "__main__":
    unittest.main()