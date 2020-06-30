"""General logger."""
import logging

from datetime import datetime
from pathlib import Path

from settings import LOGS_DIR_PATH


class DGPLogger(logging.Logger):
    """Logger that allow multiline strings and some utils."""

    # pylint: disable=too-many-arguments
    def _log(
        self, level, msg, args, exc_info=None, extra=None, stack_info=False
    ):
        """Iterate over every message lines to log each one."""
        for line in msg.splitlines():
            super()._log(
                level, line, args, exc_info=None, extra=None, stack_info=False
            )

    def sep(self, level=logging.DEBUG):
        """Log a long separator string."""
        self.log(level, "*" * 79)

    def title(self, level=logging.INFO, msg: str = ""):
        """Log a title message."""
        self.sep(level)
        self.log(level, msg)
        self.sep(level)

    def configure_dgp_logger(
        self,
        log_stream_level: str = "INFO",
        log_file_stem_sufix: str = None,
        log_file_dir: Path = LOGS_DIR_PATH,
    ) -> None:
        """Configure the DGPLOGGER.

        :parma log_stream_level: logging level of the :class:`StreamHandler`.
        :param log_file_stem_sufix: extra text after the date for the file
            name.
        :param log_file_dir: path to the directory in which the
            :class:`FileHandler` will write the log output.
        :returns: the configured logger.

        """

        def _validate_level(level: str, default=logging.INFO) -> int:
            """Validate a string logging level.

            :param level: the level string.
            :param default: the default value returned if the ``level`` is not
                know.
            :returns: the logging level as integer.

            """
            validated_level = logging.getLevelName(level)
            return (
                validated_level
                if isinstance(validated_level, int)
                else default
            )

        datetime_format = "%y%b%d_%H%M%S"
        current_date = datetime.now().strftime(datetime_format)

        # Configure the stream handler
        stream_handler = logging.StreamHandler()
        stream_level_var = log_stream_level
        stream_handler.setLevel(
            _validate_level(stream_level_var, logging.INFO)
        )
        stream_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s|%(message)s", datefmt=datetime_format,
            )
        )
        self.addHandler(stream_handler)

        # Configure the file handler
        if log_file_stem_sufix:
            file_handler = logging.FileHandler(
                log_file_dir / f"{current_date}_{log_file_stem_sufix}.log"
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s|%(message)s", datefmt=datetime_format,
                )
            )
            self.addHandler(file_handler)


DGPLOGGER = DGPLogger("DeepGProp", logging.DEBUG)
