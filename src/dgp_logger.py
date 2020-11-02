"""General logger."""
from datetime import datetime
from logging import (
    DEBUG,
    INFO,
    FileHandler,
    Formatter,
    Logger,
    StreamHandler,
    getLevelName,
)
from pathlib import Path

from settings import LOGS_DIR_PATH


class DGPLogger(Logger):
    """Logger that allow multiline strings and some utils."""

    @staticmethod
    def _validate_level(level: str, default=INFO) -> int:
        """Validate a string logging level.

        :param level: the level string.
        :param default: the default value returned if the ``level`` is not
            know.
        :returns: the logging level as integer.

        """
        validated_level = getLevelName(str(level).upper())
        return validated_level if isinstance(validated_level, int) else default

    # pylint: disable=too-many-arguments
    def _log(
        self, level, msg, args, exc_info=None, extra=None, stack_info=False
    ):
        """Iterate over every message lines to log each one."""
        for line in msg.splitlines():
            super()._log(
                level, line, args, exc_info=None, extra=None, stack_info=False
            )

    def sep(self, level=DEBUG):
        """Log a long separator string."""
        self.log(DGPLogger._validate_level(level, DEBUG), "*" * 79)

    def title(self, level=INFO, msg: str = ""):
        """Log a title message."""
        self.sep(level)
        self.log(DGPLogger._validate_level(level, INFO), msg)
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
        datetime_format = "%y%b%d_%H%M%S"

        # Configure the stream handler
        stream_handler = StreamHandler()
        stream_level_var = log_stream_level
        stream_handler.setLevel(
            DGPLogger._validate_level(stream_level_var, INFO)
        )
        stream_handler.setFormatter(
            Formatter(fmt="%(asctime)s|%(message)s", datefmt=datetime_format)
        )
        self.addHandler(stream_handler)

        # Configure the file handler
        if log_file_stem_sufix:
            current_date = datetime.now().strftime("%y%m%d_%H%M%S")
            file_handler = FileHandler(
                log_file_dir / f"{current_date}_{log_file_stem_sufix}.log"
            )
            file_handler.setLevel(DEBUG)
            file_handler.setFormatter(
                Formatter(
                    fmt="%(asctime)s|%(message)s", datefmt=datetime_format
                )
            )
            self.addHandler(file_handler)


DGPLOGGER = DGPLogger("DeepGProp", DEBUG)
