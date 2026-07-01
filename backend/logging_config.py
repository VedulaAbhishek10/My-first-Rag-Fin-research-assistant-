"""
Logging configuration for the Financial Research Assistant.

Why structured logging?
  In production, logs are the primary way to understand what is happening
  inside a running application. A consistent format (timestamp | level | module
  | message) makes logs easy to read in a terminal and easy to parse with log
  aggregation tools like Grafana Loki or AWS CloudWatch.

Usage:
    from backend.logging_config import setup_logging, get_logger

    setup_logging("INFO")           # call once at app startup
    logger = get_logger(__name__)   # call in each module
    logger.info("Something happened")
"""

import logging
import sys

# A fixed format so every log line looks the same across all modules.
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure the root logger for the entire application.

    Args:
        log_level: One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
                   Loaded from Settings.log_level at startup.
    """
    logging.basicConfig(
        level=log_level.upper(),
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # override any handlers set by imported libraries
    )


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    Args:
        name: Typically __name__ of the calling module. This makes the
              module path appear in log lines, e.g. backend.api.routes.
    """
    return logging.getLogger(name)
