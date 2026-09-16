"""Bao AI - Structured Telemetry and Logging.

Replaces silent `except: pass` patterns across the app with structured,
JSON-formatted logs that carry live CPU/RAM telemetry on every line.
"""

import json
import logging
import time

import psutil

_ROOT_LOGGER_NAME = "bao"


# Telemetry is sampled at most this often, not once per log record.
#
# Measured before changing it: reading psutil on every record cost 0.121 ms,
# and a single turn emits roughly six records — about 0.72 ms, which is 27%
# of the 2.62 ms "fast offline path" this project reports. The instrument
# was a material part of what it was measuring.
#
# The reading was also less meaningful than it looked: psutil.cpu_percent()
# with no interval returns the load since the previous call, so sampling it
# every few microseconds inside one turn reports near-noise. A cached value
# refreshed on a wall-clock interval is both cheaper AND more accurate.
_TELEMETRY_TTL_SECONDS = 2.0
_telemetry_cache: dict = {"at": 0.0, "value": {"cpu_percent": 0.0, "ram_used_gb": 0.0, "ram_percent": 0.0}}


def current_telemetry() -> dict:
    """Cached CPU/RAM snapshot, refreshed at most every few seconds."""
    now = time.monotonic()
    if now - _telemetry_cache["at"] >= _TELEMETRY_TTL_SECONDS:
        memory = psutil.virtual_memory()  # one call, not two
        _telemetry_cache["value"] = {
            "cpu_percent": psutil.cpu_percent(),
            "ram_used_gb": round(memory.used / (1024 ** 3), 2),
            "ram_percent": memory.percent,
        }
        _telemetry_cache["at"] = now
    return _telemetry_cache["value"]


class TelemetryFormatter(logging.Formatter):
    """JSON formatter that attaches a recent hardware telemetry snapshot."""

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
            "telemetry": current_telemetry(),
        }
        return json.dumps(log_record)


def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    """Initializes the root 'bao' logger. Safe to call more than once."""
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(log_level)
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(TelemetryFormatter())
        logger.addHandler(console_handler)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Returns a child logger under the 'bao' namespace.

    Every module should call `get_logger(__name__)` instead of
    `logging.getLogger(__name__)` directly, so all log output shares the
    structured telemetry formatter set up once, here.
    """
    setup_logging()
    if not name:
        return logging.getLogger(_ROOT_LOGGER_NAME)
    if name == _ROOT_LOGGER_NAME or name.startswith(f"{_ROOT_LOGGER_NAME}."):
        # Caller already passed a dotted path inside the bao.* namespace
        # (the common case: get_logger(__name__) from a module already
        # named bao.knowledge.loader) — don't prefix it again.
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


# Initialize immediately on import, same as before.
logger = setup_logging()
