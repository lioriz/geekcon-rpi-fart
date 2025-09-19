#!/usr/bin/env python3
"""
Logger utilities for consistent logging across the project
"""

import logging

LOGGER_DEFAULT_FORMATTER = logging.Formatter('%(asctime)s<%(threadName)s>%(levelname)s::%(message)s')

def get_base_logger(name):
    """Get a base logger with handlers removed"""
    logger = logging.getLogger(name)
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    return logger

def get_logger(name, level=logging.INFO):
    """Get a configured logger with consistent formatting"""
    logger = get_base_logger(name)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(LOGGER_DEFAULT_FORMATTER)
    logger.setLevel(level)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger
