# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict


import logging


def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
