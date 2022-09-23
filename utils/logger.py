#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging


def setlogger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # asctime是日志时间，message是日志信息
    logFormatter = logging.Formatter("%(asctime)s %(message)s", "%m-%d %H:%M:%S")

    fileHandler = logging.FileHandler(path)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
