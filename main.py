import argparse
import logging
import sys
import tkinter as tk

import pywt._extensions._cwt
import scipy.spatial.ckdtree
from scipy.linalg import _fblas

from src.path_connector import PathConnector

LOGGER = logging.getLogger(__name__)


def argparser():
    parser = argparse.ArgumentParser(description='Some arguement for path connector')
    parser.add_argument('-m', '--max', dest='max', type=int, default=300, help='maximum frame for displaying path')
    parser.add_argument('-t', '--tolerance', dest='tolerance', type=int, default=38, help='maximum tolerance of distance')
    return parser


def log_handler(*loggers):
    formatter = logging.Formatter(
        '%(asctime)s %(filename)12s:L%(lineno)3s [%(levelname)8s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # stream handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    for logger in loggers:
        logger.addHandler(sh)
        logger.setLevel(logging.DEBUG)


if __name__ == '__main__':
    parser = argparser()
    args = parser.parse_args()
    pc = PathConnector(args.max, args.tolerance)
    pc.start()
