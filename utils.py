#!/usr/bin/env python3
"""Utility functions and classes
"""

import json
import logging
import os
import shutil
from itertools import islice

import torch

class Params():
    """Class to load hyperparameters from a json file.
    """
    def __init__(self, json_path):
        with open(json_path, encoding='utf-8') as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        """Save parameters to json file at json_path
        """
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file at json_path
        """
        with open(json_path, encoding='utf-8') as f:
            params = json.load(f)
            self.__dict__.update(params)


def set_logger(log_path):
    """Set the logger to log info in terminal and file at log_path.
    Args:
        log_path: (string) location of log file
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def save_dict_to_json(data, json_path):
    """Saves a dict of floats to json file
    Args:
        data: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w', encoding='utf-8') as f:
        data = {k: float(v) for k, v in data.items()}
        json.dump(data, f, indent=4)

def save_checkpoint(state, is_best, checkpoint):
    """Saves model at checkpoint
    Args:
        state: (dict) contains model's state_dict, epoch, optimizer state_dict etc.
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    safe_makedir(checkpoint)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))

def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model state_dict from checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise"File doesn't exist {}".format(checkpoint)
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

def safe_makedir(path):
    """Make directory given the path if it doesn't already exist
    Args:
        path: path of the directory to be made
    """
    if not os.path.exists(path):
        print("Directory doesn't exist! Making directory {0}.".format(path))
        os.makedirs(path)
    else:
        print("Directory {0} Exists!".format(path))

def window(seq, width=2):
    """Returns a sliding window of width n over data from the iterable
    Args:
        seq (iterable): sequence to iterate over
        width (int): width of window
    Yields:
        (s0, s1 ... sn-1), (s1, s2 ... sn),
    """
    itr = iter(seq)
    result = tuple(islice(itr, width))
    if len(result) == width:
        yield result
    for elem in itr:
        result = result[1:] + (elem,)
        yield result
