#!/usr/bin/env python 3
"""Peform hyperparemeters search"""

import argparse
import os
from subprocess import check_call
import sys
from itertools import product

import utils


PYTHON = sys.executable

def args_parser():
    """Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent_dir',
                        default='experiments/param_search',
                        help='Directory containing params.json')
    parser.add_argument('--data_dir',
                        default='../datasets',
                        help="Directory containing the dataset")
    return parser.parse_args()


def launch_training_job(model_dir, data_dir, job_name, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name
    Args:
        model_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        job_name: (string) name of the experiment to search hyperparameters
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(model_dir, job_name)
    utils.safe_makedir(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} train.py --model_dir={model_dir} --data_dir {data_dir}".format(
        python=PYTHON, model_dir=model_dir, data_dir=data_dir)
    print(cmd)
    check_call(cmd, shell=True)


def main():
    """Main function
    """
    # Load the "reference" parameters from parent_dir json file
    args = args_parser()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Perform hypersearch over one parameters
    configurations = {
        "learning_rate": [0.01, 0.001, 0.0001],
        "margin": [0.2, 0.5, 0.8],
        "normalize": [True, False],
    }
    conf_values = list(configurations.values())
    conf_names = list(configurations.keys())

    for vals in product(*conf_values):
        # Modify the relevant parameter in params
        conf = dict(zip(conf_names, vals))
        params.__dict__.update(conf)

        # Launch job (name has to be unique)
        name = ""
        for key, val in conf.items():
            name += '_' + str(key) + '_' + str(val)
        job_name = "params{}".format(name)
        launch_training_job(args.parent_dir, args.data_dir, job_name, params)


if __name__ == "__main__":
    main()
