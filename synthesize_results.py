#!/usr/bin/env python3
"""Aggregates results from the json files in all experiments folder in a parent folder"""

import argparse
import json
import os

from tabulate import tabulate


def args_parser():
    """Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent_dir',
                        default='experiments',
                        help='Directory containing results of experiments')
    return parser.parse_args()


def aggregate_metrics(parent_dir, metrics):
    """Aggregate the metrics of all experiments in folder parent_dir.
    Args:
        parent_dir: (string) path to directory containing experiments results
        metrics: (dict) a dictionary of metrics
    """
    # Get the metrics for the folder if it has results from an experiment
    metrics_file = os.path.join(parent_dir, 'metrics_val_last_weights.json')
    if os.path.isfile(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics[parent_dir] = json.load(f)

    # Check every subdirectory of parent_dir
    for subdir in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, subdir)):
            continue
        else:
            aggregate_metrics(os.path.join(parent_dir, subdir), metrics)


def metrics_to_table(metrics):
    """Convert a dictionary containing metrics into table form
    Args:
        metrics: (dict) Metrics to find accuracy of the algorithm
    Returns:
        res: (table) A formated plain text table
    """
    # Get the headers from the first subdir. Assumes everything has the same metrics
    headers = metrics[list(metrics.keys())[0]].keys()
    table = [[subdir] + [values[h] for h in headers] for subdir, values in metrics.items()]
    res = tabulate(table, headers, tablefmt='pipe')

    return res

def main():
    """Main function
    """
    args = args_parser()

    # Aggregate metrics from args.parent_dir directory
    metrics = dict()
    aggregate_metrics(args.parent_dir, metrics)
    table = metrics_to_table(metrics)

    # Display the table to terminal
    print(table)

    # Save results in parent_dir/results.md
    save_file = os.path.join(args.parent_dir, "results.md")
    with open(save_file, 'w') as f:
        f.write(table)

if __name__ == "__main__":
    main()
