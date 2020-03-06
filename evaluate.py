#!/usr/bin/env python3
"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import utils
from model.net import Net, loss_fn, get_metrics
import model.data_loader as d_l


def args_parser():
    """Parse commadn line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        default='../datasets/',
                        help="Directory containing the dataset")
    parser.add_argument('--model_dir',
                        default='experiments/base_model',
                        help="Directory containing params.json")
    parser.add_argument('--restore_file',
                        default='last',
                        help="name of the file in --model_dir containing weights to load")
    return parser.parse_args()


def evaluate(model, criterion, dataloader, metrics, params, writer, epoch):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network
        criterion: a function that computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
        writer : (SummaryWriter) Summary writer for tensorboard
        epoch: (int) Value of Epoch
    """
    # put model in evaluation mode
    model.eval()
    summ = []

    with torch.no_grad():
        for i, (inp_data, labels) in enumerate(dataloader):
            # move data to GPU if possible
            if params.cuda:
                inp_data = inp_data.to(params.device)
                labels = labels.to(params.device)

            # compute model output
            output = model(inp_data)
            loss = criterion(output, labels, params)

            # detach and move to cpu, convert to numpy arrays
            output = output.cpu().numpy()
            labels = labels.cpu().numpy()

            # compute all metrics on this batch
            summary_batch = {metric: metrics[metric](output, labels) for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)

            # Add to tensorboard
            writer.add_scalar('testing_loss', summary_batch['loss'],
                              epoch * len(dataloader) + i)

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : %s", metrics_string)
    return metrics_mean


def main():
    """Main function
    """
    # Load the parameters
    args = args_parser()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Create summary writer for use with tensorboard
    writer = SummaryWriter(os.path.join(args.model_dir, 'runs', 'eval'))

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)
        params.device = "cuda:0"
    else:
        params.device = "cpu"

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    logging.info("Loading the dataset...")

    # fetch dataloaders
    dataloaders = d_l.get_dataloader(['test'], args.data_dir, params)
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model
    model = Net(params)
    if params.cuda:
        model = model.to(params.device)
    writer.add_graph(model, next(iter(test_dl))[0])

    criterion = loss_fn
    metrics = get_metrics()

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, criterion, test_dl, metrics, params, writer, 0)
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)

    writer.close()


if __name__ == '__main__':
    main()
