#!/usr/bin/env python3
"""Script to train a model in pytorch"""

import argparse
import logging
import os

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

import utils
from model.net import Net, loss_fn, get_metrics
import model.data_loader as d_l
from evaluate import evaluate


def args_parser():
    """Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        default='../datasets/',
                        help="Directory containing the dataset")
    parser.add_argument('--model_dir',
                        default='experiments/base_model',
                        help="Directory containing params.json file")
    parser.add_argument('--restore_file',
                        default=None,
                        help="Optional, name of the file in --model_dir containing weights to\
                              reload before training")  # 'best' or 'last'
    return parser.parse_args()


def train(model, optimizer, criterion, dataloader, metrics, params, writer, epoch):
    """Train the model.
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        criterion: a function that computes the loss for the batch
        dataloader: (DataLoader) an object that fetches training data
        metrics: (dict) a dictionary of metrics
        params: (Params) hyperparameters
        writer : (SummaryWriter) Summary writer for tensorboard
        epoch: (int) Value of Epoch
    """
    # set model to training mode
    model.train()
    summ = []
    loss_avg = 0.0

    # Use tqdm for progress bar
    data_iterator = tqdm(dataloader, unit='batch')
    for i, (train_batch, labels) in enumerate(data_iterator):
        # move to GPU if available
        if params.cuda:
            train_batch = train_batch.to(params.device)
            labels = labels.to(params.device)

        # compute model output and loss
        output_batch = model(train_batch)
        loss = criterion(output_batch, labels, params)

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()

        # Evaluate summaries only once in a while
        if i % params.save_summary_steps == 0:
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()

            # compute all metrics on this batch
            summary_batch = {metric:metrics[metric](output_batch, labels)
                             for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)

            # Add to tensorboard
            writer.add_scalar('training_loss', summary_batch['loss'],
                              epoch * len(data_iterator) + i)

        # update the average loss
        loss_avg += loss.item()

        data_iterator.set_postfix(loss='{:05.3f}'.format(loss_avg/float(i+1)))

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: %s", metrics_string)


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, criterion, metrics,
                       params, model_dir, writer, restore_file=None):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) object that fetches training data
        val_dataloader: (DataLoader) a object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        criterion: a function to compute the loss for the batch
        metrics: (dict) a dictionary of metric functions
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        writer : (SummaryWriter) Summary writer for tensorboard
        restore_file: (string) optional name of file to restore, .pth.tar
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from %s", restore_path)
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch %d / %d", epoch + 1, params.num_epochs)

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, criterion, train_dataloader, metrics, params, writer, epoch)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, criterion, val_dataloader, metrics, params, writer, epoch)

        val_acc = val_metrics['accuracy'] if 'accuracy' in val_metrics else 0.0
        is_best = val_acc > best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)


def main():
    """Main function
    """
    # Load the parameters from json file
    args = args_parser()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Create summary writer for use with tensorboard
    writer = SummaryWriter(os.path.join(args.model_dir, 'runs', 'train'))

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)
        params.device = "cuda:0"
    else:
        params.device = "cpu"

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    dataloaders = d_l.get_dataloader(['train', 'val'], args.data_dir, params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    logging.info("- done.")

    # Define the model and optimizer
    model = Net(params)
    if params.cuda:
        model = model.to(params.device)
    writer.add_graph(model, next(iter(train_dl))[0])

    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    # fetch loss function and metrics
    criterion = loss_fn
    metrics = get_metrics()

    # Train the model
    logging.info("Starting training for %d epoch(s)", params.num_epochs)
    train_and_evaluate(model, train_dl, val_dl, optimizer, criterion, metrics, params,
                       args.model_dir, writer, args.restore_file)
    writer.close()


if __name__ == '__main__':
    main()
