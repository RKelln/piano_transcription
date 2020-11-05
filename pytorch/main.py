import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import librosa
import logging
import datetime

import matplotlib.pyplot as plt
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import config
from utilities import (create_folder, get_filename, create_logging, 
    StatisticsContainer, RegressionPostProcessor, highest_iteration) 
from data_generator import MaestroDataset, Augmentor, Sampler, TestSampler, collate_fn
from models import Regress_onset_offset_frame_velocity_CRNN, Regress_pedal_CRNN
from pytorch_utils import move_data_to_device
from losses import get_loss_func
from evaluate import SegmentEvaluator




def train(args):
    """Train a piano transcription system.

    Args:
      workspace: str, directory of your workspace
      data_dir: str, [optional] directory of your training data, defaults to workspace
      model_type: str, e.g. 'Regressonset_regressoffset_frame_velocity_CRNN'
      loss_type: str, e.g. 'regress_onset_offset_frame_velocity_bce'
      augmentation: str, e.g. 'none'
      batch_size: int
      learning_rate: float
      reduce_iteration: int
      resume_iteration: int
      validation_iteration: int
      checkpoint_iteration: int
      early_stop: int
      device: 'cuda' | 'cpu'
      mini_data: bool
      bidirectional: bool [optional] defaults to True
    """

    # Arugments & parameters
    workspace = os.path.expanduser(args.workspace)
    data_dir = os.path.expanduser(args.data_dir) if args.data_dir else workspace
    model_type = args.model_type
    loss_type = args.loss_type
    augmentation = args.augmentation
    max_note_shift = args.max_note_shift
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    reduce_iteration = args.reduce_iteration
    resume_iteration = args.resume_iteration
    validation_iteration = args.validation_iteration
    checkpoint_iteration = args.checkpoint_iteration
    early_stop = args.early_stop
    bidirectional = args.bidirectional
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    mini_data = args.mini_data
    filename = args.filename

    sample_rate = config.sample_rate
    segment_seconds = config.segment_seconds
    hop_seconds = config.hop_seconds
    segment_samples = int(segment_seconds * sample_rate)
    frames_per_second = config.frames_per_second
    classes_num = config.classes_num
    num_workers = 8 

    # Loss function
    loss_func = get_loss_func(loss_type)

    def dir_path(base_path):
        return os.path.join(workspace, base_path, filename, 
            model_type, 'loss_type={}'.format(loss_type), 
            'augmentation={}'.format(augmentation), 
            'max_note_shift={}'.format(max_note_shift), 
            'batch_size={}'.format(batch_size),
            'bidirectional={}'.format(bidirectional))

    # Paths
    logs_dir = dir_path("logs")
    create_folder(logs_dir)
    
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    hdf5s_dir = os.path.join(data_dir, 'hdf5s', 'maestro')
    logging.info("Using training data from {}".format(hdf5s_dir))

    checkpoints_dir = dir_path("checkpoints")
    create_folder(checkpoints_dir)

    statistics_path = os.path.join(dir_path("statistics"), 'statistics.pkl')
    create_folder(os.path.dirname(statistics_path))

    if 'cuda' in str(device):
        logging.info('Using GPU.')
        device = 'cuda'
    else:
        logging.info('Using CPU.')
        device = 'cpu'
    
    # Model
    Model = eval(model_type)
    model = Model(frames_per_second=frames_per_second, classes_num=classes_num, bidirectional=bidirectional)

    if augmentation == 'none':
        augmentor = None
    elif augmentation == 'aug':
        augmentor = Augmentor()
    else:
        raise Exception('Incorrect argumentation!')
    
    # Dataset
    train_dataset = MaestroDataset(hdf5s_dir=hdf5s_dir, 
        segment_seconds=segment_seconds, frames_per_second=frames_per_second, 
        max_note_shift=max_note_shift, augmentor=augmentor)

    evaluate_dataset = MaestroDataset(hdf5s_dir=hdf5s_dir, 
        segment_seconds=segment_seconds, frames_per_second=frames_per_second, 
        max_note_shift=0)

    # Sampler for training
    train_sampler = Sampler(hdf5s_dir=hdf5s_dir, split='train', 
        segment_seconds=segment_seconds, hop_seconds=hop_seconds, 
        batch_size=batch_size, mini_data=mini_data)

    # Sampler for evaluation
    evaluate_train_sampler = TestSampler(hdf5s_dir=hdf5s_dir, 
        split='train', segment_seconds=segment_seconds, hop_seconds=hop_seconds, 
        batch_size=batch_size, mini_data=mini_data)

    evaluate_validate_sampler = TestSampler(hdf5s_dir=hdf5s_dir, 
        split='validation', segment_seconds=segment_seconds, hop_seconds=hop_seconds, 
        batch_size=batch_size, mini_data=mini_data)

    evaluate_test_sampler = TestSampler(hdf5s_dir=hdf5s_dir, 
        split='test', segment_seconds=segment_seconds, hop_seconds=hop_seconds, 
        batch_size=batch_size, mini_data=mini_data)

    # Dataloader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
        batch_sampler=train_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    evaluate_train_loader = torch.utils.data.DataLoader(dataset=evaluate_dataset, 
        batch_sampler=evaluate_train_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    validate_loader = torch.utils.data.DataLoader(dataset=evaluate_dataset, 
        batch_sampler=evaluate_validate_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(dataset=evaluate_dataset, 
        batch_sampler=evaluate_test_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    # Evaluator
    evaluator = SegmentEvaluator(model, batch_size)

    # Statistics
    statistics_container = StatisticsContainer(statistics_path)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    # Resume training
    if resume_iteration < 0:
        # load latest iteration
        resume_iteration = highest_iteration(checkpoints_dir)

    if resume_iteration > 0:
        resume_checkpoint_path = os.path.join(checkpoints_dir, 
                '{}_iterations.pth'.format(resume_iteration))

        logging.info('Loading checkpoint {}'.format(resume_checkpoint_path))
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        train_sampler.load_state_dict(checkpoint['sampler'])
        statistics_container.load_state_dict(resume_iteration)
        iteration = checkpoint['iteration']
        # Reduce learning rate
        if iteration > 0:
            for i in range(int(iteration / reduce_iteration)):
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.9

    else:
        iteration = 0
    
    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)

    train_bgn_time = time.time()
    iteration_time = time.time()

    try:
        for batch_data_dict in train_loader:
            
            # Evaluation 
            if iteration % validation_iteration == 0 and iteration > 0:
                logging.info('------------------------------------')
                logging.info('Iteration: {}'.format(iteration))

                train_fin_time = time.time()

                evaluate_train_statistics = evaluator.evaluate(evaluate_train_loader)
                validate_statistics = evaluator.evaluate(validate_loader)
                test_statistics = evaluator.evaluate(test_loader)

                logging.info('    Train statistics: {}'.format(evaluate_train_statistics))
                logging.info('    Validation statistics: {}'.format(validate_statistics))
                logging.info('    Test statistics: {}'.format(test_statistics))

                statistics_container.append(iteration, evaluate_train_statistics, data_type='train')
                statistics_container.append(iteration, validate_statistics, data_type='validation')
                statistics_container.append(iteration, test_statistics, data_type='test')
                statistics_container.dump()

                train_time = train_fin_time - train_bgn_time
                validate_time = time.time() - train_fin_time

                logging.info(
                    'Train time: {:.3f} s, validate time: {:.3f} s'
                    ''.format(train_time, validate_time))

                train_bgn_time = time.time()
            
            # Save model
            if iteration % checkpoint_iteration == 0 and resume_iteration != iteration:
                checkpoint = {
                    'iteration': iteration, 
                    'model': model.module.state_dict(), 
                    'sampler': train_sampler.state_dict()}

                checkpoint_path = os.path.join(
                    checkpoints_dir, '{}_iterations.pth'.format(iteration))
                    
                torch.save(checkpoint, checkpoint_path)
                logging.info('Model saved to {}'.format(checkpoint_path))
            
            # Reduce learning rate
            if iteration % reduce_iteration == 0 and iteration > 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.9
            
            # Move data to device
            for key in batch_data_dict.keys():
                batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)
             
            model.train()
            batch_output_dict = model(batch_data_dict['waveform'])

            loss = loss_func(model, batch_output_dict, batch_data_dict)

            iteration_time = time.time() - iteration_time
            rounded_time = round(iteration_time, 2)
            checkpoint_time = (checkpoint_iteration - (iteration % checkpoint_iteration)) * rounded_time
            remaining_time = (early_stop - iteration) * rounded_time
            checkpoint_time = datetime.timedelta(seconds=checkpoint_time)
            remaining_time = datetime.timedelta(seconds=remaining_time)
            print(iteration, loss, "{:.2f}s".format(iteration_time), 
                " chckpnt:", str(checkpoint_time).split(".")[0],
                " finish:", str(remaining_time).split(".")[0], end='\r')
            iteration_time = time.time()

            # Backward
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Stop learning
            if iteration == early_stop:
                break

            iteration += 1
    except KeyboardInterrupt:
        print("Interrupted! Stopping logger")
        logging.shutdown()
    finally:
        print("Stopping logger")
        logging.shutdown()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train') 
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--data_dir', type=str, required=False)
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--loss_type', type=str, required=True)
    parser_train.add_argument('--augmentation', type=str, required=True, choices=['none', 'aug'])
    parser_train.add_argument('--max_note_shift', type=int, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--learning_rate', type=float, required=True)
    parser_train.add_argument('--reduce_iteration', type=int, required=True)
    parser_train.add_argument('--resume_iteration', type=int, required=True)
    parser_train.add_argument('--validation_iteration', type=int, default=5000)
    parser_train.add_argument('--checkpoint_iteration', type=int, required=20000)
    parser_train.add_argument('--early_stop', type=int, required=True)
    parser_train.add_argument('--mini_data', action='store_true', default=False)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--bidirectional', dest='bidirectional', action='store_true')
    parser_train.add_argument('--unidirectional', dest='bidirectional', action='store_false')
    parser_train.set_defaults(bidirectional=True)
    
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    else:
        raise Exception('Error argument!')