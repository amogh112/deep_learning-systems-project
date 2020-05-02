import os
import time
import glob
import shutil
import platform
import random
import argparse

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets

import copy
import numpy as np
import signal
import sys
import math
from collections import defaultdict
import scipy.stats

# from ptflops import get_model_complexity_info

import models.taskonomy_models as models


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Taskonomy Training')

    parser.add_argument('--dataset',        type=str,   required=True)
    parser.add_argument('--model',          type=str,   required=True)
    parser.add_argument('--loading',        action='store_true')
    parser.add_argument('--resume',         action='store_true')
    parser.add_argument('--resume_path', type=str, default='', help='sgd/adam')
    parser.add_argument('--debug',          action='store_true')
    parser.add_argument('--equally',          action='store_true')
    parser.add_argument('--customize_class',action='store_true')
    parser.add_argument('--comet',          action='store_true')
    parser.add_argument('--class_to_train', type=str)
    parser.add_argument('--class_to_test',  type=str)
    parser.add_argument('--seed',           type=int,       default=42,     help='seed')
    parser.add_argument('--mt_lambda',      type=float,     default=0,      help='mt_lambda')
    parser.add_argument('--step_size_schedule',type=str,    default='[[0, 0.01], [140, 0.001], [200, 0.0001]]', help='lr schedule')
    parser.add_argument('--optim',          type=str,       default='sgd',  help='sgd/adam')
    parser.add_argument('--adv_train',      action='store_true')
    parser.add_argument('--customize_schedule',      action='store_true')
    parser.add_argument('--schedule_str',type=str,    default='s', help='schedule string')
    parser.add_argument('--prune', type=str, default=None, help='config name for pruning')
    # parser.add_argument('--data_dir', '-d', dest='data_dir', required=True,
    #                     help='path to training set')
    # parser.add_argument('--arch', '-a', metavar='ARCH', required=True,
    #                     choices=model_names,
    #                     help='model architecture: ' +
    #                          ' | '.join(model_names) +
    #                          ' (required)')
    # parser.add_argument('-b', '--batch-size', default=64, type=int,
    #                     help='mini-batch size (default: 64)')
    # parser.add_argument('--tasks', '-ts', default='sdnkt', dest='tasks',
    #                     help='which tasks to train on')
    # parser.add_argument('--model_dir', default='models', dest='model_dir',
    #                     help='where to save models')
    # parser.add_argument('--image-size', default=256, type=int,
    #                     help='size of image side (images are square)')
    # parser.add_argument('-j', '--workers', default=4, type=int,
    #                     help='number of data loading workers (default: 4)')
    # parser.add_argument('--epochs', default=100, type=int,
    #                     help='maximum number of epochs to run')
    # parser.add_argument('-mlr', '--minimum_learning_rate', default=3e-5, type=float,
    #                     metavar='LR', help='End trianing when learning rate falls below this value.')
    #
    # # parser.add_argument('-lr', '--learning-rate', dest='lr', default=0.1, type=float,
    # #                     metavar='LR', help='initial learning rate')
    # parser.add_argument('-ltw0', '--loss_tracking_window_initial', default=500000, type=int,
    #                     help='inital loss tracking window (default: 500000)')
    # parser.add_argument('-mltw', '--maximum_loss_tracking_window', default=2000000, type=int,
    #                     help='maximum loss tracking window (default: 2000000)')
    # parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
    #                     help='momentum')
    # parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
    #                     metavar='W', help='weight decay (default: 1e-4)')
    # parser.add_argument('--resume', '--restart', default='', type=str, metavar='PATH',
    #                     help='path to latest checkpoint (default: none)')
    # # parser.add_argument('--start-epoch', default=0, type=int,
    # #                     help='manual epoch number (useful on restarts)')
    # parser.add_argument('-n', '--experiment_name', default='', type=str,
    #                     help='name to prepend to experiment saves.')
    # parser.add_argument('-v', '--validate', dest='validate', action='store_true',
    #                     help='evaluate model on validation set')
    # parser.add_argument('-t', '--test', dest='test', action='store_true',
    #                     help='evaluate model on test set')
    #
    # # parser.add_argument('-r', '--no_rotate_loss', dest='no_rotate_loss', action='store_true',
    # #                     help='should loss rotation occur')
    # parser.add_argument('--pretrained', dest='pretrained', default='',
    #                     help='use pre-trained model')
    # parser.add_argument('-vb', '--virtual-batch-multiplier', default=1, type=int,
    #                     metavar='N', help='number of forward/backward passes per parameter update')
    # parser.add_argument('--fp16', action='store_true',
    #                     help='Run model fp16 mode.')
    # parser.add_argument('-ml', '--model-limit', default=None, type=int,
    #                     help='Limit the number of training instances from a single 3d building model.')



    cudnn.benchmark = False
    args = parser.parse_args()
    import socket, json
    # print(glob.glob("./config/*"))
    print("here ", os.listdir("./"))
    print("here ", os.listdir("config"))
    config_file_path = "config/{}_{}_config.json".format(args.model, args.dataset)
    with open (config_file_path) as config_file:
        config = json.load(config_file)
    if socket.gethostname() == "amogh":
        args.data_dir = config['data-dir_amogh']
        args.pretrained = config['pretrained_amogh']
        args.backup_output_dir = config['backup_output_dir_amogh']
    elif "proj" in socket.gethostname():
        args.data_dir = config['data-dir_amogh_instance']
        args.pretrained = config['pretrained_amogh_instance']
        args.backup_output_dir = config['backup_output_dir_amogh_instance']
    else:
        args.data_dir = config['data-dir']
        args.pretrained = config['pretrained']
        args.backup_output_dir = config['backup_output_dir']

    args.step = config['step']
    args.arch = config['arch']
    args.batch_size = config['batch-size']
    args.test_batch_size = config['test-batch-size']
    args.epochs = config['epochs']
    # args.mt_lambda = config['mt_lambda']


    args.lr_change = config['lr_change']
    args.lr = config['lr']
    args.lr_mode = config['lr-mode']
    args.momentum = config['momentum']
    args.weight_decay = config['weight-decay']

    import ast
    args.step_size_schedule = ast.literal_eval(args.step_size_schedule)


    args.workers = config['workers']

    args.print_freq = config['print_freq']

    args.classes = config['classes']



    # ADDED FOR CITYSCAPES
    args.random_scale = config['random-scale']
    args.random_rotate = config['random-rotate']
    args.crop_size = config['crop-size']
    args.list_dir = config['list-dir']

    if args.customize_class:  # TODO: Notice here the sequence of each task is hard coded, during the testing, this sequence must be fully followed.
        # TODO: because the nn.ModuleList does not have the key for each decoder, thus if decoder is swithched sequence during loading, error will occur,
        # Even if no error is raised, the decoder is loaded with wrong weights thus results would be wrong.

        t_list = []
        if 's' in args.class_to_train:
            t_list.append("segmentsemantic")
        if 'd' in args.class_to_train:
            t_list.append("depth_zbuffer")
        if 'e' in args.class_to_train:
            t_list.append("edge_texture")
        if 'k' in args.class_to_train:
            t_list.append("keypoints2d")
        if 'n' in args.class_to_train:
            t_list.append("normal")
        if 'r' in args.class_to_train:
            t_list.append("reshading")

        if 'K' in args.class_to_train:
            t_list.append("keypoints3d")
        if 'D' in args.class_to_train:
            t_list.append("depth_euclidean")
        if 'A' in args.class_to_train:
            t_list.append("autoencoder")
        if 'E' in args.class_to_train:
            t_list.append("edge_occlusion")
        if 'p' in args.class_to_train:
            t_list.append("principal_curvature")
        if 'u' in args.class_to_train:
            t_list.append("segment_unsup2d")
        if 'U' in args.class_to_train:
            t_list.append("segment_unsup25d")


        test_t_list = []
        if 's' in args.class_to_test:
            assert 's' in args.class_to_train
            test_t_list.append("segmentsemantic")
        if 'd' in args.class_to_test:
            test_t_list.append("depth_zbuffer")
        if 'e' in args.class_to_test:
            test_t_list.append("edge_texture")
        if 'k' in args.class_to_test:
            test_t_list.append("keypoints2d")
        if 'n' in args.class_to_test:
            test_t_list.append("normal")
        if 'r' in args.class_to_test:
            test_t_list.append("reshading")

        if 'K' in args.class_to_test:
            test_t_list.append("keypoints3d")
        if 'D' in args.class_to_test:
            test_t_list.append("depth_euclidean")
        if 'A' in args.class_to_test:
            test_t_list.append("autoencoder")
        if 'E' in args.class_to_test:
            test_t_list.append("edge_occlusion")
        if 'p' in args.class_to_test:
            test_t_list.append("principal_curvature")
        if 'u' in args.class_to_test:
            test_t_list.append("segment_unsup2d")
        if 'U' in args.class_to_test:
            test_t_list.append("segment_unsup25d")

        args.task_set = t_list
        args.test_task_set = test_t_list

    else:
        args.task_set = config['task_set']
        args.test_task_set = config['test_task_set']

    args.adv_val = config['adv_val']
    args.val_freq = config['val_freq']
    # args.adv_train = False

    args.epsilon = config['epsilon']
    args.step_size = config['step_size']
    args.steps = config['steps']

    if args.customize_schedule:
        schedule = args.schedule_str
        task_lists = schedule.split(";")
        final_schedule = []
        for task_list in task_lists:
            task_list_final = []
            if 's' in task_list:
                task_list_final.append("segmentsemantic")
            if 'd' in task_list:
                task_list_final.append("depth_zbuffer")
            if 'A' in task_list:
                task_list_final.append("autoencoder")
            final_schedule.append(task_list_final)

        args.adv_train_task_schedule = final_schedule

    args.prune_config = None 
    if args.prune != None:
        args.prune_config = args.prune
        prune_config_file_path = "config/pruning/{}.json".format(args.prune)
        with open (prune_config_file_path) as config_file:
            args.prune = json.load(config_file)
    print(args.prune)
    return args


def main():
    args = parse_args()
    # args.arch = 'res18'
    print('starting on', platform.node())
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        print('cuda gpus:', os.environ['CUDA_VISIBLE_DEVICES'])

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        main_stream = torch.cuda.Stream()

        torch.cuda.set_rng_state(torch.cuda.get_rng_state())
        torch.backends.cudnn.deterministic = True

    from learning.mtask_train_loop import train_mtasks
    train_mtasks(args)

if __name__ == '__main__':
    main()
