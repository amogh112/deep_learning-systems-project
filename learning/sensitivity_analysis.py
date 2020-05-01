
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from comet_ml import Experiment

# import models.drn as drn
# from models.DRNSeg import DRNSeg
# from models.FCN32s import FCN32s
# import data_transforms as transforms
import json
import math
import os
from os.path import exists, join, split
import threading

import time, datetime

import numpy as np
import shutil

import sys
from PIL import Image
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from learning.utils_learn import *
from learning.dataloader import SegList, SegListMS, get_loader, get_info
import logging
from learning.validate import validate
# import data_transforms as transforms

from dataloaders.utils import decode_segmap

from torch.utils.tensorboard import SummaryWriter
from learning.mtask_validate import mtask_validate
import torchvision
from learning.prune_models import prune_model


FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def validate_mtasks_sensitivity_analysis(args):
    batch_size = args.batch_size
    num_workers = args.workers


    print(' '.join(sys.argv))

    for k, v in args.__dict__.items(): # Prints arguments and contents of config file
        print(k, ':', v)

    from models.mtask_losses import get_losses_and_tasks

    criteria, taskonomy_tasks = get_losses_and_tasks(args)

    if args.arch == 'resnet-18':
        from models.taskonomy_models import resnet18_taskonomy
        model = resnet18_taskonomy(pretrained=False, tasks=args.task_set)

    elif args.arch == 'resnet-50':
        from models.taskonomy_models import resnet50_taskonomy
        model = resnet50_taskonomy(pretrained=False, tasks=args.task_set)

    if args.pretrained and args.loading:
        print('args.pretrained', args.pretrained)
        model.load_state_dict(torch.load(args.pretrained))

    out_dir = 'output/{}_{:03d}'.format(args.arch, 0)

    print("including the following tasks:", taskonomy_tasks)

    model = torch.nn.DataParallel(model)

    if torch.cuda.is_available():
        model.cuda()

    # Data loading code
    info = get_info(args.dataset)
    train_loader = get_loader(args, "train")
    val_loader = get_loader(args, "val", out_name=True)
    adv_val_loader = get_loader(args, "adv_val", out_name=True)

    # define loss function (criterion) and optimizer
    if args.optim == 'sgd':
        print("Using SGD")
        optimizer = torch.optim.SGD(model.parameters(),
                                    args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        print("Using Adam")
        optimizer = torch.optim.Adam(model.parameters())
    

    cudnn.benchmark = True
    best_prec1 = 0
    start_epoch = 0

    # Backup files before resuming/starting training
    backup_output_dir = args.backup_output_dir

    os.makedirs(backup_output_dir, exist_ok=True)

    if os.path.exists(backup_output_dir):
        import uuid
        unique_str = str(uuid.uuid4())[:8]
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

        if args.prune_config:
            experiment_name = "train_" + args.arch + "_" + args.arch + "_" + args.dataset + "_" + timestamp + "_" + unique_str\
                                       + "_trainset_{}_testset_{}_lambda_{}_seed_{}_lrs_{}_{}_prune_{}".format(args.class_to_train, args.class_to_test, args.mt_lambda, args.seed, args.step_size_schedule[1][0], args.step_size_schedule[2][0],args.prune_config)
        else:
            experiment_name = "train_" + args.arch + "_" + args.arch + "_" + args.dataset + "_" + timestamp + "_" + unique_str \
                              + "_trainset_{}_testset_{}_lambda_{}_seed_{}_lrs_{}_{}".format(
                args.class_to_train, args.class_to_test, args.mt_lambda, args.seed, args.step_size_schedule[1][0],
                args.step_size_schedule[2][0])

        if args.equally:
            experiment_name = experiment_name + "_equal"
        experiment_backup_folder = os.path.join(backup_output_dir, experiment_name)
        print(experiment_backup_folder)
        shutil.copytree('.', experiment_backup_folder, ignore=include_patterns('*.py', '*.json'))

    # Logging with TensorBoard
    log_dir = os.path.join(experiment_backup_folder, "runs")

    # os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    eval_writer = SummaryWriter(log_dir=log_dir + '/validate_runs/')

    fh = logging.FileHandler(experiment_backup_folder+'/log.txt')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    
        
    #for different configs of pruning, get validation score
    backbone = model.module.encoder
    
    module_names = [name for name, module in backbone.named_modules() if isinstance(module, torch.nn.Conv2d)]
    results_encoder = []
    results_normal = []
    results_reshading = []
    for name in module_names:
        layer_results_encoder = [name]
        layer_results_normal = [name]
        layer_results_reshading = [name]
        #layer by layer
        for prune_factor in [0, 0.2, 0.4, 0.6 , 0.8, 1.0]:
            #start from a fresh model without prune masks
            if args.arch == 'resnet-18':
                from models.taskonomy_models import resnet18_taskonomy
                model = resnet18_taskonomy(pretrained=False, tasks=args.task_set)

            elif args.arch == 'resnet-50':
                from models.taskonomy_models import resnet50_taskonomy
                model = resnet50_taskonomy(pretrained=False, tasks=args.task_set)
            model = torch.nn.DataParallel(model)

            #resume from checkpoint       
            if args.resume:
                print("resuming", args.resume_path)
                if os.path.isfile(args.resume_path):
                    print("=> loading checkpoint '{}'".format(args.resume_path))
                    checkpoint = torch.load(args.resume_path)
                    start_epoch = checkpoint['epoch']
                    best_prec1 = checkpoint['best_prec1']
                    model.load_state_dict(checkpoint['state_dict'])
                    print("=> loaded checkpoint '{}' (epoch {})"
                          .format(args.resume_path, checkpoint['epoch']))
            
            #prune according to config
            #using the config file just for the strategy, amount is decided according to the experiment
            prune_config = args.prune
            prune_config['amount'] = prune_factor
            prune_config['layers'] = [name]
            print(prune_config['layers'])
            model = prune_model(model, args.prune)
            
            torch.cuda.empty_cache()

            # evaluate on validation set
            losses = mtask_validate(val_loader, model, criteria, eval_writer, args=args, info=info, return_all_losses=True)
            layer_results_encoder.append(losses[0])
            layer_results_normal.append(losses[1])
            layer_results_reshading.append(losses[2])
            print('Prune_factor {}, Total_loss {}'.format(prune_factor, losses))
            
        results_encoder.append(layer_results_encoder)
        results_normal.append(layer_results_normal)
        results_reshading.append(layer_results_reshading)
    
    import pandas as pd
    df_encoder = pd.DataFrame(results_encoder)
    df_normal = pd.DataFrame(results_normal)
    df_reshading = pd.DataFrame(results_reshading)
    
    df_encoder.to_csv('encoder.csv')
    df_normal.to_csv('normal.csv')
    df_reshading.to_csv('reshading.csv')
    

    writer.close()