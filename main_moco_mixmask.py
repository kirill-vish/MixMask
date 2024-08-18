#!/usr/bin/env python

import argparse
import builtins
import os
import random
import time
import warnings

import wandb
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np

import moco.loader
import moco.builder
import moco.builder_both
from utils import AverageMeter, ProgressMeter, MaskingGenerator, get_block_mask, \
      get_discrete_mask, save_checkpoint, rand_bbox, adjust_learning_rate, accuracy

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10002', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--moco-tm', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
# options for un-mix
parser.add_argument('--unmix_beta', default=1.0, type=float, help='beta distribution parameter')
parser.add_argument('--unmix_prob', default=0.5, type=float, metavar='U',
                    help='probability to choose local or global mixtures, please tune this to obtain better performance')
parser.add_argument('--exp_path', type=str, help='path to the experiment folder')
parser.add_argument('--method', type=str, default='mixmask', help='method to run: mixmask, unmix or both')
parser.add_argument('--checkpoint_freq', type=int, default=100, help='checkpoint_freq')
parser.add_argument('--grid_size', type=int, default=4, help='grid size for masking')
parser.add_argument('--mask_lam', type=float, default=0.5, help='lambda for mask')
parser.add_argument('--mask_type', type=str, default='block', help="mask type")
parser.add_argument('--debug', action='store_true', help='debug mode')

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    args.size = 224
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    run = None
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        if args.rank == 0:
            name = f"train_{args.exp_path.split('/')[-1]}"
            run = wandb.init(
                name=name, 
                project="mixmask",
                config=args,
                mode='disabled' if args.debug else 'online'
            )
            wandb.save('main_moco_mixmask.py', policy='now')
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.method == 'mixmask' or args.method == 'unmix':
        model = moco.builder.MoCo(
            models.__dict__[args.arch],
            args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.moco_tm, args.mlp)
    elif args.method == 'both':
        model = moco.builder_both.MoCoUnMixMixMask(
            models.__dict__[args.arch],
            args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.moco_tm, args.mlp)
    print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    train_dataset = datasets.ImageFolder(
        traindir,
        moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    best_train_acc1 = 0
    mask_lam_arr = []
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_loss, top1, top5, lam_arr = train(train_loader, model, criterion, optimizer, epoch, args, run)
        mask_lam_arr.extend(lam_arr)
        if run is not None:
            best_train_acc1 = max(top1, best_train_acc1)
            run.log({"epoch": epoch + 1, 
                     "train/loss": train_loss, 
                     "train/acc1": top1, 
                     "train/acc5": top5, \
                     "mask_lam": wandb.Histogram(mask_lam_arr)})
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):

            if (epoch + 1) % args.checkpoint_freq == 0 or epoch + 1 == args.epochs:
                fname = os.path.join(args.exp_path, f'{epoch + 1}.pth.tar')
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best=False, filename=fname)
        sys.stdout.flush()

    if run is not None:
        run.log({"train/best_acc1": best_train_acc1})
        sys.stdout.flush()
        run.finish()

def train(train_loader, model, criterion, optimizer, epoch, args, run=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    mask_generator = MaskingGenerator(input_size=args.grid_size, num_masking_patches=int(args.mask_lam * args.grid_size ** 2))
    mask_lam_arr = []
    for i, (images, _) in enumerate(train_loader):
        mask_lam_arr.append(args.mask_lam)
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
        
        r = np.random.rand(1)
        images_reverse = torch.flip(images[0], (0,))
        if args.method == 'mixmask':
            if args.mask_type == 'discrete':
                mask = get_discrete_mask(args)
            elif args.mask_type == 'block':
                mask = get_block_mask(mask_generator, args)
            mask = mask.to(images_reverse.device)
            mixed_images = mask * images[0] + (1 - mask) * images_reverse
            lam = mask.sum() / (mask.size()[-2] * mask.size()[-1])
        elif args.method == 'unmix':
            if r < args.unmix_prob:
                mixed_images = lam * images[0] + (1 - lam) * images_reverse
            else:
                mixed_images = images[0].clone()
                bbx1, bby1, bbx2, bby2 = rand_bbox(images[0].size(), lam)
                mixed_images[:, :, bbx1:bbx2, bby1:bby2] = images_reverse[:, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images[0].size()[-1] * images[0].size()[-2]))
        elif args.method == 'both':
            # unmix
            lam_unmix = np.random.beta(args.unmix_beta, args.unmix_beta)
            if r < args.unmix_prob:
                mixed_images_unmix = lam_unmix * images[0] + (1 - lam_unmix) * images_reverse
            else:
                mixed_images_unmix = images[0].clone()
                bbx1, bby1, bbx2, bby2 = rand_bbox(images[0].size(), lam_unmix)
                mixed_images_unmix[:, :, bbx1:bbx2, bby1:bby2] = images_reverse[:, :, bbx1:bbx2, bby1:bby2]
                lam_unmix = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images[0].size()[-1] * images[0].size()[-2]))
            
            # mixmask
            index_mask = torch.randperm(len(images[0])) # using different purmutation
            if args.mask_type == 'discrete':
                mask = get_discrete_mask(args)
            elif args.mask_type == 'block':
                mask = get_block_mask(mask_generator, args)
            mask = mask.to(images[0].device)
            mixed_images_mask = mask * images[0] + (1 - mask) * images[0][index_mask]
            lam_mask = mask.sum() / (mask.size()[-2] * mask.size()[-1])

        # compute output
        if args.method == 'mixmask' or args.method == 'unmix':
            output, target, output_m, output_m_flip = model(im_q=images[0], im_k=images[1], im_qm = mixed_images)
            loss_o = criterion(output, target)
            loss_m = criterion(output_m, target)
            loss_m_flip = criterion(output_m_flip, target)
            loss = loss_o + lam * loss_m + (1 - lam) * loss_m_flip
        elif args.method == 'both':
            output, target, output_m, output_m_flip, output_mask, output_mask_flip = model(im_q=images[0], im_k=images[1], \
            im_qm=mixed_images_unmix, im_q_mask=mixed_images_mask, index_mask=index_mask)
        
            loss_o = criterion(output, target)
            loss_m = criterion(output_m, target)
            loss_m_flip = criterion(output_m_flip, target)
            loss_mask = criterion(output_mask, target)
            loss_mask_flip = criterion(output_mask_flip, target)

            loss = loss_o + lam_unmix * loss_m + (1 - lam_unmix) * loss_m_flip + lam_mask * loss_mask + (1 - lam_mask) * loss_mask_flip

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            if run is not None:
                run.log({"train_batch/loss": loss.item(), "train_batch/acc1": acc1[0], "train_batch/acc5": acc5[0]})

    return losses.avg, top1.avg, top5.avg, mask_lam_arr

if __name__ == '__main__':
    main()
