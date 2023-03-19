#!/usr/bin/env python
import os
import time
import json
import torch.optim
import torchvision

import seed.builder
import torch.nn.parallel
import seed.models as models
import torch.distributed as dist
from tools.opts import parse_opt
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
from tools.dataset import TSVDataset
from tools.logger import setup_logger
from torch.utils.tensorboard import SummaryWriter
from tools.utils import simclr_aug, mocov1_aug, mocov2_aug, swav_aug, adjust_learning_rate, \
     soft_cross_entropy,  AverageMeter, ValueMeter, ProgressMeter, resume_training, \
     load_simclr_teacher_encoder, load_moco_teacher_encoder, load_swav_teacher_encoder, save_checkpoint


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))



def main(args):
    global logger

    # set-up the output directory
    os.makedirs(args.output, exist_ok=True)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        cudnn.benchmark = True

        # create logger
        logger = setup_logger(output=args.output, distributed_rank=dist.get_rank(),
                              color=False, name="SEED")

        if dist.get_rank() == 0:
            path = os.path.join(args.output, "config.json")
            with open(path, 'w') as f:
                json.dump(vars(args), f, indent=2)
            logger.info("Full config saved to {}".format(path))

        # save the distributed node machine
        logger.info('world size: {}'.format(dist.get_world_size()))
        logger.info('local_rank: {}'.format(args.local_rank))
        logger.info('dist.get_rank(): {}'.format(dist.get_rank()))

    else:
        # create logger
        logger = setup_logger(output=args.output, color=False, name="SEED")

        logger.info('Single GPU mode for debugging.')

    # create model
    logger.info("=> creating student encoder '{}'".format(args.student_arch))
    logger.info("=> creating teacher encoder '{}'".format(args.teacher_arch))

    # use SimCLR and SWAV used their customized ResNet architecture with minor differences.
    if args.teacher_ssl != 'moco':
        args.teacher_arch = args.teacher_ssl + '_' + args.teacher_arch

    # some architectures are not supported yet. It needs to be expanded manually.
    assert args.teacher_arch in models.__dict__

    # SWAV have different MLP length
    if args.teacher_ssl == 'swav':
        # hidden_dim: resnet50-2048, resnet50w4-8192, resnet50w5-10240
        if args.teacher_arch == 'swav_resnet50':
            swav_mlp = 2048
        elif args.teacher_arch == 'swav_resnet50w2':
            swav_mlp = 8192
        elif args.teacher_arch == 'swav_resnet50w4':
            swav_mlp = 8192
        elif args.teacher_arch == 'swav_resnet50w5':
            swav_mlp = 10240

        # initialize model object, feed student and teacher into encoders.
        model = seed.builder.SEED(models.__dict__[args.student_arch],
                                  models.__dict__[args.teacher_arch],
                                  args.dim,
                                  args.queue,
                                  args.temp,
                                  mlp=args.student_mlp,
                                  temp=args.distill_t,
                                  dist=args.distributed,
                                  swav_mlp=swav_mlp,
                                  stu=args.teacher_ssl)

    else:
        # initialize model object, feed student and teacher into encoders.
        model = seed.builder.SEED(models.__dict__[args.student_arch],
                                  models.__dict__[args.teacher_arch],
                                  args.dim,
                                  args.queue,
                                  args.temp,
                                  mlp=args.student_mlp,
                                  temp=args.distill_t,
                                  dist=args.distributed,
                                  stu=args.teacher_ssl)

    logger.info(model)

    if args.distributed:
        logger.info('Entering distributed mode.')

        model = torch.nn.parallel.DistributedDataParallel(model.cuda(),
                                                          device_ids=[args.local_rank],
                                                          broadcast_buffers=False,
                                                          find_unused_parameters=True)

        logger.info('Model now distributed.')

        args.lr_mult = args.batch_size / 256
        args.warmup_epochs = 5
        optimizer = torch.optim.SGD(model.parameters(),
                                    args.lr_mult * args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        # tensorboard
        if dist.get_rank() == 0:
            summary_writer = SummaryWriter(log_dir=args.output)
        else:
            summary_writer = None

    else:
        args.lr_mult = 1
        args.warmup_epochs = 5

        model = model.cuda()

        optimizer = torch.optim.SGD(model.parameters(), args.lr,  momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        summary_writer = SummaryWriter(log_dir=args.output)

    # load the SSL pre-trained teacher encoder into model.teacher
    if args.distill:
        if os.path.isfile(args.distill):
            if args.teacher_ssl == 'moco':
                model = load_moco_teacher_encoder(args, model, logger, distributed=args.distributed)
            elif args.teacher_ssl == 'simclr':
                model = load_simclr_teacher_encoder(args, model, logger, distributed=args.distributed)
            elif args.teacher_ssl == 'swav':
                model = load_swav_teacher_encoder(args, model, logger, distributed=args.distributed)

            logger.info("=> Teacher checkpoint successfully loaded from '{}'".format(args.distill))
        else:
            logger.info("wrong distillation checkpoint.")

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            model = resume_training(args, model, optimizer, logger)
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # clear unnecessary weights
    torch.cuda.empty_cache()

    if args.teacher_ssl == 'swav': augmentation = swav_aug
    elif args.teacher_ssl == 'simclr': augmentation = simclr_aug
    elif args.teacher_ssl == 'moco' and args.student_mlp: augmentation = mocov2_aug
    else: augmentation = mocov1_aug

    # train_dataset = TSVDataset(os.path.join(args.data, 'train.tsv'), augmentation)
    # train_dataset=torchvision.datasets.ImageFolder(args.data,transform=augmentation)
    # train_dataset=torchvision.datasets.CIFAR100(args.data, train=True,download=True)
    train_dataset=torchvision.datasets.CIFAR100(root='../../data/cifar-100-python/', train=True,
                                                transform=torchvision.transforms.ToTensor(),download=True)
    val_dataset = torchvision.datasets.CIFAR100(root='../../data/cifar-100-python/', train=False,
                                                  transform=torchvision.transforms.ToTensor(), download=True)
    logger.info('Dataset done.')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

        # ensure batch size is dividable by # of GPUs
        assert args.batch_size // dist.get_world_size() == args.batch_size / dist.get_world_size(), \
            'Batch size is not divisible by num of gpus.'

        # create distributed dataloader
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    else:
        # create distributed dataloader
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True,
            drop_last=True)
        val_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True,
            drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed: train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        loss = train(train_loader, model, soft_cross_entropy, optimizer, epoch, args, logger)

        # evaluate on validation set
        acc1 = validate(val_loader, model, soft_cross_entropy, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if summary_writer is not None:
            # Tensor-board logger
            summary_writer.add_scalar('train_loss', loss, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # if dist.get_rank() == 0:

        file_str = 'Teacher_{}_T-Epoch_{}_Student_{}_distill-Epoch_{}-checkpoint_{:04d}.pth.tar'\
            .format(args.teacher_ssl, args.epochs, args.student_arch, args.teacher_arch, epoch)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.student_arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(args.output, file_str))

        logger.info('==============> checkpoint saved to {}'.format(os.path.join(args.output, file_str)))


def train(train_loader, model, criterion, optimizer, epoch, args, logger):
    batch_time = AverageMeter('Batch Time', ':5.3f')
    data_time = AverageMeter('Data Time', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    lr = ValueMeter('LR', ':5.3f')
    mem = ValueMeter('GPU Memory Used', ':5.0f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, lr, losses, mem],
        prefix="Epoch: [{}]".format(epoch))

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    lr.update(get_learning_rate(optimizer))
    mem.update(torch.cuda.max_memory_allocated(device=0) / 1024.0 / 1024.0)

    # switch to train mode
    model.train()

    # make key-encoder at eval to freeze BN
    if args.distributed:
        model.module.teacher.eval()

        # check the sanity of key-encoder
        for name, param in model.module.teacher.named_parameters():
            if param.requires_grad:
                logger.info("====================> Key-encoder Sanity Failed, parameters are not frozen.")

    else:
        model.teacher.eval()

        # check the sanity of key-encoder
        for name, param in model.teacher.named_parameters():
           if param.requires_grad:
                logger.info("====================> Key-encoder Sanity Failed, parameters are not frozen.")

    end = time.time()

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for i, (images, _) in enumerate(train_loader):

        if not args.distributed:
            images = images.cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        with torch.cuda.amp.autocast(enabled=True):

            logit, label = model(image=images)
            loss = criterion(logit, label)

        losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, logger)

    return losses.avg

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    top1 = AverageMeter('Acc@1', ':5.2f')
    top5 = AverageMeter('Acc@5', ':5.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            output,_ = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main(parse_opt())
