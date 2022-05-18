import argparse
import copy
import datetime
import importlib
import logging
import os
import sys
from pathlib import Path
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations

from torch.autograd import Variable

from data_utils import provider
from data_utils.ModelNetDataLoader import ModelNetDataLoader, ExemplarDataset
from data_utils.ShapeNetDataLoader import ShapeNetDataLoader
from data_utils.ScanObjectNNDataLoader import ScanObjectNNDataLoader
from data_utils.exemplar import icarl_construct_exemplar_set, icarl_construct_exemplar_sets
from lib.util import LabelSmoothingCrossEntropy, mixup_data, mixup_criterion, weight_norm

# from tensorboardX import SummaryWriter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))

avg_acc = []


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--model', default='vn_pointnet_cls', help='Model name [default: pointnet_cls]',
                        choices=['pointnet_cls', 'vn_pointnet_cls', 'dgcnn_cls', 'vn_dgcnn_cls', 'eqcnn_cls'])

    # training hyperparameters
    parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=40, help='number of training epochs')
    parser.add_argument('--start_epoch', type=int, default=1, help='number of training epochs')
    parser.add_argument('--epochs_sd', type=int, default=15, help='number of training epochs for self-distillation')
    parser.add_argument('--K', type=int, default=1000, help='memory budget')
    parser.add_argument('--save_freq', type=int, default=1, help='memory budget')
    parser.add_argument('--gpu', type=str, default='0', help='Specify gpu device [default: 0]')

    # incremental learning
    parser.add_argument('--new_classes', type=int, default=4, help='number of classes in new task')
    parser.add_argument('--start_classes', type=int, default=4, help='number of classes in old task')

    # optimization
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr-min', type=float, default=0.0005, help='lower end of cosine decay')
    parser.add_argument('--lr_sd', type=float, default=0.1, help='learning rate for self-distillation')  #之前0.01
    parser.add_argument('--lr_ft', type=float, default=0.01, help='learning rate for task-2 onwards')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')  # 之前1e-4
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--cosine', action='store_true', help='use cosine learning rate')

    # root folders
    parser.add_argument('--data_root', type=str, default='/home/huangyanhui/wxh/VNPCCL2.0/data/modelnet40_normal_resampled',
                        help='root directory of dataset')
    parser.add_argument('--output_root', type=str, default='./output', help='root directory for output')

    # save and load
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--resume', action='store_true', help='use class moco')
    parser.add_argument('--resume_path', type=str, default='./output/classification/2022-03_11_16_09/', )
    parser.add_argument('--save', action='store_true', help='to save checkpoint')

    # loss function
    parser.add_argument('--pow', type=float, default=0.66, help='hyperparameter of adaptive weight')
    parser.add_argument('--lamda', type=float, default=10, help='weighting of classification and distillation')
    parser.add_argument('--lamda_sd', type=float, default=10, help='weighting of classification and distillation')
    parser.add_argument('--const_lamda', action='store_true',
                        help='use constant lamda value, default: adaptive weighting')

    parser.add_argument('--w_cls', type=float, default=1.0, help='weightage of new classification loss')

    # kd loss
    parser.add_argument('--kd', action='store_true', help='use kd loss')
    parser.add_argument('--w-kd', type=float, default=1.0, help='weightage of knowledge distillation loss')
    parser.add_argument('--T', type=float, default=2, help='temperature scaling for KD')
    parser.add_argument('--T-sd', type=float, default=2, help='temperature scaling for KD')

    # self-distillation
    parser.add_argument('--num_sd', type=int, default=0, help='number of self-distillation generations')
    parser.add_argument('--sd_factor', type=float, default=5.0,
                        help='weighting between classification and distillation')

    # mixup
    parser.add_argument('--mixup', action='store_true', help='use mixup augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.1, help='mixup alpha value')

    # label smoothing
    parser.add_argument('--label_smoothing', action='store_true', help='use label smoothing')
    parser.add_argument('--smoothing_alpha', type=float, default=0.1, help='label smoothing alpha value')

    # PC
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')

    # VNN
    parser.add_argument('--pooling', type=str, default='mean', help='VNN only: pooling method [default: mean]',
                        choices=['mean', 'max'])
    parser.add_argument('--rot', type=str, default='so3',
                        help='Rotation augmentation to input data [default: aligned]',
                        choices=['aligned', 'z', 'so3'])
    parser.add_argument('--test_rot', type=str, default='so3',
                        help='Rotation augmentation to input data [default: aligned]',
                        choices=['aligned', 'z', 'so3'])
    parser.add_argument('--n_knn', default=20, type=int,
                        help='Number of nearest neighbors to use, not applicable to PointNet [default: 20]')
    # dgcnn
    parser.add_argument('--dropout', type=float, default=0.5, help='initial dropout rate')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')

    # tesnorboard
    parser.add_argument('--tensorboard', action='store_true', help='查看loss曲线')

    parser.add_argument('--aug', action='store_true', help='data augment')

    parser.add_argument('--datasets', type=str, default='shapenetcorev2', help='_____',
                        choices=['modelnet40', 'shapenetcorev2', 'scanobjectnn'])
    parser.add_argument('--seed_num', type=int, default=1996, help='__')
    args = parser.parse_args()
    return args


def train(model, old_model, epoch, lr, tempature, lamda, train_loader, use_sd, checkPoint):
    tolerance_cnt = 0
    step = 0
    best_acc = 0
    T = args.T

    model.cuda()
    old_model.cuda()

    model = nn.DataParallel(model)
    old_model = nn.DataParallel(old_model)

    criterion_ce = nn.CrossEntropyLoss()  #ignore_index=-1
    criterion_ce_smooth = LabelSmoothingCrossEntropy()  # for label smoothing

    # reduce learning rate after first epoch (LowLR)
    if len(test_classes) // CLASS_NUM_IN_BATCH > 1:
        lr = args.lr_ft

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.weight_decay)

    if len(test_classes) // CLASS_NUM_IN_BATCH == 1 and use_sd == True:
        if args.cosine:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=0.001)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60], gamma=0.1)
    else:
        if args.cosine:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=args.lr_min)
        else:
            # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 90], gamma=0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    if len(test_classes) // CLASS_NUM_IN_BATCH > 1:
        exemplar_set = ExemplarDataset(exemplar_sets)
        exemplar_loader = torch.utils.data.DataLoader(exemplar_set, batch_size=args.batch_size, shuffle=True,
                                                      num_workers=args.num_workers, drop_last=True)
        exemplar_loader_iter = iter(exemplar_loader)

        old_model.eval()
        num_old_classes = old_model.module.fc3.out_features
    # writer = SummaryWriter(os.path.join(exp_dir, 'runs/log_ch'))
    for epoch_index in range(1, epoch + 1):

        dist_loss = 0.0
        sum_loss = 0
        sum_dist_loss = 0
        sum_cls_new_loss = 0
        sum_cls_old_loss = 0
        sum_cls_loss = 0

        model.train()
        old_model.eval()
        old_model.module.freeze_weight()

        for param_group in optimizer.param_groups:
            print('learning rate: {:.4f}'.format(param_group['lr']))

        for batch_idx, (pc, label) in enumerate(train_loader):

            optimizer.zero_grad()

            if args.aug:
                pc = pc.data.numpy()
                pc = provider.random_point_dropout(pc)
                pc[:, :, 0:3] = provider.random_scale_point_cloud(pc[:, :, 0:3])
                pc[:, :, 0:3] = provider.shift_point_cloud(pc[:, :, 0:3])
                pc = torch.Tensor(pc)

            pc, feat = pc[:, :, :3], pc[:, :, 3:]
            trot = None
            if args.rot == 'z':
                trot = RotateAxisAngle(angle=torch.rand(pc.shape[0]) * 360, axis="Z", degrees=True)
            elif args.rot == 'so3':
                trot = Rotate(R=random_rotations(pc.shape[0]))
            if trot is not None:
                pc = trot.transform_points(pc)
            x = torch.cat((pc, feat), 2).cuda().transpose(2, 1)
            targets = label.cuda()

            # use mixup for task-1
            if args.mixup:
                inputs, targets_a, targets_b, lam = mixup_data(x, targets, args.mixup_alpha)
                inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))

                logits = model(inputs)
                outputs = logits[:, -CLASS_NUM_IN_BATCH:]
                cls_loss_new = mixup_criterion(criterion_ce, outputs, targets_a, targets_b, lam)

            # use label smoothing for task-1
            elif args.label_smoothing:
                logits = model(x)
                cls_loss_new = criterion_ce_smooth(logits[:, -CLASS_NUM_IN_BATCH:], targets, args.smoothing_alpha)

            else:
                logits = model(x)
                # cls_loss_new = criterion_ce(logits[:, -CLASS_NUM_IN_BATCH:], F.one_hot(targets, num_classes=3))
                cls_loss_new = criterion_ce(logits[:, -CLASS_NUM_IN_BATCH:], targets)

            loss = args.w_cls * cls_loss_new
            sum_cls_new_loss += cls_loss_new.item()

            # use fixed lamda value or adaptive weighting
            if args.const_lamda:
                factor = args.lamda
            elif use_sd:
                factor = args.lamda_sd
            else:
                # factor = ((len(test_classes) / CLASS_NUM_IN_BATCH) ** (args.pow)) * args.lamda
                w = 1 / (1 + np.e ** (10 * (epoch_index / args.epochs - 0.5)))
                factor = w * args.lamda
            # while using self-distillation
            if len(test_classes) // CLASS_NUM_IN_BATCH == 1 and use_sd:
                if args.kd:
                    with torch.no_grad():
                        dist_target = old_model(x)
                    logits_dist = logits
                    T_sd = args.T_sd
                    dist_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(logits_dist / T_sd, dim=1),
                                               F.softmax(dist_target / T_sd, dim=1)) * (T_sd * T_sd)  # best model
                    sum_dist_loss += dist_loss.item()

                    loss += factor * args.w_kd * dist_loss

            # Distillation : task-2 onwards
            if len(test_classes) // CLASS_NUM_IN_BATCH > 1:

                if args.kd:
                    with torch.no_grad():
                        dist_target = old_model(x)
                    logits_dist = logits[:, :-CLASS_NUM_IN_BATCH]
                    T = args.T
                    dist_loss_new = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(logits_dist / T, dim=1),
                                                   F.softmax(dist_target / T, dim=1)) * (T * T)

                try:
                    batch_ex = next(exemplar_loader_iter)
                except:
                    exemplar_loader_iter = iter(exemplar_loader)
                    batch_ex = next(exemplar_loader_iter)

                # Classification loss: exemplar classes loss
                x_old, target_old = batch_ex
                x_old, target_old = x_old.cuda().transpose(2, 1), target_old.cuda()
                logits_old = model(x_old)

                # old_classes = len(test_classes) - CLASS_NUM_IN_BATCH
                cls_loss_old = criterion_ce(logits_old, target_old.to(torch.int64))

                loss += cls_loss_old
                sum_cls_old_loss += cls_loss_old.item()

                if args.kd:
                    # KD exemplar
                    with torch.no_grad():
                        dist_target_old = old_model(x_old)
                    logits_dist_old = logits_old[:, :-CLASS_NUM_IN_BATCH]
                    dist_loss_old = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(logits_dist_old / T, dim=1),
                                                   F.softmax(dist_target_old / T, dim=1)) * (T * T)  # best model

                    dist_loss = dist_loss_old + dist_loss_new
                    sum_dist_loss += dist_loss.item()
                    loss += factor * args.w_kd * dist_loss

            sum_loss += loss.item()
            # if args.tensorboard and batch_idx % 20 == 19:
                # writer.add_scalar('sum_loss',
                #                   sum_loss / (batch_idx + 1),
                #                   (epoch_index - 1) * len(train_loader) + batch_idx)
                # writer.add_scalar('sum_dist_loss',
                #                   sum_dist_loss / (batch_idx + 1),
                #                   (epoch_index - 1) * len(train_loader) + batch_idx)
                # writer.add_scalar('sum_cls_new_loss',
                #                   sum_cls_new_loss / (batch_idx + 1),
                #                   (epoch_index - 1) * len(train_loader) + batch_idx)
                # writer.add_scalar('sum_cls_old_loss',
                #                   sum_cls_old_loss / (batch_idx + 1),
                #                   (epoch_index - 1) * len(train_loader) + batch_idx)

            loss.backward()
            optimizer.step()
            step += 1

            if (batch_idx + 1) % checkPoint == 0 or (batch_idx + 1) == len(trainLoader):
                print(
                    '==>>> epoch: {}, batch index: {}, step: {}, train loss: {:.3f}, dist_loss: {:3f}, cls_new_loss: {:.3f}, cls_old_loss: {:.3f}'.
                    format(epoch_index, batch_idx + 1, step, sum_loss / (batch_idx + 1),
                           sum_dist_loss / (batch_idx + 1), sum_cls_new_loss / (batch_idx + 1),
                           sum_cls_old_loss / (batch_idx + 1)))

        scheduler.step()
        print(avg_acc)


def evaluate_net(model, train_classes, test_classes):
    model.eval()
    model = nn.DataParallel(model)

    if (args.datasets == 'modelnet40'):
        train_dataset = ModelNetDataLoader(root=args.data_root, args=args, class_cls=train_classes, split='test',
                                           process_data=args.process_data)
    elif(args.datasets == 'shapenetcorev2'):
        train_dataset = ShapeNetDataLoader(root='/home/huangyanhui/wxh/VNPCCL2.0/data',args=args, class_cls=train_classes,
                                           dataset_name=args.datasets, split='test')
    else:
        train_dataset = ScanObjectNNDataLoader(root='/dataset/ScanObjectNN/main_split_nobg', args=args,
                                               class_cls=train_classes, split='test')
    #
    # train_dataset = ModelNetDataLoader(root=args.data_root, args=args, class_cls=train_classes, split='test',
    #                                    process_data=args.process_data)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.num_workers)

    total = 0.0
    correct = 0.0
    compute_means = True
    for j, (pc_sets, labels) in enumerate(train_loader):
        points, feats = pc_sets[:, :, :3], pc_sets[:, :, 3:]
        trot = None
        args.rot = args.test_rot
        if args.rot == 'z':
            trot = RotateAxisAngle(angle=torch.rand(points.shape[0]) * 360, axis="Z", degrees=True)
        elif args.rot == 'so3':
            trot = Rotate(R=random_rotations(points.shape[0]))
        if trot is not None:
            points = trot.transform_points(points)
        points = torch.cat((points, feats), 2).cuda().transpose(2, 1)
        labels = labels.cuda()
        _, preds = torch.max(torch.softmax(model(points), dim=1), dim=1, keepdim=False)
        labels = [y.item() for y in labels]

        # if args.datasets == 'scanobjectnn':
        #     np.asarray(labels)
        # else:
        if t == 0:
            np.asarray(labels)
        else:
            labels = labels + np.array(args.start_classes + args.new_classes * (t - 1))
            np.asarray(labels)
        # np.asarray(labels)
        total += preds.size(0)
        correct += (preds.cpu().numpy() == labels).sum()

    # Train Accuracy
    print('correct: ', correct, 'total: ', total)
    print('Train Accuracy : %.2f ,' % (100.0 * correct / total))

    if (args.datasets == 'modelnet40'):
        test_dataset = ModelNetDataLoader(root=args.data_root, args=args, class_cls=test_classes, split='test',
                                           process_data=args.process_data)
    elif(args.datasets == 'shapenetcorev2'):
        test_dataset = ShapeNetDataLoader(root='/home/huangyanhui/wxh/VNPCCL2.0/data',args=args, class_cls=test_classes,
                                           dataset_name=args.datasets, split='test')
    else:
        test_dataset = ScanObjectNNDataLoader(root='/dataset/ScanObjectNN/main_split_nobg',args=args, class_cls=test_classes, split='test')
    # test_dataset = ModelNetDataLoader(root=args.data_root, args=args, class_cls=test_classes, split='test',
    #                                   process_data=args.process_data)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers)

    total = 0.0
    correct = 0.0
    for j, (pc_sets, labels) in enumerate(test_loader):
        points, feats = pc_sets[:, :, :3], pc_sets[:, :, 3:]
        args.rot = args.test_rot
        trot = None
        if args.rot == 'z':
            trot = RotateAxisAngle(angle=torch.rand(points.shape[0]) * 360, axis="Z", degrees=True)
        elif args.rot == 'so3':
            trot = Rotate(R=random_rotations(points.shape[0]))
        if trot is not None:
            points = trot.transform_points(points)
        points = torch.cat((points, feats), 2).cuda().transpose(2, 1)
        labels = labels.cuda()
        out = torch.softmax(model(points), dim=1)
        _, preds = torch.max(out, dim=1, keepdim=False)
        labels = [y.item() for y in labels]
        np.asarray(labels)
        total += preds.size(0)
        correct += (preds.cpu().numpy() == labels).sum()

    # Test Accuracy
    test_acc = 100.0 * correct / total
    print('correct: ', correct, 'total: ', total)
    print('Test Accuracy : %.2f' % test_acc)

    return test_acc


if __name__ == '__main__':
    args = parse_option()

    torch.cuda.manual_seed(args.seed)
    def log_string(str):
        logger.info(str)
        print(str)


    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./%s' % args.output_root)
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(parents=True, exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkPoints/')
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(parents=True, exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''parameter'''
    if args.datasets == 'scanobjectnn':
        TOTAL_CLASS_NUM = 15
    else:
        TOTAL_CLASS_NUM = 40 if args.datasets == 'modelnet40' else 54
    CLASS_NUM_IN_BATCH = args.start_classes
    TOTAL_CLASS_BATCH_NUM = TOTAL_CLASS_NUM // CLASS_NUM_IN_BATCH
    T = args.T

    K = args.K
    # exemplar_sets = []
    exemplar_means = []
    compute_means = True

    class_index = [i for i in range(0, TOTAL_CLASS_NUM)]
    np.random.seed(args.seed_num)
    np.random.shuffle(class_index)

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    net = MODEL.get_model(args, num_classes=CLASS_NUM_IN_BATCH, normal_channel=args.use_normals).cuda()


    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('number of trainable parameters: ', params)
    old_net = copy.deepcopy(net)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    old_net.cuda()

    if(args.datasets =='modelnet40'):
        cls_list = [0] + [a for a in range(args.start_classes, 40, args.new_classes)]
    elif(args.datasets == 'shapenetcorev2'):
        cls_list = [0] + [a for a in range(args.start_classes, 54, args.new_classes)]
    else:
        cls_list = [0] + [a for a in range(args.start_classes, 15, args.new_classes)]

    t = 0
    for i in cls_list:
        if i == args.start_classes:
            CLASS_NUM_IN_BATCH = args.new_classes

        print("==> Current Class: ", class_index[i:i + CLASS_NUM_IN_BATCH])

        if i == args.start_classes:
            net.change_output_dim(new_dim=i + CLASS_NUM_IN_BATCH)
        if i > args.start_classes:
            net.change_output_dim(new_dim=i + CLASS_NUM_IN_BATCH, second_iter=True)
        print('current net output dim:', net.get_output_dim())

        # net = nn.DataParallel(net)
        # old_net = nn.DataParallel(old_net)
        class_cls = class_index[i: i + CLASS_NUM_IN_BATCH]
        if(args.datasets == 'modelnet40'):
            train_dataset = ModelNetDataLoader(root=args.data_root, args=args, class_cls=class_cls, split='train',process_data=args.process_data)
        elif (args.datasets == 'shapenetcorev2'):
            train_dataset = ShapeNetDataLoader(root='/home/huangyanhui/wxh/VNPCCL2.0/data', args=args,
                                              class_cls=class_cls,
                                              dataset_name=args.datasets, split='train')
        else:
            train_dataset = ScanObjectNNDataLoader(root='/dataset/ScanObjectNN/main_split_nobg', args=args,
                                                  class_cls=class_cls, split='train')
        trainLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=args.num_workers, drop_last=True)

        train_classes = class_index[i: i + CLASS_NUM_IN_BATCH]
        test_classes = class_index[:i + CLASS_NUM_IN_BATCH]

        print(train_classes)
        print(test_classes)
        m = K // (i + CLASS_NUM_IN_BATCH)

        if i != 0:
            exemplar_sets = icarl_construct_exemplar_sets(m, exemplar_sets)

        # if args.datasets == 'scanobjectnn':
        #     for y in range(i, i + len(train_classes)):
        #         print("Constructing exemplar set for class-%d..." % (class_index[y]))
        #         pc_set = train_dataset.get_pc_class(class_index[y])
        #         exemplar_sets = icarl_construct_exemplar_set(net, pc_set, m)
        #         print("Done")
        # else:
        for y in range(i, i + CLASS_NUM_IN_BATCH):
            print("Constructing exemplar set for class-%d..." % (class_index[y]))
            pc_set = train_dataset.get_pc_class(y, t)
            exemplar_sets = icarl_construct_exemplar_set(net, pc_set, m)
            print("Done")

        # train and save model
        if args.resume and i == 0:
            net.load_state_dict(torch.load(args.resume_path))
            net.train()
        else:
            net.train()
            train(model=net, old_model=old_net, epoch=args.epochs, lr=args.lr, tempature=T, lamda=args.lamda,
                  train_loader=trainLoader, use_sd=False, checkPoint=20)

        if i != 0:
            weight_norm(net)

        old_net = copy.deepcopy(net)
        # old_net = nn.DataParallel(old_net)
        old_net.cuda()

        # Do self-distillation
        if i == 0 and not args.resume:
            for sd in range(args.num_sd):
                train(model=net, old_model=old_net, epoch=args.epochs_sd, lr=args.lr_sd, tempature=T,
                      lamda=args.lamda,
                      train_loader=trainLoader, use_sd=True, checkPoint=20)
                old_net = copy.deepcopy(net)
                # old_net = nn.DataParallel(old_net)
                old_net.cuda()

        if args.save:
            save_path = os.path.join(checkpoints_dir)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(net.state_dict(),
                       os.path.join(save_path, 'checkpoint_' + str(i + CLASS_NUM_IN_BATCH) + '.pth'))

        test_acc = evaluate_net(model=net, train_classes=class_index[i:i + CLASS_NUM_IN_BATCH],
                                test_classes=class_index[:i + CLASS_NUM_IN_BATCH])
        avg_acc.append(test_acc)
        t = t + 1


    print(avg_acc)
    print('Avg accuracy:', sum(avg_acc) / len(avg_acc))
    print('completed！！！')
