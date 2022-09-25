"""
the general training framework
"""

from __future__ import print_function
from genericpath import exists

import os
import argparse
import socket
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed, Conv1x1Reg
from models.util import Connector, Translator, Paraphraser, SelfA

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.tiny_imagenet import get_tinyimagenet_dataloaders_sample, get_tinyimagenet_dataloader

from helper.util import adjust_learning_rate

from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss, SPPloss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss, SemCKDLoss
from crd.criterion import CRDLoss

from helper.loops import train_distill as train, validate
from helper.pretrain import init


def parse_option():
    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # optimization  
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='tiny-imagenet', choices=['cifar100', 'tiny-imagenet'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet20',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', ])
    parser.add_argument('--path_t', type=str, default='./save/tiny-imagenet-models/resnet56_vanilla/ckpt_epoch_240.pth',
                        help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='spp', choices=['kd', 'hint', 'attention', 'similarity',
                                                                       'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                       'rkd', 'pkt', 'abound', 'factor', 'nst', 'spp',
                                                                       'semckd'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=0, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=1, help='weight balance for other losses')
    parser.add_argument('-f', '--factor', type=int, default=2, help='factor size of SimKD')
    parser.add_argument('-s', '--soft', type=float, default=1.0, help='attention scale of SemCKD')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # spp distillation
    parser.add_argument('--max_w', type=float, default=0.9, help='weight balance for spp max dist')
    parser.add_argument('--min_w', type=float, default=2.5, help='weight balance for spp min dist')
    parser.add_argument('--only_last', type=bool, default=True,
                        help='only distill the last feature map or not')  # this parameter is True now
    parser.add_argument('--spp_layer', type=int, default=4, help='number of spp layer')
    parser.add_argument('--topk', type=int, default=5, help='topk parameter in SPP')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # hint layer
    parser.add_argument('--hint_layer', default=3, type=int, choices=[0, 1, 2, 3, 4], help="index of last hint layer")


    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    opt.model_path = 'save/student_model'
    opt.tb_path = 'save/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    # opt.model_name = 'S:{}_T:{}_{}_{}_r:{}_a:{}_b:{}_{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.distill,
    #                                                             opt.gamma, opt.alpha, opt.beta, opt.trial)

    opt.model_name = 'S_{}_T_{}_{}_{}_r_{}_a_{}_b_{}_{}_max_w_{}_min_w_{}_only_last_{}_spp_layer_{}_topk_{}'.format(
        opt.model_s, opt.model_t, opt.dataset, opt.distill,
        opt.gamma, opt.alpha, opt.beta, opt.trial, opt.max_w, opt.min_w, opt.only_last, opt.spp_layer, opt.topk)
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)

    print('opt.tb_path', opt.tb_path)
    print('opt.model_name', opt.model_name)
    print('opt.tb_folder', opt.tb_folder)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    return model_path.split('/')[-2].split('_vanilla')[0]


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model


def main():
    best_acc = 0
    opt = parse_option()
    opt.rank = 0
    opt.world_size = 1

    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    n_cls = {
        'cifar100': 100,
        'tiny-imagenet': 200,
    }.get(opt.dataset, None)

    # dataloader
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        else:
            train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                num_workers=opt.num_workers)
    elif opt.dataset == 'tiny-imagenet':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_tinyimagenet_dataloaders_sample(batch_size=opt.batch_size,
                                                                                   num_workers=opt.num_workers,
                                                                                   k=opt.nce_k,
                                                                                   mode=opt.mode)
        else:
            train_loader, val_loader = get_tinyimagenet_dataloader(batch_size=opt.batch_size,
                                                                   num_workers=opt.num_workers)
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model_t = load_teacher(opt.path_t, n_cls)
    model_s = model_dict[opt.model_s](num_classes=n_cls)

    init_epoch = 1
    exists_model_s_path = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
    if os.path.exists(exists_model_s_path):
        exists_model_s = torch.load(exists_model_s_path)
        model_s.load_state_dict(exists_model_s['model'])
        init_epoch = exists_model_s['epoch'] + 1
        best_acc = exists_model_s['best_acc']
        print("Load existing student model, best acc is %f" % best_acc)

    if opt.dataset == 'cifar100':
        data = torch.randn(2, 3, 32, 32)
    elif opt.dataset == 'tiny-imagenet':
        data = torch.torch.randn(2, 3, 64, 64)
    else:
        raise NotImplementedError(opt.dataset)
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)
    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'spp':
        criterion_kd = SPPloss(opt.max_w, opt.min_w, opt.only_last, opt.spp_layer, opt.topk)
        regress_s = Conv1x1Reg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'nst':
        criterion_kd = NSTLoss()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif opt.distill == 'pkt':
        criterion_kd = PKT()
    elif opt.distill == 'kdsvd':
        criterion_kd = KDSVD()
    elif opt.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif opt.distill == 'abound':
        s_shapes = [f.shape for f in feat_s[1:-1]]
        t_shapes = [f.shape for f in feat_t[1:-1]]
        connector = Connector(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(connector)
        init_trainable_list.append(model_s.get_feat_modules())
        criterion_kd = ABLoss(len(feat_s[1:-1]))
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification
        module_list.append(connector)
    elif opt.distill == 'factor':
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_list, criterion_init, train_loader, logger, opt)
        # classification
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        trainable_list.append(translator)
    elif opt.distill == 'fsp':
        s_shapes = [s.shape for s in feat_s[:-1]]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(model_s.get_feat_modules())
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification training
    elif opt.distill == 'semckd':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = SemCKDLoss()
        self_attention = SelfA(opt.batch_size, s_n, t_n, opt.soft)
        module_list.append(self_attention)
        trainable_list.append(self_attention)
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)  # other knowledge distillation loss

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # validate teacher accuracy
    # teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    # print('teacher accuracy: ', teacher_acc)

    # routine
    for epoch in range(init_epoch, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss, loss_cls, loss_div, loss_kd = train(epoch, train_loader, module_list, criterion_list,
                                                                   optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)
        logger.log_value('loss_cls', loss_cls, epoch)
        logger.log_value('loss_div', loss_div, epoch)
        logger.log_value('loss_kd', loss_kd, epoch)
        test_acc, tect_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('test_acc_top5', tect_acc_top5, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    print('best accuracy:', best_acc)

    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
