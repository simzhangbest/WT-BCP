from asyncore import write
import imp
import os
from sre_parse import SPECIAL_CHARS
import sys
from xml.etree.ElementInclude import default_loader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import random
import numpy as np
from medpy import metric
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
import pdb

from yaml import parse
from skimage.measure import label
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import losses, ramps, feature_memory, contrastive_losses, test_3d_patch_xnet_v2_3d
from dataloaders.dataset import *
from networks.net_factory import net_factory, BCP_xnet_v2_3d
from utils.BCP_utils import context_mask, mix_loss, parameter_sharing, update_ema_variables, haar_wavelet_transform_3d
from utils.losses import segmentation_loss
import pywt
import datetime
current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/sim_dataset/06_LA', help='Name of Dataset')
parser.add_argument('--exp', type=str,  default='BCP_xnet_v2_3d', help='exp_name')
parser.add_argument('--model', type=str, default='xnetv2_3d', help='model_name')
parser.add_argument('--pre_max_iteration', type=int,  default=4000, help='maximum pre-train iteration to train')
parser.add_argument('--self_max_iteration', type=int,  default=15000, help='maximum self-train iteration to train')
parser.add_argument('--max_samples', type=int,  default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=8, help='trained samples')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='10.0', help='magnitude')
# -- setting of BCP
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--mask_ratio', type=float, default=2/3, help='ratio of mask/image')
# -- setting of mixup
parser.add_argument('--u_alpha', type=float, default=2.0, help='unlabeled image ratio of mixuped image')
parser.add_argument('--loss_weight', type=float, default=0.5, help='loss weight of unimage term')
args = parser.parse_args()

def get_cut_mask(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks

def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max()!= 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)

    # Convert list of NumPy arrays to a single NumPy array
    batch_array = np.array(batch_list)

    # Convert NumPy array to PyTorch tensor and move to GPU
    return torch.Tensor(batch_array).cuda()

def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

train_data_path = args.root_path

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
pre_max_iterations = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr = args.base_lr
CE = nn.CrossEntropyLoss(reduction='none')

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

patch_size = (112, 112, 80)
num_classes = 2
criterion = segmentation_loss('dice', False).cuda()

def pre_train(args, snapshot_path):
    
    
    # model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    model = BCP_xnet_v2_3d(1, class_num=num_classes)
    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    
    db_test = LAHeart(base_dir=train_data_path,
                       split='test',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)
    sub_bs = int(args.labeled_bs/2)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    testloader = DataLoader(db_test, 20, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    DICE = losses.mask_DiceLoss(nclass=2)

    model.train()
    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = pre_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        
        unsup_weight = args.u_weight * (epoch_num + 1) / max_epoch
        
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'][:args.labeled_bs], sampled_batch['label'][:args.labeled_bs]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:]
            with torch.no_grad():
                img_mask, loss_mask = context_mask(img_a, args.mask_ratio)

            """Mix Input"""
            volume_batch = img_a * img_mask + img_b * (1 - img_mask)
            label_batch = lab_a * img_mask + lab_b * (1 - img_mask)
            
            # volume 处理   [2, 1, 112, 112, 80]

            volumn_low, volumn_high = haar_wavelet_transform_3d(volume_batch)
            outputs_m, outputs_low, outputs_high = model(volume_batch, volumn_low, volumn_high)
            
            # 一致性损失
            
            max_train_unsup_m = torch.max(outputs_m, dim=1)[1].long()
            max_train_unsup_low = torch.max(outputs_low, dim=1)[1].long()
            max_train_unsup_high = torch.max(outputs_high, dim=1)[1].long()
            
            loss_train_unsup = criterion(outputs_m, max_train_unsup_low) + \
                               criterion(outputs_low, max_train_unsup_m) + \
                               criterion(outputs_m, max_train_unsup_high) + \
                               criterion(outputs_high, max_train_unsup_m)
            loss_train_unsup = (loss_train_unsup / 4 )* unsup_weight
            # loss_train_unsup.backward(retain_graph=True)
            
            # torch.cuda.empty_cache()
            
            # 监督损失
        
            
            loss_ce_m = F.cross_entropy(outputs_m, label_batch)
            loss_dice_m = DICE(outputs_m, label_batch)
            
            
            loss_ce_low = F.cross_entropy(outputs_low, label_batch)
            loss_dice_low = DICE(outputs_low, label_batch)
            
            loss_ce_high = F.cross_entropy(outputs_low, label_batch)
            loss_dice_high = DICE(outputs_low, label_batch)
            
            loss_ce = loss_ce_m + loss_ce_low + loss_ce_high
            loss_dice = loss_dice_m + loss_dice_low + loss_dice_high
            loss = (loss_ce + loss_dice) / 6
            
            loss_all = loss  # 不放 consis loss 就只算自己的监督
            iter_num += 1
            writer.add_scalar('pre/loss_dice', loss_dice, iter_num)
            writer.add_scalar('pre/loss_ce', loss_ce, iter_num)
            writer.add_scalar('pre/loss_all', loss, iter_num)
            # writer.add_scalar('pre/loss_train_unsup', loss_train_unsup, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss: %03f, loss_dice: %03f, loss_ce: %03f, loss_all: %03f'%(iter_num, loss, loss_dice, loss_ce, loss_all))

            if iter_num % 100 == 0:
                model.eval()
                dice_sample = test_3d_patch_xnet_v2_3d.var_all_case_LA(testloader, model, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(snapshot_path,  'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)

                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('4_Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num >= pre_max_iterations:
                break

        if iter_num >= pre_max_iterations:
            iterator.close()
            break
    writer.close()


def self_train(args, pre_snapshot_path, self_snapshot_path):
    
    model = BCP_xnet_v2_3d(1,2)
    ema_model = BCP_xnet_v2_3d(1,2,True)

    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    db_test =  LAHeart(base_dir=train_data_path,
                       split='test',
                       transform = transforms.Compose([
                        #   RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)
    sub_bs = int(args.labeled_bs/2)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    testloader = DataLoader(db_test, 20, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    pretrained_model = os.path.join(pre_snapshot_path, f'{args.model}_best_model.pth')
    load_net(model, pretrained_model)
    load_net(ema_model, pretrained_model)
    
    model.train()
    ema_model.train()
    writer = SummaryWriter(self_snapshot_path+'/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = self_max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch in iterator:
        
        unsup_weight = args.u_weight * (epoch + 1) / max_epoch
        
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:args.labeled_bs]
            uimg_a, uimg_b = volume_batch[args.labeled_bs:args.labeled_bs+sub_bs], volume_batch[args.labeled_bs+sub_bs:]
            with torch.no_grad():
                
                # simzhang
                uimg_a_low, uimg_a_high = haar_wavelet_transform_3d(uimg_a)
                uimg_b_low, uimg_b_high = haar_wavelet_transform_3d(uimg_b)
                
                
                ua_out_m, ua_out_low, ua_out_high = ema_model(uimg_a, uimg_a_low, uimg_a_high)  # 获得伪标签
                ub_out_m, ub_out_low, ub_out_high = ema_model(uimg_b, uimg_b_low, uimg_b_high)
                
                # ua_m = get_cut_mask(ua_out_m, nms=1) #
                # ua_low = get_cut_mask(ua_out_low, nms=1)
                # ua_high = get_cut_mask(ua_out_high, nms=1)
                # loss_train_unsup_a = criterion(ua_out_m, ua_low) + criterion(ua_out_m, ua_high) + criterion(ua_out_low, ua_m) + criterion(ua_out_high, ua_m)
                
                
                plab_a = get_cut_mask(ua_out_m, nms=1)
                plab_b = get_cut_mask(ub_out_m, nms=1)
                img_mask, loss_mask = context_mask(img_a, args.mask_ratio)
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            mixl_img = img_a * img_mask + uimg_a * (1 - img_mask)
            mixu_img = uimg_b * img_mask + img_b * (1 - img_mask)
            mixl_lab = lab_a * img_mask + plab_a * (1 - img_mask)
            mixu_lab = plab_b * img_mask + lab_b * (1 - img_mask)
            
            # simzhang wt here
            net_input_unl = mixu_img
            net_input_l = mixl_img
            net_input_unl_low, net_input_unl_high = haar_wavelet_transform_3d(net_input_unl)
            net_input_l_low, net_input_l_high = haar_wavelet_transform_3d(net_input_l)
            
            
            out_unl_m, out_unl_low, out_unl_high = model(net_input_unl, net_input_unl_low, net_input_unl_high)
            out_l_m, out_l_low, out_l_high = model(net_input_l, net_input_l_low, net_input_l_high)
                        
            out_unl = out_unl_m
            out_l   = out_l_m
            
            # unl_m = get_cut_mask(out_unl_m, nms=1)
            # unl_low = get_cut_mask(out_unl_low, nms=1)
            # unl_high = get_cut_mask(out_unl_high, nms=1)
            unl_m = torch.max(out_unl_m, dim=1)[1].long()
            unl_low = torch.max(out_unl_low, dim=1)[1].long()
            unl_high = torch.max(out_unl_high, dim=1)[1].long()
            
            unl_loss = criterion(out_unl_low, unl_m) + criterion(out_unl_high, unl_m) \
                + criterion(out_unl_m, unl_low) + criterion(out_unl_m, unl_high)

            loss_consis = unl_loss * unsup_weight # 只有unlabel 进行一致性的loss
            loss_consis.backward(retain_graph=True)
            torch.cuda.empty_cache()
            
            l_m = mix_loss(out_l_m, lab_a, plab_a, loss_mask, u_weight=args.u_weight)
            l_low = mix_loss(out_l_low, lab_a, plab_a, loss_mask, u_weight=args.u_weight)
            l_high = mix_loss(out_l_high, lab_a, plab_a, loss_mask, u_weight=args.u_weight)
            
            unl_m = mix_loss(out_unl_m, plab_b, lab_b, loss_mask, u_weight=args.u_weight, unlab=True)
            unl_low = mix_loss(out_unl_low, plab_b, lab_b, loss_mask, u_weight=args.u_weight, unlab=True)
            unl_high = mix_loss(out_unl_high, plab_b, lab_b, loss_mask, u_weight=args.u_weight, unlab=True)
            


            loss_sup = (unl_m +  unl_low + unl_high + l_m + l_low + l_high ) / 6
            
            # loss_all = loss_consis + loss_sup
            
            loss_all =  loss_sup

            iter_num += 1
            writer.add_scalar('Self/consistency', consistency_weight, iter_num)
            writer.add_scalar('Self/loss_sup', loss_sup, iter_num)

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            # logging.info('iteration %d : loss_sup: %03f , loss_all: %03f'%(iter_num, loss_sup, loss_all ))
            
            logging.info('iteration %d : loss_sup: %03f , loss_consis: %03f'%(iter_num, loss_sup, loss_consis ))

            update_ema_variables(model, ema_model, 0.99)

             # change lr  2500 --> 250  simzhang test
            if iter_num % 250 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 250)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num % 100 == 0:
                model.eval()
                # dice_sample = test_3d_patch.var_all_case_LA(model, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4)
                dice_sample = test_3d_patch_xnet_v2_3d.var_all_case_LA(testloader, model, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4)
                logging.info('simzhang dice sample: %03f '%(dice_sample))
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(self_snapshot_path,  'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(self_snapshot_path,'{}_best_model.pth'.format(args.model))
                    # save_net_opt(model, optimizer, save_mode_path)
                    # save_net_opt(model, optimizer, save_best_path)
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('4_Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)
                model.train()
            
            if iter_num % 100 == 1:
                ins_width = 2
                B,C,H,W,D = out_l.size()
                snapshot_img = torch.zeros(size = (D, 3, 3*H + 3 * ins_width, W + ins_width), dtype = torch.float32)

                snapshot_img[:,:, H:H+ ins_width,:] = 1
                snapshot_img[:,:, 2*H + ins_width:2*H + 2*ins_width,:] = 1
                snapshot_img[:,:, 3*H + 2*ins_width:3*H + 3*ins_width,:] = 1
                snapshot_img[:,:, :,W:W+ins_width] = 1

                outputs_l_soft = F.softmax(out_l, dim=1)
                seg_out = outputs_l_soft[0,1,...].permute(2,0,1) # y
                target =  mixl_lab[0,...].permute(2,0,1)
                train_img = mixl_img[0,0,...].permute(2,0,1)

                snapshot_img[:, 0,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 1,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 2,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))

                snapshot_img[:, 0, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 1, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 2, H+ ins_width:2*H+ ins_width,:W] = target

                snapshot_img[:, 0, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                snapshot_img[:, 1, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                snapshot_img[:, 2, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                
                writer.add_images('Epoch_%d_Iter_%d_labeled'% (epoch, iter_num), snapshot_img)

                outputs_u_soft = F.softmax(out_unl, dim=1)
                seg_out = outputs_u_soft[0,1,...].permute(2,0,1) # y
                target =  mixu_lab[0,...].permute(2,0,1)
                train_img = mixu_img[0,0,...].permute(2,0,1)

                snapshot_img[:, 0,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 1,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 2,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))

                snapshot_img[:, 0, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 1, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 2, H+ ins_width:2*H+ ins_width,:W] = target

                snapshot_img[:, 0, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                snapshot_img[:, 1, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                snapshot_img[:, 2, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out

                writer.add_images('Epoch_%d_Iter_%d_unlabel'% (epoch, iter_num), snapshot_img)

            if iter_num >= self_max_iterations:
                break

        if iter_num >= self_max_iterations:
            iterator.close()
            break
    writer.close()


import datetime
if __name__ == "__main__":
    ## make logger file
    current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    # pre_snapshot_path = "./sim_model/BCP/LA_{}_{}_labeled/{}/pre_train".format(args.exp, args.labelnum, current_time)
    # self_snapshot_path = "./sim_model/BCP/LA_{}_{}_labeled/{}/self_train".format(args.exp, args.labelnum, current_time)
    
    pre_snapshot_path = "/mnt/zmy/code/BCP/sim_model/BCP/LA_BCP_xnet_v2_3d_8_labeled/20241021234425/pre_train"
    self_snapshot_path = "/mnt/zmy/code/BCP/sim_model/BCP/LA_BCP_xnet_v2_3d_8_labeled/20241021234425/self_train"
    print("Starting BCP training.")
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
    shutil.copy('/mnt/zmy/code/BCP/code/LA_BCP_train_xnet_v2_3d.py', self_snapshot_path)
    # -- Pre-Training
    # logging.basicConfig(filename=pre_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info(str(args))
    # pre_train(args, pre_snapshot_path)
    
    
    # -- Self-training
    logging.basicConfig(filename=self_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)

    
