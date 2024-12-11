from asyncore import write
from audioop import avg
from cgi import test
import imp
from multiprocessing import reduction
from turtle import pd
from unittest import loader, result

from yaml import load
import torch
import os
import pdb
import torch.nn as nn

from tqdm import tqdm as tqdm_load
from pancreas_utils import *
from test_util_xnet_v2_3d import test_calculate_metric
from losses import DiceLoss, softmax_mse_loss, mix_loss, segmentation_loss
from dataloaders import get_ema_model_and_dataloader, haar_wavelet_transform_3d



"""Global Variables"""
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
seed_test = 2020
seed_reproducer(seed = seed_test)
import datetime
current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
data_root, split_name = '/mnt/zmy/code/BCP/dataset_mnt/07_Pancreas/data', 'pancreas'
result_dir = '/mnt/zmy/code/BCP/code/pancreas/result/xnetv2/' + current_time
# result_dir = '/mnt/zmy/code/BCP/code/pancreas/result/xnetv2/' + '2024-10-22-17-45-35'
mkdir(result_dir)
batch_size, lr = 2, 1e-3
pretraining_epochs, self_training_epochs = 200, 200
pretrain_save_step, st_save_step = 5, 5  # 20, 20, 5
alpha, consistency, consistency_rampup = 0.99, 0.1, 40
label_percent = 20
u_weight = 1.5
connect_mode = 2
try_second = 1
sec_t = 0.5
self_train_name = 'self_train'

sub_batch = int(batch_size/2)
consistency_criterion = softmax_mse_loss
CE = nn.CrossEntropyLoss()
CE_r = nn.CrossEntropyLoss(reduction='none')
DICE = DiceLoss(nclass=2)
patch_size = 64

logger = None
criterion = segmentation_loss('dice', False).cuda()

def pretrain(net1, optimizer, lab_loader_a, labe_loader_b, test_loader):
    """pretrain image- & patch-aware network"""

    """Create Path"""
    save_path = Path(result_dir) / 'pretrain'
    save_path.mkdir(exist_ok=True)

    """Create logger and measures"""
    global logger
    logger, writer = cutmix_config_log(save_path, tensorboard=True)
    logger.info("cutmix Pretrain, patch_size: {}, save path: {}".format(patch_size, str(save_path)))

    max_dice = 0
    measures = CutPreMeasures(writer, logger)
    for epoch in tqdm_load(range(1, pretraining_epochs + 1), ncols=70):
        
        # TODO 超参数，要调整
        unsup_weight = u_weight * (epoch + 1) / pretraining_epochs
        
        measures.reset()
        """Testing"""
        if epoch % pretrain_save_step == 0:
            avg_metric1 = test_calculate_metric(net1, test_loader.dataset)  # sim bug fixed
            logger.info('average metric is : {}'.format(avg_metric1))
            val_dice = avg_metric1[0][0]

            if val_dice > max_dice:
                val_dice = round(val_dice, 4)
                save_net_opt(net1, optimizer, save_path / f'dice_{val_dice}.pth', epoch)
                save_net_opt(net1, optimizer, save_path / f'best_ema_{label_percent}_pre.pth', epoch)
                max_dice = val_dice
            
            writer.add_scalar('test_dice', val_dice, epoch)
            logger.info('Evaluation: val_dice: %.4f, val_maxdice: %.4f '%(val_dice, max_dice))
            # save_net_opt(net1, optimizer, save_path / ('%d.pth' % epoch), epoch)
        
        """Training"""
        net1.train()
        for step, ((img_a, lab_a), (img_b, lab_b)) in enumerate(zip(lab_loader_a, lab_loader_b)):
            img_a, img_b, lab_a, lab_b  = img_a.cuda(), img_b.cuda(), lab_a.cuda(), lab_b.cuda()
            img_mask, loss_mask = generate_mask(img_a, patch_size)   

            # img = img_a * img_mask + img_b * (1 - img_mask)
            # lab = lab_a * img_mask + lab_b * (1 - img_mask)
            
            volume_batch = img_a * img_mask + img_b * (1 - img_mask)
            label_batch = lab_a * img_mask + lab_b * (1 - img_mask)

            # TODO: sim work here
            
            volumn_low, volumn_high = haar_wavelet_transform_3d(volume_batch)
            
            outputs_m, outputs_low, outputs_high = net1(volume_batch, volumn_low, volumn_high)
                        
            # 一致性损失
            max_train_unsup_m = torch.max(outputs_m, dim=1)[1].long()
            max_train_unsup_low = torch.max(outputs_low, dim=1)[1].long()
            max_train_unsup_high = torch.max(outputs_high, dim=1)[1].long()
            
            loss_train_unsup = criterion(outputs_m, max_train_unsup_low) + \
                               criterion(outputs_low, max_train_unsup_m) + \
                               criterion(outputs_m, max_train_unsup_high) + \
                               criterion(outputs_high, max_train_unsup_m)
            loss_train_unsup = (loss_train_unsup / 4 )* unsup_weight
            
            # 监督损失
            loss_ce_m = F.cross_entropy(outputs_m, label_batch)
            loss_dice_m = DICE(outputs_m, label_batch)
            
            loss_ce_low = F.cross_entropy(outputs_low, label_batch)
            loss_dice_low = DICE(outputs_low, label_batch)
            
            loss_ce_high = F.cross_entropy(outputs_low, label_batch)
            loss_dice_high = DICE(outputs_low, label_batch)
            
            loss_ce = loss_ce_m + loss_ce_low + loss_ce_high
            loss_dice = loss_dice_m + loss_dice_low + loss_dice_high
            loss_sup = (loss_ce + loss_dice) / 6

            loss = loss_sup + loss_train_unsup
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            measures.update(outputs_m, label_batch, loss_ce, loss_dice, loss)
            measures.log(epoch, epoch * len(lab_loader_a) + step)
            
        writer.flush()
    return max_dice

def ema_cutmix(net, ema_net, optimizer, lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, test_loader):
    """Create Path"""
    save_path = Path(result_dir) / self_train_name
    save_path.mkdir(exist_ok=True)

    """Create logger and measures"""
    global logger 
    logger, writer = config_log(save_path, tensorboard=True)
    logger.info("EMA_training, save_path: {}".format(str(save_path)))
    measures = CutmixFTMeasures(writer, logger)

    """Load Model"""
    pretrained_path = Path(result_dir) / 'pretrain'
    load_net_opt(net, optimizer, pretrained_path / f'best_ema_{label_percent}_pre.pth')
    load_net_opt(ema_net, optimizer, pretrained_path / f'best_ema_{label_percent}_pre.pth')
    logger.info('Loaded from {}'.format(pretrained_path))

    max_dice = 0
    max_list = None
    for epoch in tqdm_load(range(1, self_training_epochs+1)):
        
        unsup_weight = u_weight * (epoch + 1) / pretraining_epochs # TODO hyper params
        
        measures.reset()
        logger.info('')

        """Testing"""
        if epoch % st_save_step == 0:
            avg_metric = test_calculate_metric(net, test_loader.dataset)
            logger.info('average metric is : {}'.format(avg_metric))
            val_dice = avg_metric[0]  #  Dice:{}, Jd:{}, ASD:{}, HD:{}
            writer.add_scalar('val_dice', val_dice[0], epoch)
            writer.add_scalar('val_jd', val_dice[1], epoch)
            writer.add_scalar('val_asd', val_dice[2], epoch)
            writer.add_scalar('val_hd', val_dice[3], epoch)

            """Save Model"""
            if val_dice[0] > max_dice:
                dice = round(val_dice[0], 4)
                save_net(net, str(save_path / f'dice_{dice}_self.pth'))
                save_net(net, str(save_path / f'best_ema_{label_percent}_self.pth'))
                max_dice = dice
                max_list = avg_metric

            logger.info('Evaluation: val_dice: %.4f, val_maxdice: %.4f' % (dice, max_dice))

        """Training"""
        net.train()
        ema_net.train()
        for step, ((img_a, lab_a), (img_b, lab_b), (unimg_a, unlab_a), (unimg_b, unlab_b)) in enumerate(zip(lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b)):
            img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b = to_cuda([img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b])
            """Generate Pseudo Label"""
            with torch.no_grad():

                # simzhang
                uimg_a_low, uimg_a_high = haar_wavelet_transform_3d(unimg_a)
                uimg_b_low, uimg_b_high = haar_wavelet_transform_3d(unimg_b)
                
                
                ua_out_m, ua_out_low, ua_out_high = ema_net(unimg_a, uimg_a_low, uimg_a_high)  # 获得伪标签
                ub_out_m, ub_out_low, ub_out_high = ema_net(unimg_b, uimg_b_low, uimg_b_high)
                
                plab_a = get_cut_mask(ua_out_m, nms=1)
                plab_b = get_cut_mask(ub_out_m, nms=1)
                plab_a, plab_b = plab_a.long(), plab_b.long()
                img_mask, loss_mask = generate_mask(img_a, patch_size)
                
            mixl_img = img_a * img_mask + unimg_a * (1 - img_mask)
            mixu_img = unimg_b * img_mask + img_b * (1 - img_mask)
            mixl_lab = lab_a * img_mask + plab_a * (1 - img_mask)
            mixu_lab = plab_b * img_mask + lab_b * (1 - img_mask)
            
            net_input_unl = mixu_img
            net_input_l = mixl_img
            net_input_unl_low, net_input_unl_high = haar_wavelet_transform_3d(net_input_unl)
            net_input_l_low, net_input_l_high = haar_wavelet_transform_3d(net_input_l)
            
            out_unl_m, out_unl_low, out_unl_high = net(net_input_unl, net_input_unl_low, net_input_unl_high)
            out_l_m, out_l_low, out_l_high = net(net_input_l, net_input_l_low, net_input_l_high)
            
            out_unl = out_unl_m
            out_l   = out_l_m
            
            unl_m = torch.max(out_unl_m, dim=1)[1].long()
            unl_low = torch.max(out_unl_low, dim=1)[1].long()
            unl_high = torch.max(out_unl_high, dim=1)[1].long()      
            
            unl_loss = criterion(out_unl_low, unl_m) + criterion(out_unl_high, unl_m) \
            + criterion(out_unl_m, unl_low) + criterion(out_unl_m, unl_high)
            
            loss_consis = unl_loss * unsup_weight # 只有unlabel 进行一致性的loss
            # loss_consis.backward(retain_graph=True)
            # torch.cuda.empty_cache()
            
            l_m = mix_loss(out_l_m, lab_a, plab_a, loss_mask, u_weight= u_weight)
            l_low = mix_loss(out_l_low, lab_a, plab_a, loss_mask, u_weight= u_weight)
            l_high = mix_loss(out_l_high, lab_a, plab_a, loss_mask, u_weight= u_weight)
            
            unl_m = mix_loss(out_unl_m, plab_b, lab_b, loss_mask, u_weight= u_weight, unlab=True)
            unl_low = mix_loss(out_unl_low, plab_b, lab_b, loss_mask, u_weight= u_weight, unlab=True)
            unl_high = mix_loss(out_unl_high, plab_b, lab_b, loss_mask, u_weight= u_weight, unlab=True)
            
            
            loss_sup = (unl_m +  unl_low + unl_high + l_m + l_low + l_high ) / 6
            
            loss = loss_consis + loss_sup
                
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_ema_variables(net, ema_net, alpha)

            measures.update(loss_sup, loss_consis, loss)  
            measures.log(epoch, epoch*len(lab_loader_a) + step)

        if epoch ==  self_training_epochs:
            save_net(net, str(save_path / f'best_ema_{label_percent}_self_latest.pth'))
        writer.flush()
    return max_dice, max_list

def test_model(net, test_loader):
    load_path = Path(result_dir) / self_train_name
    load_net(net, load_path / 'best_2.pth')
    print('Successful Loaded')
    avg_metric, m_list = test_calculate_metric(net, test_loader.dataset, s_xy=16, s_z=4)
    test_dice = avg_metric[0] # 要索引
    return avg_metric, m_list


if __name__ == '__main__':
    try:
        net, ema_net, optimizer, lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, test_loader = get_ema_model_and_dataloader(data_root, split_name, batch_size, lr, labelp=label_percent)
        pretrain(net, optimizer, lab_loader_a, lab_loader_b, test_loader)
        ema_cutmix(net, ema_net, optimizer, lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, test_loader)
        # avg_metric, m_list = test_model(net, test_loader)
    except Exception as e:
        logger.exception("BUG FOUNDED ! ! !")


