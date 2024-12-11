from locale import normalize
from multiprocessing import reduction
import pdb
from turtle import pd
import numpy as np
import torch.nn as nn
import torch
import random
from utils.losses import mask_DiceLoss
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

import torchio as tio
from torchio import transforms as T
import pywt
from torch.nn.functional import interpolate

DICE = mask_DiceLoss(nclass=2)
CE = nn.CrossEntropyLoss(reduction='none')

patch_size = (112, 112, 80)
normalize_t = tio.ZNormalization.mean
 # L H augmentation
L_H_aug = T.Compose([T.Resize(patch_size), T.ZNormalization(masking_method=normalize_t)])


def haar_wavelet_transform(net_input):
    batch_size, channels, height, width = net_input.shape
    out_low = torch.zeros((batch_size, channels, height // 2, width // 2))
    out_high = torch.zeros((batch_size, channels, height // 2, width // 2))
    for i in range(batch_size):
        img = net_input[i, 0, :, :].cpu().numpy()
        # 对图像的最后两个维度进行 Haar 小波变换
        LL, (LH, HL, HH) = pywt.dwt2(img, 'haar')
        low = LL
        low = (low - low.min()) / (low.max() - low.min())
        high = HH + HL + LH
        high = (high - high.min()) / (high.max() - high.min())
        # 将变换后的结果组合成一个新的张量
        out_low[i, 0, :, :] = torch.from_numpy(low[:height // 2, :width // 2])
        out_high[i, 0, :, :] = torch.from_numpy(high[:height // 2, :width // 2])

    # 上采样 low 和 high 到和输入 net_input 同尺寸
    upsampled_low = interpolate(out_low, size=(height, width), mode='bilinear', align_corners=False)
    upsampled_high = interpolate(out_high, size=(height, width), mode='bilinear', align_corners=False)

    return upsampled_low.cuda(), upsampled_high.cuda()


def haar_wavelet_transform_fusion(net_input):
    batch_size, channels, height, width = net_input.shape
    out_low = torch.zeros((batch_size, channels, height // 2, width // 2))
    out_high = torch.zeros((batch_size, channels, height // 2, width // 2))
    for i in range(batch_size):
        img = net_input[i, 0, :, :].cpu().numpy()
        # 对图像的最后两个维度进行 Haar 小波变换
        LL, (LH, HL, HH) = pywt.dwt2(img, 'haar')
        low = LL
        low = (low - low.min()) / (low.max() - low.min())
        high = HH + HL + LH
        high = (high - high.min()) / (high.max() - high.min())
        # 将变换后的结果组合成一个新的张量
        out_low[i, 0, :, :] = torch.from_numpy(low[:height // 2, :width // 2])
        out_high[i, 0, :, :] = torch.from_numpy(high[:height // 2, :width // 2])

    # 上采样 low 和 high 到和输入 net_input 同尺寸
    upsampled_low = interpolate(out_low, size=(height, width), mode='bilinear', align_corners=False)
    upsampled_high = interpolate(out_high, size=(height, width), mode='bilinear', align_corners=False)
    
    upsampled_low = upsampled_low.cuda()
    upsampled_high = upsampled_high.cuda()
    
    low = upsampled_low + net_input  # fusion
    high = upsampled_high + net_input
    low = (low - low.min()) / (low.max() - low.min())
    high = (high - high.min()) / (high.max() - high.min())

    return low, high



# alpha=[0.2, 0.2], beta=[0.65, 0.65]
# alpha=[0.2, 0.2], beta=[0.8, 0.8]  ab  对照日志
# alpha=[0.1, 0.1], beta=[0.9, 0.9]  ab2
def haar_wavelet_transform_tri(net_input, alpha=[0.1, 0.1], beta=[0.9, 0.9]):
    batch_size, channels, height, width = net_input.shape
    out_low = torch.zeros((batch_size, channels, height // 2, width // 2))
    out_high = torch.zeros((batch_size, channels, height // 2, width // 2))
    for i in range(batch_size):
        img = net_input[i, 0, :, :].cpu().numpy()
        # 对图像的最后两个维度进行 Haar 小波变换
        L_alpha = random.uniform(alpha[0], alpha[1])
        H_beta = random.uniform(beta[0], beta[1])
        LL, (LH, HL, HH) = pywt.dwt2(img, 'haar')
        low = LL
        low = (low - low.min()) / (low.max() - low.min())
        high = HH + HL + LH
        high = (high - high.min()) / (high.max() - high.min())
        

        low = low + L_alpha * high
        high = high + H_beta * low
              
        
        # 将变换后的结果组合成一个新的张量
        out_low[i, 0, :, :] = torch.from_numpy(low[:height // 2, :width // 2])
        out_high[i, 0, :, :] = torch.from_numpy(high[:height // 2, :width // 2])



    # 上采样 low 和 high 到和输入 net_input 同尺寸
    upsampled_low = interpolate(out_low, size=(height, width), mode='bilinear', align_corners=False)
    upsampled_high = interpolate(out_high, size=(height, width), mode='bilinear', align_corners=False)

    return upsampled_low.cuda(), upsampled_high.cuda()


# alpha=[0.2, 0.2], beta=[0.65, 0.65]
# alpha=[0.2, 0.2], beta=[0.8, 0.8]  ab  对照日志
# alpha=[0.1, 0.1], beta=[0.9, 0.9]  ab2
# [2, 1, 112, 112, 80]
def haar_wavelet_transform_3d(net_input, alpha=[0.2, 0.2], beta=[0.5, 0.5]):
    volumn_numpy = net_input.cpu().detach().numpy()
    img_train_low = torch.zeros_like(net_input, device='cpu')
    img_train_high = torch.zeros_like(net_input, device='cpu')
    batch_size, channels, height, width, z = net_input.shape
    for j in range(batch_size):
        img = volumn_numpy[j, 0]
        img_wavelet = pywt.dwtn(img, 'haar')
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = img_wavelet['aaa'], img_wavelet['aad'], img_wavelet['ada'], img_wavelet['add'], img_wavelet['daa'], img_wavelet['dad'], img_wavelet['dda'], img_wavelet['ddd']
        LLL = (LLL - LLL.min()) / (LLL.max() - LLL.min())
        LLH = (LLH - LLH.min()) / (LLH.max() - LLH.min())
        LHL = (LHL - LHL.min()) / (LHL.max() - LHL.min())
        LHH = (LHH - LHH.min()) / (LHH.max() - LHH.min())
        HLL = (HLL - HLL.min()) / (HLL.max() - HLL.min())
        HLH = (HLH - HLH.min()) / (HLH.max() - HLH.min())
        HHL = (HHL - HHL.min()) / (HHL.max() - HHL.min())
        HHH = (HHH - HHH.min()) / (HHH.max() - HHH.min())

        H_ = LLH + LHL + LHH + HLL + HLH + HHL + HHH
        H_ = (H_ - H_.min()) / (H_.max() - H_.min())

        L_alpha = random.uniform(alpha[0], alpha[1])
        L = LLL + L_alpha * H_
        L = (L - L.min()) / (L.max() - L.min())

        H_beta = random.uniform(beta[0], beta[1])
        H = H_ + H_beta * LLL
        H = (H - H.min()) / (H.max() - H.min())
        
        L = torch.tensor(L).unsqueeze(0)
        H = torch.tensor(H).unsqueeze(0)
        L = L_H_aug(L)
        H = L_H_aug(H)
        img_train_low[j] = L
        img_train_high[j] = H
    
    
    return img_train_low.cuda(), img_train_high.cuda()
    
    


def context_mask(img, mask_ratio):
    batch_size, channel, img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x*mask_ratio), int(img_y*mask_ratio), int(img_z*mask_ratio)
    w = np.random.randint(0, 112 - patch_pixel_x)
    h = np.random.randint(0, 112 - patch_pixel_y)
    z = np.random.randint(0, 80 - patch_pixel_z)
    mask[w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
    loss_mask[:, w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
    return mask.long(), loss_mask.long()

def random_mask(img):
    batch_size, channel, img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x*2/3), int(img_y*2/3), int(img_z*2/3)
    mask_num = 27
    mask_size_x, mask_size_y, mask_size_z = int(patch_pixel_x/3)+1, int(patch_pixel_y/3)+1, int(patch_pixel_z/3)
    size_x, size_y, size_z = int(img_x/3), int(img_y/3), int(img_z/3)
    for xs in range(3):
        for ys in range(3):
            for zs in range(3):
                w = np.random.randint(xs*size_x, (xs+1)*size_x - mask_size_x - 1)
                h = np.random.randint(ys*size_y, (ys+1)*size_y - mask_size_y - 1)
                z = np.random.randint(zs*size_z, (zs+1)*size_z - mask_size_z - 1)
                mask[w:w+mask_size_x, h:h+mask_size_y, z:z+mask_size_z] = 0
                loss_mask[:, w:w+mask_size_x, h:h+mask_size_y, z:z+mask_size_z] = 0
    return mask.long(), loss_mask.long()

def concate_mask(img):
    batch_size, channel, img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    z_length = int(img_z * 8 / 27)
    z = np.random.randint(0, img_z - z_length -1)
    mask[:, :, z:z+z_length] = 0
    loss_mask[:, :, :, z:z+z_length] = 0
    return mask.long(), loss_mask.long()

def mix_loss(net3_output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    dice_loss = DICE(net3_output, img_l, mask) * image_weight 
    dice_loss += DICE(net3_output, patch_l, patch_mask) * patch_weight
    loss_ce = image_weight * (CE(net3_output, img_l) * mask).sum() / (mask.sum() + 1e-16) 
    loss_ce += patch_weight * (CE(net3_output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)
    loss = (dice_loss + loss_ce) / 2
    return loss

def sup_loss(output, label):
    label = label.type(torch.int64)
    dice_loss = DICE(output, label)
    loss_ce = torch.mean(CE(output, label))
    loss = (dice_loss + loss_ce) / 2
    return loss

@torch.no_grad()
def update_ema_variables(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)

@torch.no_grad()
def update_ema_students(model1, model2, ema_model, alpha):
    for ema_param, param1, param2 in zip(ema_model.parameters(), model1.parameters(), model2.parameters()):
        ema_param.data.mul_(alpha).add_(((1 - alpha)/2) * param1.data).add_(((1 - alpha)/2) * param2.data)

@torch.no_grad()
def parameter_sharing(model, ema_model):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data = param.data

class BBoxException(Exception):
    pass

def get_non_empty_min_max_idx_along_axis(mask, axis):
    """
    Get non zero min and max index along given axis.
    :param mask:
    :param axis:
    :return:
    """
    if isinstance(mask, torch.Tensor):
        # pytorch is the axis you want to get
        nonzero_idx = (mask != 0).nonzero()
        if len(nonzero_idx) == 0:
            min = max = 0
        else:
            max = nonzero_idx[:, axis].max()
            min = nonzero_idx[:, axis].min()
    elif isinstance(mask, np.ndarray):
        nonzero_idx = (mask != 0).nonzero()
        if len(nonzero_idx[axis]) == 0:
            min = max = 0
        else:
            max = nonzero_idx[axis].max()
            min = nonzero_idx[axis].min()
    else:
        raise BBoxException("Wrong type")
    max += 1
    return min, max


def get_bbox_3d(mask):
    """ Input : [D, H, W] , output : ((min_x, max_x), (min_y, max_y), (min_z, max_z))
    Return non zero value's min and max index for a mask
    If no value exists, an array of all zero returns
    :param mask:  numpy of [D, H, W]
    :return:
    """
    assert len(mask.shape) == 3
    min_z, max_z = get_non_empty_min_max_idx_along_axis(mask, 2)
    min_y, max_y = get_non_empty_min_max_idx_along_axis(mask, 1)
    min_x, max_x = get_non_empty_min_max_idx_along_axis(mask, 0)

    return np.array(((min_x, max_x),
                     (min_y, max_y),
                     (min_z, max_z)))

def get_bbox_mask(mask):
    batch_szie, x_dim, y_dim, z_dim = mask.shape[0], mask.shape[1], mask.shape[2], mask.shape[3]
    mix_mask = torch.ones(batch_szie, 1, x_dim, y_dim, z_dim).cuda()
    for i in range(batch_szie):
        curr_mask = mask[i, ...].squeeze()
        (min_x, max_x), (min_y, max_y), (min_z, max_z) = get_bbox_3d(curr_mask)
        mix_mask[i, :, min_x:max_x, min_y:max_y, min_z:max_z] = 0
    return mix_mask.long()

