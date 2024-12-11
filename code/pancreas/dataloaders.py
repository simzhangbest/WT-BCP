import numpy as np
import torch
import h5py

from torch import import_ir_module, nn as nn, optim as optim
from torch.utils.data import DataLoader
from Vnet import VNet
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from XNetv2_3D import XNetv2_3D_min
import pywt
import torchio as tio
from torchio import transforms as T
import random

patch_size = (96, 96, 96)
normalize_t = tio.ZNormalization.mean
L_H_aug = T.Compose([T.Resize(patch_size), T.ZNormalization(masking_method=normalize_t)])

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


def create_Vnet(ema=False):
    net = VNet()
    net = nn.DataParallel(net)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def BCP_xnet_v2_3d(in_chns=1, class_num=2, ema=False):
    net = XNetv2_3D_min(in_chns, class_num).cuda()
    if ema:
        for param in net.parameters():
            param.detach_()
    return net


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def _get_transform(self, x):
        if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - x.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - x.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - x.shape[2]) // 2 + 1, 0)
            x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = x.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        def do_transform(image):
            if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= self.output_size[2]:
                try:
                    image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
                except Exception as e:
                    print(e)
            image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return image

        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]
    

class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def _get_transform(self, label):
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 1, 0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = label.shape
        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        def do_transform(x):
            if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2]:
                x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            x = x[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return x

        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]
    
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample[0]
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        sample = [image] + [*sample[1:]]
        return [torch.from_numpy(s.astype(np.float32)) for s in sample]
    

def get_dataset_path(dataset='pancreas', labelp='20percent'):
    files = ['train_lab.txt', 'train_unlab.txt', 'test.txt']
    return ['/'.join(['/mnt/zmy/code/BCP/dataset_mnt/07_Pancreas/data_lists', dataset, labelp, f]) for f in files]



class Pancreas(Dataset):
    """ Pancreas Dataset """
    def __init__(self, base_dir, name, split, no_crop=False, labelp=20, reverse=False, TTA=False):  # train 脚本中给了20%
        self._base_dir = base_dir
        self.split = split
        self.reverse=reverse
        self.labelp = '10percent'
        if labelp == 20:
            self.labelp = '20percent'

        tr_transform = Compose([
            # RandomRotFlip(),
            RandomCrop((96, 96, 96)),
            # RandomNoise(),
            ToTensor()
        ])
        if no_crop:
            test_transform = Compose([
                # CenterCrop((160, 160, 128)),
                CenterCrop((96, 96, 96)),
                ToTensor()
            ])
        else:
            test_transform = Compose([
                CenterCrop((96, 96, 96)),
                ToTensor()
            ])

        data_list_paths = get_dataset_path(name, self.labelp)

        if split == 'train_lab':
            data_path = data_list_paths[0]
            self.transform = tr_transform
        elif split == 'train_unlab':
            data_path = data_list_paths[1]
            self.transform = test_transform  # tr_transform
        else: # test
            data_path = data_list_paths[2]
            self.transform = test_transform

        with open(data_path, 'r') as f:
            self.image_list = f.readlines()

        self.image_list = [self._base_dir + "/{}".format(item.strip()) for item in self.image_list]
        print("Split : {}, total {} samples".format(split, len(self.image_list)))

    def __len__(self):
        if self.split == 'train_lab' and self.labelp == '20percent':
            return len(self.image_list) * 5
        elif self.split == 'train_lab' and self.labelp == '10percent':
            return len(self.image_list) * 10
        else:
            return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx % len(self.image_list)]
        if self.reverse:
            image_path = self.image_list[len(self.image_list) - idx % len(self.image_list) - 1]
        h5f = h5py.File(image_path, 'r')
        image, label = h5f['image'][:], h5f['label'][:].astype(np.float32)
        samples = image, label
        if self.transform:
            tr_samples = self.transform(samples)
        image_, label_ = tr_samples
        return image_.float(), label_.long()


def get_ema_model_and_dataloader(data_root, split_name, batch_size, lr, labelp=10):
    print("Initialize ema cutmix: network, optimizer and datasets...")
    """Net & optimizer"""
    net = BCP_xnet_v2_3d()
    ema_net = BCP_xnet_v2_3d(ema=True).cuda()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    trainset_lab_a = Pancreas(data_root, split_name, split='train_lab', labelp=labelp)
    lab_loader_a = DataLoader(trainset_lab_a, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_lab_b = Pancreas(data_root, split_name, split='train_lab', labelp=labelp, reverse=True)
    lab_loader_b = DataLoader(trainset_lab_b, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    
    trainset_unlab_a = Pancreas(data_root, split_name, split='train_unlab', labelp=labelp)
    unlab_loader_a = DataLoader(trainset_unlab_a, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_unlab_b = Pancreas(data_root, split_name, split='train_unlab', labelp=labelp, reverse=True)
    unlab_loader_b = DataLoader(trainset_unlab_b, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    
    testset = Pancreas(data_root, split_name, split='test')
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    return net, ema_net, optimizer, lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, test_loader