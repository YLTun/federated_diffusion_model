import torch
from torchvision.transforms import functional as TF
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision import utils as tv_utils
from torchvision.utils import save_image

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))

from glob import glob
import numpy as np
import collections
from contextlib import contextmanager
from IPython import display
from tqdm import tqdm

from diffusion import sample
import utils

def get_config(dataset_name):
    """Dataset settings."""
    data_config = {
       'cifar_10':  {'img_size': 32, 'channels': 3, 'batch_size': 512, 'train_transform': get_transform('cifar_10', True), 'test_transform': get_transform('cifar_10')},
       'svhn':  {'img_size': 32, 'channels': 3, 'batch_size': 512, 'train_transform': get_transform('svhn', True), 'test_transform': get_transform('svhn')},
       'fashion_mnist': {'img_size': 32, 'channels': 3, 'batch_size': 512, 'train_transform': get_transform('fashion_mnist', True), 'test_transform': get_transform('fashion_mnist')},
       'mnist': {'img_size': 32, 'channels': 3, 'batch_size': 512, 'train_transform': get_transform('mnist', True), 'test_transform': get_transform('mnist')},
       'chars74k_fnt_num':  {'img_size': 32, 'channels': 3, 'batch_size': 512, 'train_transform': get_transform('chars74k_fnt_num', True), 'test_transform': get_transform('chars74k_fnt_num')},
       'sars_cov_2_ct_scan': {'img_size': 64, 'channels': 3, 'batch_size': 128, 'train_transform': get_transform('sars_cov_2_ct_scan', train_set=True, img_size=64)},
    }

    train_config = {
        'lr':2e-4, 
        'timesteps': 300,       # 500               # The number of timesteps to use when sampling. 
        'epochs': 1500,          # 300, 1500              # rounds * local_epochs
        'rounds': 300,
        'local_epochs': 5,
        'ema_decay': 0.998,                     
        'eta': 1.,                              # The amount of noise to add each timestep when sampling (0 = no noise (DDIM), 1 = full noise (DDPM))
        'save_interval': 100,
    }

    return data_config[dataset_name], train_config
    

def get_transform(dataset_name, train_set=False, img_size=32):
    """Given dataset name, we get the corresponding transform."""

    tf = list()
    if train_set and dataset_name != 'mnist' and dataset_name != 'svhn' and dataset_name != 'chars74k_fnt_num':
        tf.append(transforms.RandomHorizontalFlip())

    tf.extend([
            transforms.ToTensor(), 
            transforms.Resize(size=(img_size, img_size)),
            transforms.Normalize([0.5], [0.5]),
    ])
    return transforms.Compose(tf)


class ImgDataset(Dataset):
    def __init__(self, parent_dir, label_idx_dict=None, transform=None):
        self.img_list = []
        self.label_list = []
        self.label_idx_dict = label_idx_dict
        self.label_count_dict = {}
        
        sub_dirs = [f.name for f in os.scandir(parent_dir) if f.is_dir()]
        sub_dirs.sort()
        if self.label_idx_dict is None:
            self.label_idx_dict = {label:idx for idx, label in enumerate(sub_dirs)}

        self.classes = self.label_idx_dict.keys()  # To show what classes are in the dataset.
            
        for sub_dir in sub_dirs:
            full_path = os.path.join(parent_dir, sub_dir)
            img_paths = glob(os.path.join(full_path, '*.JPG')) + glob(os.path.join(full_path, '*.jpg')) + glob(os.path.join(full_path, '*.png'))
            labels = [self.label_idx_dict[sub_dir]] * len(img_paths)
            self.img_list += img_paths
            self.label_list += labels
            self.label_count_dict[sub_dir] = len(labels)
            
        self.transform = transform
        
    def __len__(self):
        return len(self.label_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = default_loader(img_path)
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    
# Utilities
@contextmanager
def train_mode(model, mode=True):
    """A context manager that places a model into training mode and restores
    the previous mode on exit."""
    modes = [module.training for module in model.modules()]
    try:
        yield model.train(mode)
    finally:
        for i, module in enumerate(model.modules()):
            module.training = modes[i]


def eval_mode(model):
    """A context manager that places a model into evaluation mode and restores
    the previous mode on exit."""
    return train_mode(model, False)


@torch.no_grad()
def ema_update(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].mul_(decay).add_(param, alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)
        

# @torch.no_grad()
# @torch.random.fork_rng()
# # @eval_mode(model_ema)
# def demo(model, save_path, steps, eta, seed=42):
#   device = model.device
#   torch.manual_seed(seed)

#   noise = torch.randn([100, 3, 32, 32], device=device)
#   fakes_classes = torch.arange(10, device=device).repeat_interleave(10, 0)
#   with eval_mode(model):
#       fakes = sample(model, noise, steps, eta, fakes_classes)

#   grid = tv_utils.make_grid(fakes, 10).cpu()
#   TF.to_pil_image(grid.add(1).div(2).clamp(0, 1)).save(save_path)
#   display.display(display.Image(save_path))
#   tqdm.write('')
    
    
@torch.no_grad()
@torch.random.fork_rng()
# @eval_mode(model_ema)
def demo(model, save_path, steps, eta, img_size=32, num_images=10, num_classes=10, classes=None, num_img_per_row=None, seed=42):
    device = model.device
    torch.manual_seed(seed)

    if classes:
        noise = torch.randn([len(classes) * num_images, 3, img_size, img_size], device=device)
        fakes_classes = torch.tensor(classes, device=device).repeat_interleave(num_images, 0)
    else:
        noise = torch.randn([num_classes * num_images, 3, img_size, img_size], device=device)
        fakes_classes = torch.arange(num_classes, device=device).repeat_interleave(num_images, 0)
    
    with eval_mode(model):
        fakes = sample(model, noise, steps, eta, fakes_classes)

    num_img_per_row = num_images if num_img_per_row == None else num_img_per_row
    grid = tv_utils.make_grid(fakes, num_img_per_row).cpu()
    TF.to_pil_image(grid.add(1).div(2).clamp(0, 1)).save(save_path)
    display.display(display.Image(save_path))
    tqdm.write('')

    
@torch.no_grad()
@torch.random.fork_rng()
def generate_img(model, save_path, steps, eta, num_imgs, img_class, img_size=32, seed=42):
    device = model.device
    torch.manual_seed(seed)
    
    noise = torch.randn([num_imgs, 3, img_size, img_size], device=device)
    fakes_classes = torch.tensor(img_class).repeat(num_imgs).to(device)
    with eval_mode(model):
        fakes = sample(model, noise, steps, eta, fakes_classes)
    
    for i, fake_img in enumerate(fakes):
        img_save_path = os.path.join(save_path, '{}.png'.format(str(i + 1).zfill(3)))
        TF.to_pil_image(fake_img.add(1).div(2).clamp(0, 1)).save(img_save_path)

    
@torch.no_grad()
def evaluate_model(model, data_loader, loss_fn, tqdm_desc=None, seed=42):
    
    device = model.device
    loss_metric = utils.MeanMetric()
    with eval_mode(model):
        torch.manual_seed(seed)
        for (x, y) in tqdm(data_loader, desc=tqdm_desc):
            x = x.to(device)
            y = y.to(device)
            loss = loss_fn(model, x, y)
            loss_metric.update_state(loss.item())

    return loss_metric.result()
