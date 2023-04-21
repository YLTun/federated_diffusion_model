import pickle
import os
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import functional as TF
import numpy as np
from IPython import display
# from sklearn import metrics
from pynvml import *
from contextlib import contextmanager


# Util for saving objects in pickle format.
def save_pickle(file_path, obj):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


# Util for loading objects from pickle format.
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def print_separator(text='', seperator_length=20):
    print('\n', '='*seperator_length + ' ' + text + ' ' + '='*seperator_length + '\n')


# Util for printing result at the end of each round.
def print_result(performance_log):
    print('train     -              loss :    {:.4f}          acc:    {:.4f}'.format(performance_log['train_loss'][-1], performance_log['train_acc'][-1]))
    print('valid     -              loss :    {:.4f}          acc:    {:.4f}'.format(performance_log['valid_loss'][-1], performance_log['valid_acc'][-1]))
    print()


# For plotting model history.
def save_history_plot(history_log, plot_config):
    
    plt.figure()
    for attribute, label in zip(plot_config['attributes'], plot_config['labels']):
        plt.plot(history_log[attribute], label=label)
    plt.title(plot_config['title'])
    plt.xlabel(plot_config['xlabel'])
    plt.ylabel(plot_config['ylabel'])
    plt.grid(True, linestyle='-.')
    plt.legend()

    if plot_config['save_dir'] != None:
        plt.savefig(plot_config['save_dir'], dpi=200, bbox_inches='tight')
    if plot_config['show_img']:
        plt.show()

    plt.close('all')


# Compute accuracy.
def compute_accuracy(y_batch, y_pred):
    _, predicted = torch.max(y_pred, 1)
    accuracy = (predicted == y_batch).sum().item() / len(y_batch)
    # accuracy = metrics.accuracy_score(y_batch.cpu(), predicted.cpu())
    return accuracy


# Helper for computing metrics at each epoch.
class MeanMetric():
    
    def __init__(self):
        self.total = np.float32(0)
        self.count = np.float32(0)

    def update_state(self, value):
        self.total += value
        self.count += 1
        
    def result(self):
        if self.count > 0:
            return self.total / self.count
        else:
            return np.nan

    def reset_state(self):
        self.total = np.float32(0)
        self.count = np.float32(0)


# Plotting settings
LOSS_PLOT_CONFIG = {
    'figsize' : (8, 6),
    'attributes': ('train_loss', 'valid_loss'),
    'labels': ('train', 'valid'),
    'title': 'Loss',
    'xlabel': 'rounds',
    'ylabel': 'loss',
    'save_dir': None,   
    'show_img': False,
}

ACC_PLOT_CONFIG = {
    'figsize' : (8, 6),
    'attributes': ['train_acc', 'valid_acc'],
    'labels': ['train', 'valid'],
    'title': 'Accuracy',
    'xlabel': 'rounds',
    'ylabel': 'accuracy',
    'save_dir': None,
    'show_img': False,
}

# For plotting model history.
def save_history_plot(history_log, plot_config):
    
    plt.figure(figsize=plot_config['figsize'])
    for attribute, label in zip(plot_config['attributes'], plot_config['labels']):
        plt.plot(history_log[attribute], label=label)
    plt.title(plot_config['title'])
    plt.xlabel(plot_config['xlabel'])
    plt.ylabel(plot_config['ylabel'])
    plt.grid(True, linestyle='-.')
    plt.legend()

    if plot_config['save_dir'] != None:
        plt.savefig(plot_config['save_dir'], dpi=300, bbox_inches='tight')
    if plot_config['show_img']:
        plt.show()

    plt.close('all')


def save_plot(data_list, plot_config):
    plt.figure(figsize=plot_config['figsize'])
    for data, label in zip(data_list, plot_config['labels']):
        plt.plot(data, label=label)
    plt.title(plot_config['title'])
    plt.xlabel(plot_config['xlabel'])
    plt.ylabel(plot_config['ylabel'])
    plt.grid(True, linestyle='-.')
    plt.legend()

    if plot_config['save_dir'] != None:
        plt.savefig(plot_config['save_dir'], dpi=300, bbox_inches='tight')
    if plot_config['show_img']:
        plt.show()

    plt.close('all')


# Helpers for logging model performance.
def get_performance_loggers(metric_keys = {'train_loss', 'train_acc', 'valid_loss', 'valid_acc'}):
	performance_dict, performance_log = dict(), dict()
	for key in metric_keys:
	    performance_dict[key] = MeanMetric()
	    performance_log[key] = list()
	return performance_dict, performance_log


# GPU memory allocation workaround.
def allocate_gpu_memory(mem_amount=10504699904):
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {info.total}')
    print(f'free     : {info.free}')
    print(f'used     : {info.used}')

    block_mem = int(info.free * 0.7) // (32 // 8)
    block_mem = int(block_mem * 0.7)
    block_mem

    device_count = torch.cuda.device_count()
    for i in range(device_count):
        device = torch.device('cuda:{}'.format(i) if torch.cuda.is_available() else 'cpu')
        x = torch.rand(block_mem, dtype=torch.float32).to(device)
        x = torch.rand(1)
        del x

    nvmlShutdown()


def save_notes(file_path, notes):
    with open(file_path, 'w') as f:
        f.write(notes)


# Util for showing results.
def get_mean_std(num_list):
    print('mean:{:.2f}'.format(np.mean(num_list)))
    print('std:{:.2f}'.format(np.std(num_list)))


# For showing tensor form images.
# def show_img_tensor(img_tensor):
#     img_np = img_tensor.numpy()
#     plt.imshow(np.transpose(img_np, (1, 2, 0)))
#     plt.show()


def show_img_tensor(img_tensor, save_path=None, dpi=1200):
    img_tensor = img_tensor.cpu()
    img = TF.to_pil_image(img_tensor.add(1).div(2).clamp(0, 1))
    plt.axis('off')
    plt.imshow(img)
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()


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