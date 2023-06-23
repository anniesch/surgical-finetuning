from pathlib import Path
from robustbench.data import load_cifar10c
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import torchvision
import numpy as np


def get_loaders(cfg, corruption_type, severity):
    if cfg.data.dataset_name == "cifar10":
        x_corr, y_corr = load_cifar10c(
            10000, severity, cfg.user.root_dir, False, [corruption_type]
        )
        assert cfg.args.train_n <= 9000
        labels = {}
        num_classes = int(max(y_corr)) + 1
        for i in range(num_classes):
            labels[i] = [ind for ind, n in enumerate(y_corr) if n == i]
        num_ex = cfg.args.train_n // num_classes
        tr_idxs = []
        val_idxs = []
        test_idxs = []
        for i in range(len(labels.keys())):
            np.random.shuffle(labels[i])
            tr_idxs.append(labels[i][:num_ex])
            val_idxs.append(labels[i][num_ex:num_ex+10])
            test_idxs.append(labels[i][num_ex+10:num_ex+100])
        tr_idxs = np.concatenate(tr_idxs)
        val_idxs = np.concatenate(val_idxs)
        test_idxs = np.concatenate(test_idxs)
        
        tr_dataset = TensorDataset(x_corr[tr_idxs], y_corr[tr_idxs])
        val_dataset = TensorDataset(x_corr[val_idxs], y_corr[val_idxs])
        te_dataset = TensorDataset(x_corr[test_idxs], y_corr[test_idxs])
    
    elif cfg.data.dataset_name == "imagenet-c":
        data_root = Path(cfg.user.root_dir)
        image_dir = data_root / "ImageNet-C" / corruption_type / str(severity)
        dataset = ImageFolder(image_dir, transform=transforms.ToTensor())
        indices = list(range(len(dataset.imgs))) #50k examples --> 50 per class
        assert cfg.args.train_n <= 20000
        labels = {}
        y_corr = dataset.targets
        for i in range(max(y_corr)+1):
            labels[i] = [ind for ind, n in enumerate(y_corr) if n == i] 
        num_ex = cfg.args.train_n // (max(y_corr)+1)
        tr_idxs = []
        val_idxs = []
        test_idxs = []
        for i in range(len(labels.keys())):
            np.random.shuffle(labels[i])
            tr_idxs.append(labels[i][:num_ex])
            val_idxs.append(labels[i][num_ex:num_ex+10])
            test_idxs.append(labels[i][num_ex+10:num_ex+20])
        tr_idxs = np.concatenate(tr_idxs)
        val_idxs = np.concatenate(val_idxs)
        test_idxs = np.concatenate(test_idxs)
        tr_dataset = Subset(dataset, tr_idxs)
        val_dataset = Subset(dataset, val_idxs)
        te_dataset = Subset(dataset, test_idxs)

    