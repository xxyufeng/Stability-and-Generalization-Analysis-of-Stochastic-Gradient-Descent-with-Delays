import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler, random_split
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST
import numpy as np
import os
import urllib.request
import sklearn.datasets
from torchvision import models

############################
# Dataset preparation
############################
class LibSVMDataset(torch.utils.data.Dataset):
    def __init__(self, url, dataset_path, download=False, dimensionality=None, classes=None):
        self.url = url
        self.dataset_path = dataset_path
        self._dimensionality = dimensionality

        self.filename = os.path.basename(url)
        self.dataset_type = os.path.basename(os.path.dirname(url))

        if not os.path.isfile(self.local_filename):
            if download:
                print(f"Downloading {url}")
                self._download()
            else:
                raise RuntimeError(
                    "Dataset not found. You can use download=True to download it."
                )
        else:
            print("Files already downloaded")

        self.data, y = sklearn.datasets.load_svmlight_file(self.local_filename)

        sparsity = self.data.nnz / (self.data.shape[0] * self.data.shape[1])
        if sparsity > 0.1:
            self.data = self.data.todense().astype(np.float32)
            self._is_sparse = False
        else:
            self._is_sparse = True

        # convert labels to [0, 1]
        if classes is None:
            classes = np.unique(y)
        self.classes = np.sort(classes)
        self.targets = torch.zeros(len(y), dtype=torch.int64)
        for i, label in enumerate(self.classes):
            self.targets[y == label] = i

        self.class_to_idx = {cl: idx for idx, cl in enumerate(self.classes)}

        super().__init__()

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_features(self):
        return self.data.shape[1]
    

    def __getitem__(self, idx):
        if self._is_sparse:
            x = torch.from_numpy(self.data[idx].todense().astype(np.float32)).flatten()
        else:
            x = torch.from_numpy(self.data[idx]).flatten()
        y = self.targets[idx]

        if self._dimensionality is not None:
            if len(x) < self._dimensionality:
                x = torch.cat([x, torch.zeros([self._dimensionality - len(x)], dtype=x.dtype, device=x.device)])
            elif len(x) > self._dimensionality:
                raise RuntimeError("Dimensionality is set wrong.")

        return x, y

    def __len__(self):
        return len(self.targets)

    @property
    def local_filename(self):
        return os.path.join(self.dataset_path, self.dataset_type, self.filename)

    def _download(self):
        os.makedirs(os.path.dirname(self.local_filename), exist_ok=True)
        urllib.request.urlretrieve(self.url, filename=self.local_filename)


class RCV1(LibSVMDataset):
    def __init__(self, split, download=False, dataset_path=None):
        if split == "train":
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2"
        elif split == "test":
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2"
        else:
            raise RuntimeError(f"Unavailable split {split}")
        super().__init__(url=url, download=download, dataset_path=dataset_path)


class GISETTE(LibSVMDataset):
    def __init__(self, split, download=False, dataset_path=None):
        if split == "train":
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2"
        elif split == "test":
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.t.bz2"
        else:
            raise RuntimeError(f"Unavailable split {split}")
        super().__init__(url=url, download=download, dataset_path=dataset_path)

class ijcnn(LibSVMDataset):
    def __init__(self, split, download=False, dataset_path=None):
        if split == "train":
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.tr.bz2"
        elif split == "test":
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.t.bz2"
        else:
            raise RuntimeError(f"Unavailable split {split}")
        super().__init__(url=url, download=download, dataset_path=dataset_path)

class w1a(LibSVMDataset):
    def __init__(self, split, download=False, dataset_path=None, dimensionality=None, subset=1):
        if split == "train":
            url = f"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w{subset}a"
        elif split == "test":
            url = f"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w{subset}a.t"
        else:
            raise RuntimeError(f"Unavailable split {split}")
        super().__init__(url=url, download=download, dataset_path=dataset_path)

class a1a(LibSVMDataset):
    def __init__(self, split, download=False, dataset_path=None, dimensionality=None, subset=1):

        if split == "train":
            url = f"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a{subset}a"
        elif split == "test":
            url = f"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a{subset}a.t"
        else:
            raise RuntimeError(f"Unavailable split {split}")
        super().__init__(url=url, download=download, dataset_path=dataset_path, dimensionality=dimensionality)


# Data loading
def load_data(dataset_name, dataset_path, split_type):
    if split_type == 'train':
        if dataset_name == 'cifar10':
          train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
          return CIFAR10(root=dataset_path, train=True, download=True, transform=train_transform)
        elif dataset_name == 'mnist':
          train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
          return MNIST(root=dataset_path, train=True, download=True, transform=train_transform)
        elif dataset_name == 'rcv1':
            return RCV1("train", download=True, dataset_path=dataset_path)
        elif dataset_name == 'gisette':
            return GISETTE("train", download=True, dataset_path=dataset_path)
        elif dataset_name == 'ijcnn':
            return ijcnn("train", download=True, dataset_path=dataset_path)
        elif dataset_name.startswith('w') and dataset_name.endswith('a') and len(dataset_name) == 3:
            subset = int(dataset_name[1])
            return w1a("train", download=True, dataset_path=dataset_path, dimensionality=123, subset=subset)
        elif dataset_name.startswith('a') and dataset_name.endswith('a') and len(dataset_name) == 3:
            subset = int(dataset_name[1])
            return a1a("train", download=True, dataset_path=dataset_path, dimensionality=123, subset=subset)
        


    elif split_type == 'test':
        if dataset_name == 'cifar10':
          test_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
          return CIFAR10(root=dataset_path, train=False, download=True, transform=test_transform)
        elif dataset_name == 'mnist':
          test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
          return MNIST(root=dataset_path, train=False, download=True, transform=test_transform)
        elif dataset_name == 'rcv1':
            return RCV1("test", download=True, dataset_path=dataset_path)
        elif dataset_name == 'gisette':
            return GISETTE("test", download=True, dataset_path=dataset_path)
        elif dataset_name == 'ijcnn':
            return ijcnn("test", download=True, dataset_path=dataset_path)
        elif dataset_name.startswith('w') and dataset_name.endswith('a') and len(dataset_name) == 3: 
            subset = int(dataset_name[1])
            return w1a("test", download=True, dataset_path=dataset_path, dimensionality=123, subset=subset)
        elif dataset_name.startswith('a') and dataset_name.endswith('a') and len(dataset_name) == 3:
            subset = int(dataset_name[1])
            return a1a("test", download=True, dataset_path=dataset_path, dimensionality=123, subset=subset)

############################
# Model and loss functions
############################
def mse_loss(pred, target, num_classes=2):
    target = target.long()
    target = torch.zeros((target.shape[0], 2)).scatter(1, target.unsqueeze(-1), 1)
    f = nn.MSELoss()
    return f(pred, target)

def hinge_loss(pred, target, q=1.5):
    """q-norm hinge loss"""
    target = target.long()
    target = nn.functional.one_hot(target, num_classes=2).float()
    margin = 1 - (2 * target - 1) * (pred)  # make sure the label values in [-1,1]
    loss = torch.clamp(margin, min=0) ** q
    return loss.mean()

class Linear_CIFAR10(nn.Module):
    def __init__(self, loss='mse',q=1.5, random_seed=0):
        super(Linear_CIFAR10, self).__init__()
        gen = torch.Generator().manual_seed(42+random_seed)
        self.fc1 = nn.Linear(3072, 10)
        nn.init.normal_(self.fc1.weight, mean=0, std=0.1, generator=gen)
        nn.init.constant_(self.fc1.bias, 0.1)
        if loss == 'mse':
            self.loss = lambda pred, target: mse_loss(pred, target)
        elif loss == 'hingeloss':
            if q is None:
                raise ValueError("q must be specified for hinge loss")
            self.loss = lambda pred, target: hinge_loss(pred, target, q)
        elif loss == 'ce':
            self.loss = nn.CrossEntropyLoss()
        else:
            raise ValueError("Unsupported loss function. Use 'mse' or 'hingeloss' or 'ce'.")

    def forward(self, x, target):
        target = target.long()
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        output = self.fc1(x)
        loss = self.loss(output, target)
        return output, loss

class Linear_RCV1(nn.Module):
    def __init__(self, loss='mse',q=1.5, random_seed=0):
        super(Linear_RCV1, self).__init__()
        gen = torch.Generator().manual_seed(42+random_seed)
        self.fc1 = nn.Linear(47236, 2)
        nn.init.normal_(self.fc1.weight, mean=0, std=0.1, generator=gen)
        nn.init.constant_(self.fc1.bias, 0.1)
        if loss == 'mse':
            self.loss = lambda pred, target: mse_loss(pred, target)
        elif loss == 'hingeloss':
            if q is None:
                raise ValueError("q must be specified for hinge loss")
            self.loss = lambda pred, target: hinge_loss(pred, target, q)
        elif loss == 'ce':
            self.loss = nn.CrossEntropyLoss()
        else:
            raise ValueError("Unsupported loss function. Use 'mse' or 'hingeloss' or 'ce'.")

    def forward(self, x, target):
        target = target.long()
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        output = self.fc1(x)
        loss = self.loss(output, target)
        return output, loss
    
class Linear_w1a(nn.Module):
    def __init__(self, loss='mse',q=1.5, random_seed=0):
        super(Linear_w1a, self).__init__()
        gen = torch.Generator().manual_seed(42+random_seed)
        self.fc1 = nn.Linear(300, 1)
        nn.init.normal_(self.fc1.weight, mean=0, std=0.01, generator=gen)
        nn.init.constant_(self.fc1.bias, 0)
        if loss == 'mse':
            self.loss = lambda pred, target: mse_loss(pred, target)
        elif loss == 'hingeloss':
            if q is None:
                raise ValueError("q must be specified for hinge loss")
            self.loss = lambda pred, target: hinge_loss(pred, target, q)
        elif loss == 'ce':
            self.loss = nn.CrossEntropyLoss()
        else:
            raise ValueError("Unsupported loss function. Use 'mse' or 'hingeloss' or 'ce'.")

    def forward(self, x, target):
        target = target.long()
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        output = self.fc1(x)
        loss = self.loss(output, target)
        return output, loss

class Linear_a1a(nn.Module):
    def __init__(self, loss='mse',q=1.5, random_seed=0):
        super(Linear_a1a, self).__init__()
        gen = torch.Generator().manual_seed(42+random_seed)
        self.fc1 = nn.Linear(123, 2)
        nn.init.normal_(self.fc1.weight, mean=0, std=0.001, generator=gen)
        nn.init.constant_(self.fc1.bias, 0)
        if loss == 'mse':
            self.loss = lambda pred, target: mse_loss(pred, target)
        elif loss == 'hingeloss':
            if q is None:
                raise ValueError("q must be specified for hinge loss")
            self.loss = lambda pred, target: hinge_loss(pred, target, q)
        elif loss == 'ce':
            self.loss = nn.CrossEntropyLoss()
        else:
            raise ValueError("Unsupported loss function. Use 'mse' or 'hingeloss' or 'ce'.")

    def forward(self, x, target):
        target = target.long()
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        output = self.fc1(x)
        loss = self.loss(output, target)
        return output, loss

class Linear_GISETTE(nn.Module):
    def __init__(self, loss='mse',q=1.5, random_seed=0):
        super(Linear_GISETTE, self).__init__()
        gen = torch.Generator().manual_seed(42+random_seed)
        self.fc1 = nn.Linear(5000, 2)
        nn.init.normal_(self.fc1.weight, mean=0, std=0.01, generator=gen)
        nn.init.constant_(self.fc1.bias, 0)
        if loss == 'mse':
            self.loss = lambda pred, target: mse_loss(pred, target)
        elif loss == 'hingeloss':
            if q is None:
                raise ValueError("q must be specified for hinge loss")
            self.loss = lambda pred, target: hinge_loss(pred, target, q)
        elif loss == 'ce':
            self.loss = nn.CrossEntropyLoss()
        else:
            raise ValueError("Unsupported loss function. Use 'mse' or 'hingeloss' or 'ce'.")

    def forward(self, x, target):
        target = target.long()
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        output = self.fc1(x)
        loss = self.loss(output, target)
        return output, loss

class FCNET_MNIST(nn.Module):
    def __init__(self, loss='mse',q=1.5, random_seed=0):
        super(FCNET_MNIST, self).__init__()
        gen = torch.Generator().manual_seed(42+random_seed)
        self.fc1 = nn.Linear(32*32, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.kaiming_normal_(layer.weight, generator=gen)
            nn.init.constant_(layer.bias, 0.01)
        self.loss = nn.CrossEntropyLoss()
        # print("Only support CE-Loss!")

    def forward(self, x, target):
        target = target.long()
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        output = F.relu(self.fc1(x))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        loss = self.loss(output, target)
        return output, loss
    
class Linear_MNIST(nn.Module):
    def __init__(self, loss='mse',q=1.5, random_seed=0):
        super(Linear_MNIST, self).__init__()
        gen = torch.Generator().manual_seed(42+random_seed)
        self.fc1 = nn.Linear(32*32, 10)
        nn.init.kaiming_normal_(self.fc1.weight, generator=gen)
        nn.init.constant_(self.fc1.bias, 0)
        if loss == 'mse':
            self.loss = lambda pred, target: mse_loss(pred, target)
        elif loss == 'hingeloss':
            if q is None:
                raise ValueError("q must be specified for hinge loss")
            self.loss = lambda pred, target: hinge_loss(pred, target, q)
        elif loss == 'ce':
            self.loss = nn.CrossEntropyLoss()
        else:
            raise ValueError("Unsupported loss function. Use 'mse' or 'hingeloss' or 'ce'.")

    def forward(self, x, target):
        target = target.long()
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        output = self.fc1(x)
        loss = self.loss(output, target)
        return output, loss

class Linear_ijcnn(nn.Module):
    def __init__(self, loss='mse',q=1.5, random_seed=0):
        super(Linear_ijcnn, self).__init__()
        gen = torch.Generator().manual_seed(42+random_seed)
        self.fc1 = nn.Linear(22, 2)
        nn.init.normal_(self.fc1.weight, mean=0, std=0.01, generator=gen)
        nn.init.constant_(self.fc1.bias, 0)
        if loss == 'mse':
            self.loss = lambda pred, target: mse_loss(pred, target)
        elif loss == 'hingeloss':
            if q is None:
                raise ValueError("q must be specified for hinge loss")
            self.loss = lambda pred, target: hinge_loss(pred, target, q)
        elif loss == 'ce':
            self.loss = nn.CrossEntropyLoss()
        else:
            raise ValueError("Unsupported loss function. Use 'mse' or 'hingeloss'.")

    def forward(self, x, target):
        target = target.long()
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        output = self.fc1(x)
        loss = self.loss(output, target)
        return output, loss
    
class ResNet18_CIFAR10(nn.Module):
    def __init__(self, loss='ce', random_seed=0):
        super().__init__()
        torch.manual_seed(42 + random_seed)
        self.backbone = models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 10)
        nn.init.kaiming_normal_(self.backbone.fc.weight)
        nn.init.constant_(self.backbone.fc.bias, 0.0)
        if loss != 'ce':
            raise ValueError("ResNet18_CIFAR10 仅支持 CrossEntropy (loss_name='ce')")
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, target):
        target = target.long()
        # batch_size = x.size(0)
        # x = x.view(batch_size, -1)
        logits = self.backbone(x)
        loss = self.loss_fn(logits, target)
        return logits, loss

def load_model(model_name, loss='mse', q=1.5, random_seed=0):
    if model_name == 'linear_rcv1':
        model = Linear_RCV1(loss=loss, q=q, random_seed=random_seed)
        return model
    elif model_name == 'linear_gisette':
        model = Linear_GISETTE(loss=loss, q=q, random_seed=random_seed )
        return model
    elif model_name == 'fcnet_mnist':
        model = FCNET_MNIST(loss=loss, q=q, random_seed=random_seed)
        return model
    elif model_name == 'linear_mnist':
        model = Linear_MNIST(loss=loss, q=q, random_seed=random_seed)
        return model
    elif model_name == 'linear_ijcnn':
        model = Linear_ijcnn(loss=loss, q=q, random_seed=random_seed)
        return model
    elif model_name == 'linear_w1a':
        model = Linear_w1a(loss=loss, q=q, random_seed=random_seed)
        return model
    elif model_name == 'resnet18_cifar10':
        model = ResNet18_CIFAR10(loss=loss, random_seed=random_seed)
        return model
    elif model_name == 'linear_a1a':
        model = Linear_a1a(loss=loss, q=q, random_seed=random_seed)
        return model
    else:
        raise ValueError("Unsupported model name. Use 'linear_rcv1' or 'linear_gisette', 'fcnet_mnist', 'linear_ijcnn', 'resnet18_cifar10', 'linear_mnist'.")