import torch
import numpy as np
from sklearn import datasets
from torch.utils.data import Subset, Dataset, DataLoader, ConcatDataset
from sklearn.model_selection import KFold


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class Data:
    def __init__(self, params):
        self.params = params
        self.gen_datasets()
    
    def gen_datasets(self):
        raise NotImplementedError

    def get_loaders(self, batch_size, num_workers, shuffle_train=False,
                    get_test=True):
        raise NotImplementedError


class GaussianMixture(Data):
    def __init__(self, params):
        super().__init__(params)
    
    def gen_datasets(self):
        np.random.seed(1)
        V_size, S_size = self.params.v_size, self.params.s_size
        
        self.V_train, self.S_train = get_gaussian_mixture_dataset(V_size, S_size)
        self.V_val, self.S_val = get_gaussian_mixture_dataset(V_size, S_size)
        self.V_test, self.S_test = get_gaussian_mixture_dataset(V_size, S_size)
        
        self.fea_size = 2
        self.x_lim, self.y_lim = 2, 2
        
    def get_loaders(self, batch_size, num_workers, shuffle_train=False, get_test=True, transform=None):
        train_dataset = SetDataset(self.V_train, self.S_train, self.params, is_train=True, transform=transform)
        val_dataset = SetDataset(self.V_val, self.S_val, self.params, is_train=True, transform=transform)
        test_dataset = SetDataset(self.V_test, self.S_test, self.params, is_train=True, transform=transform)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                    shuffle=shuffle_train, num_workers=num_workers,
                                  collate_fn=numpy_collate)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers,
                                  collate_fn=numpy_collate)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers,
                                  collate_fn=numpy_collate) if get_test else None
        return train_loader, val_loader, test_loader

    def get_kfold_loaders(self, batch_size, num_workers, fold, shuffle_train=False, get_test=True, transform=None):
        # Combine train and validation sets for k-fold split
        combined_V = np.concatenate((self.V_train, self.V_val), axis=0)
        combined_S = np.concatenate((self.S_train, self.S_val), axis=0)

        kf = KFold(n_splits=5, shuffle=True, random_state=1)
        splits = list(kf.split(combined_V))

        # Get the indices for the specified fold
        train_indices, val_indices = splits[fold-1]

        # Create train and validation datasets for the current fold
        train_dataset = SetDataset(combined_V[train_indices], combined_S[train_indices], self.params, is_train=True,
                                   transform=transform)
        val_dataset = SetDataset(combined_V[val_indices], combined_S[val_indices], self.params, is_train=True,
                                 transform=transform)
        test_dataset = SetDataset(self.V_test, self.S_test, self.params, is_train=True, transform=transform)

        # Create data loaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  shuffle=shuffle_train, num_workers=num_workers,
                                  collate_fn=numpy_collate)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers,
                                collate_fn=numpy_collate)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers,
                                 collate_fn=numpy_collate) if get_test else None
        return train_loader, val_loader, test_loader

class SetDataset(Dataset):
    def __init__(self, V, S, params, is_train=False, transform=None):
        self.data = V
        self.labels = S
        self.is_train = is_train
        self.neg_num = params.neg_num
        self.v_size = params.v_size
        self.transform = transform

    def __getitem__(self, index):
        V = self.data[index]
        S = self.labels[index]

        S_mask = torch.zeros([self.v_size])
        S_mask[S] = 1
        if self.is_train:
            idxs = (S_mask == 0).nonzero(as_tuple=True)[0]
            neg_S = idxs[torch.randperm(idxs.shape[0])[:S.shape[0] * self.neg_num]]
            neg_S_mask = torch.zeros([self.v_size])
            neg_S_mask[S] = 1
            neg_S_mask[neg_S] = 1
            if self.transform:
                V, S_mask, neg_S_mask = self.transform(V), self.transform(S_mask), self.transform(neg_S_mask)
            return V, S_mask, neg_S_mask

        if self.transform:
            V, S_mask = self.transform(V), self.transform(S_mask)
        return V, S_mask

    def __len__(self):
        return len(self.data)

def gen_gaussian_mixture(batch_size):
    scale = 1
    centers = [ (1. / np.sqrt(2), 1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
    centers = [(scale * x, scale * y) for x, y in centers]

    data, labels = [], []
    for i in range(batch_size):
        point = np.random.randn(2) * 0.5
        idx = np.random.randint(2)
        center = centers[idx]
        point[0] += center[0]
        point[1] += center[1]
        data.append(point)
        labels.append(idx)
    data = np.array(data, dtype="float32")
    labels = np.array(labels, dtype="int32")
    
    noise_label = np.random.randint(2)
    noise = data[labels == noise_label]
    data = data[labels == (1 - noise_label)]
    # dataset /= 1.414
    return data, noise

def get_gaussian_mixture_dataset(V_size, S_size):
    V_list, S_list = [], []
    for _ in range(1000):
        data, noise = gen_gaussian_mixture(V_size*4)

        V = data[:V_size]
        S = np.random.choice(list(range(0,V_size)), S_size, replace=False)
        V[S, :] = noise[:S_size]
        
        V_list.append(V)
        S_list.append(S)
    
    V = torch.FloatTensor(V_list)
    S = torch.LongTensor(S_list)

    return V, S
