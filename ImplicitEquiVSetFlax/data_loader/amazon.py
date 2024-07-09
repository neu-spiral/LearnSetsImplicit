import os
import csv
import gdown
import torch
import zipfile
import torch.nn.functional as F
from torch.utils.data import Subset, Dataset, DataLoader, ConcatDataset
import numpy as np
from utils.flax_helper import read_from_pickle, find_not_in_set
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

    def gen_datasets(self):
        raise NotImplementedError

    def get_loaders(self, batch_size, num_workers, shuffle_train=False,
                    get_test=True):
        raise NotImplementedError


class Amazon(Data):
    def __init__(self, params):
        super().__init__(params)
        data_root = self.download_amazon()

        torch.manual_seed(1)  # fix dataset
        self.gen_datasets(data_root)

    def download_amazon(self):
        url = 'https://drive.google.com/uc?id=1OLbCOTsRyowxw3_AzhxJPVB8VAgjt2Y6'
        data_root = 'dataset/amazon'
        download_path = f'{data_root}/amazon_baby_registry.zip'
        if not os.path.exists(data_root):
            os.makedirs(data_root)
        if not os.listdir(data_root):
            gdown.download(url, download_path, quiet=False)
            with zipfile.ZipFile(download_path, 'r') as ziphandler:
                ziphandler.extractall(data_root)
        return data_root

    def read_real_data(self, data_root):
        pickle_filename = data_root + '/' + self.params.amazon_cat
        dataset_ = read_from_pickle(pickle_filename)
        for i in range(len(dataset_)):
            dataset_[i + 1] = torch.tensor(dataset_[i + 1])
        data_ = torch.zeros(len(dataset_), dataset_[1].shape[0])
        for i in range(len(dataset_)):
            data_[i, :] = dataset_[i + 1]

        csv_filename = data_root + '/' + self.params.amazon_cat + '.csv'
        with open(csv_filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC, quotechar='|')
            S = {}
            i = -1
            for row in reader:
                i = i + 1
                S[i] = torch.tensor([int(row[x]) for x in range(len(row))]).long()
        return data_, S

    def filter_S(self, data, S):
        S_list = []
        V_list = []
        for i in range(len(S)):
            if 2 < S[i].shape[0] < self.params.v_size:
                Svar = S[i] - 1  # index starts from 0
                sub_set, ground_set = self.construct_ground_set(data, Svar, V=self.params.v_size)
                S_list.append(sub_set)
                V_list.append(ground_set)
        S = S_list
        U = V_list
        return U, S

    def construct_ground_set(self, data, S, V):
        S_data = data[S]
        S_mean = S_data.mean(dim=0).unsqueeze(0)
        UnotS_data = find_not_in_set(data, S)
        S_mean_norm = F.normalize(S_mean, dim=-1)
        UnotS_data_norm = F.normalize(UnotS_data, dim=-1)

        cos_sim = (S_mean_norm @ UnotS_data_norm.T).squeeze(0)
        _, idx = torch.sort(cos_sim)
        UnotS_idx = idx[:V - S.shape[0]]
        UnotS_data = UnotS_data[UnotS_idx]

        S = torch.randperm(V)[:S.shape[0]]
        UnotS_idx = torch.ones(V, dtype=bool)
        UnotS_idx[S] = False

        U = torch.zeros([V, data.shape[-1]])
        U[S] = S_data
        U[UnotS_idx] = UnotS_data

        return S, U

    def split_into_training_test(self, data_mat, S):
        folds = [0.33, 0.33, 0.33]
        num_elem = len(data_mat)
        tr_size = int(folds[0] * num_elem)
        dev_size = int((folds[1] + folds[0]) * num_elem)
        test_size = num_elem

        V_train = data_mat[0:tr_size]
        V_dev = data_mat[tr_size:dev_size]
        V_test = data_mat[dev_size:test_size]

        S_train = S[0:tr_size]
        S_dev = S[tr_size:dev_size]
        S_test = S[dev_size:test_size]

        V_sets = (V_train, V_dev, V_test)
        S_sets = (S_train, S_dev, S_test)
        return V_sets, S_sets

    def gen_datasets(self, data_root):
        data, S = self.read_real_data(data_root)
        data, S = self.filter_S(data, S)
        V_sets, S_sets = self.split_into_training_test(data, S)

        self.V_train, self.V_val, self.V_test = V_sets
        self.S_train, self.S_val, self.S_test = S_sets

        self.fea_size = self.V_train[0].shape[-1]

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
        train_indices, val_indices = splits[fold]

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
