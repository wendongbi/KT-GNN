# my own pyg dataset class
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.transforms import ToUndirected
import shutil
import os
import numpy as np



    
def build_dataset(dataset_name, split='random', split_ratio=[0.6,0.2,0.2], remove_unobserved_feats=False):
    if dataset_name == 'twitter':
        root = '../datasets/twitter_observed' if remove_unobserved_feats else '../datasets/twitter'
        dataset = Twitter(root=root, dataset='twitter', transform=None, pre_transform=None,
                split=split, train_val_test_ratio=split_ratio, remove_unobserved_feats=remove_unobserved_feats)
        # dataset = Twitter(root='../datasets/twitter_add', dataset='twitter', transform=None, pre_transform=None,
        #         split=split, train_val_test_ratio=split_ratio)
    elif dataset_name == 'company':
        root = '../datasets/company_observed' if remove_unobserved_feats else '../datasets/company'
        dataset = Company(root=root, dataset='company', transform=None, pre_transform=None,
                split=split, train_val_test_ratio=split_ratio, remove_unobserved_feats=remove_unobserved_feats)
    else:
        assert NotImplementedError('Not implemented dataset:{}'.format(dataset_name))
    return dataset

# Twitter dataset
class Twitter(InMemoryDataset):
    def __init__(self, root='./datasets/twitter', dataset='twitter', transform=None, pre_transform=None,
                split=None, train_val_test_ratio=[0.6,0.2,0.2], remove_unobserved_feats=False): # 0.4,0.3,0.3 | 0.2,0.4,0.4 | 0.6,0.2,0.2
        self.root = root
        self.dataset = dataset
        self.train_val_test_ratio = train_val_test_ratio
        self.remove_unobserved_feats = remove_unobserved_feats
        super().__init__(root, transform, pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
        ToUndirected(merge=True)(self.data)
        if split != None:
            self.split_(split)

    @property
    def raw_file_names(self):
        return ['X.npy', 'Y.npy', 'central_mask.npy', 'edge_index.npy']

    @property
    def processed_file_names(self):
        return ['data.pt']
    def split_(self, split):
        # valid check, -1 is not included in the num classes
        assert self.num_classes == max(self.data.y) + 1
        data = self.get(0)
        lbl_num = data.y.shape[0]
        data.train_mask = torch.BoolTensor([False] * lbl_num)
        data.val_mask = torch.BoolTensor([False] * lbl_num)
        data.test_mask = torch.BoolTensor([False] * lbl_num)
        if split == 'random':
            if self.train_val_test_ratio is None:
                print('split ratio is None')
                pass
            else:
                for c in range(self.num_classes):
                    idx = ((data.y == c) * (~data.central_mask)).nonzero(as_tuple=False).view(-1)
                    num_class = len(idx)
                    num_train_per_class = int(np.ceil(num_class * self.train_val_test_ratio[0]))
                    num_val_per_class = int(np.floor(num_class * self.train_val_test_ratio[1]))
                    num_test_per_class = num_class - num_train_per_class - num_val_per_class
                    print('[Class:{}] Train:{} | Val:{} | Test:{}'.format(c, num_train_per_class, num_val_per_class, num_test_per_class))
                    assert num_test_per_class >= 0
                    idx_perm = torch.randperm(idx.size(0))
                    idx_train = idx[idx_perm[:num_train_per_class]]
                    idx_val = idx[idx_perm[num_train_per_class:num_train_per_class+num_val_per_class]]
                    idx_test = idx[idx_perm[num_train_per_class+num_val_per_class:]]
                    data.train_mask[idx_train] = True
                    data.val_mask[idx_val] = True
                    data.test_mask[idx_test] = True
                data.train_mask[data.central_mask * (data.y!=-1)] = True
                self.data, self.slices = self.collate([data])

    def process(self):
        # Read data into huge `Data` list.
        
        x = torch.from_numpy(np.load(os.path.join(self.raw_dir, self.raw_file_names[0]))).float()
        if self.remove_unobserved_feats:
            x = x[:, :300]
        y = torch.from_numpy(np.load(os.path.join(self.raw_dir, self.raw_file_names[1]))).long()
        central_mask = torch.from_numpy(np.load(os.path.join(self.raw_dir, self.raw_file_names[2]))).bool()
        edge_index = torch.from_numpy(np.load(os.path.join(self.raw_dir, self.raw_file_names[3]))).long()

        data = torch_geometric.data.Data(x=x, edge_index=edge_index, y=y)
        data.central_mask = central_mask
        data_list = [data]
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


    
# Company dataset
class Company(InMemoryDataset):
    def __init__(self, root='./datasets/company', dataset='company', transform=None, pre_transform=None,
                split=None, train_val_test_ratio=[0.6,0.2,0.2], remove_unobserved_feats=False): # 0.4,0.3,0.3 | 0.2,0.4,0.4 | 0.6,0.2,0.2
        # dim_x_o: 33
        # dim_x_u: 78
        self.root = root
        self.dataset = dataset
        self.train_val_test_ratio = train_val_test_ratio
        self.remove_unobserved_feats = remove_unobserved_feats
        super().__init__(root, transform, pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
        ToUndirected(merge=True)(self.data)
        if split != None:
            self.split_(split)

    @property
    def raw_file_names(self):
        return ['X.npy', 'Y.npy', 'central_mask.npy', 'edge_index.npy']

    @property
    def processed_file_names(self):
        return ['data.pt']
    def split_(self, split):
        # valid check, -1 is not included in the num classes
        assert self.num_classes == max(self.data.y) + 1
        data = self.get(0)
        lbl_num = data.y.shape[0]
        data.train_mask = torch.BoolTensor([False] * lbl_num)
        data.val_mask = torch.BoolTensor([False] * lbl_num)
        data.test_mask = torch.BoolTensor([False] * lbl_num)
        if split == 'random':
            if self.train_val_test_ratio is None:
                print('split ratio is None')
                pass
            else:
                for c in range(self.num_classes):
                    idx = ((data.y == c) * (~data.central_mask)).nonzero(as_tuple=False).view(-1)
                    num_class = len(idx)
                    num_train_per_class = int(np.ceil(num_class * self.train_val_test_ratio[0]))
                    num_val_per_class = int(np.floor(num_class * self.train_val_test_ratio[1]))
                    num_test_per_class = num_class - num_train_per_class - num_val_per_class
                    print('[Class:{}] Train:{} | Val:{} | Test:{}'.format(c, num_train_per_class, num_val_per_class, num_test_per_class))
                    assert num_test_per_class >= 0
                    idx_perm = torch.randperm(idx.size(0))
                    idx_train = idx[idx_perm[:num_train_per_class]]
                    idx_val = idx[idx_perm[num_train_per_class:num_train_per_class+num_val_per_class]]
                    idx_test = idx[idx_perm[num_train_per_class+num_val_per_class:]]
                    data.train_mask[idx_train] = True
                    data.val_mask[idx_val] = True
                    data.test_mask[idx_test] = True
                data.train_mask[data.central_mask * (data.y!=-1)] = True
                self.data, self.slices = self.collate([data])

    def process(self):
        # Read data into huge `Data` list.
        
        x = torch.from_numpy(np.load(os.path.join(self.raw_dir, self.raw_file_names[0]))).float()
        if self.remove_unobserved_feats:
            x = x[:, :33]
        y = torch.from_numpy(np.load(os.path.join(self.raw_dir, self.raw_file_names[1]))).long()
        central_mask = torch.from_numpy(np.load(os.path.join(self.raw_dir, self.raw_file_names[2]))).bool()
        edge_index = torch.from_numpy(np.load(os.path.join(self.raw_dir, self.raw_file_names[3]))).long()

        data = torch_geometric.data.Data(x=x, edge_index=edge_index, y=y)
        data.central_mask = central_mask
        data_list = [data]
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
