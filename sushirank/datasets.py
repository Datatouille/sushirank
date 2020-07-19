import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PointwiseDataset(Dataset):
    def __init__(
        self,
        df,
        label_col,
        cat_cols,
        num_cols,
    ):
        self.label_col = label_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.df = df
        self.cat_dims = [self.df[i].nunique() for i in self.cat_cols]
        self.features = []
        self._build()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]
    
    def _build(self):
        cat_features = torch.tensor(self.df[self.cat_cols].values,dtype=torch.long).to(device)
        num_features = torch.tensor(self.df[self.num_cols].values,dtype=torch.float).to(device)
        label = torch.tensor(self.df[self.label_col].values,dtype=torch.float).to(device)
        for i in tqdm(range(self.df.shape[0])):
            feat = {
                'cat_feature': cat_features[i],
                'num_feature': num_features[i],
                'label': label[i]
            }
            self.features.append(feat)
            
class PairwiseDataset(Dataset):
    def __init__(
        self,
        df,
        label_col,
        cat_cols,
        num_cols,
    ):
        self.label_col = label_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.df = df
        self.cat_dims = [self.df[i].nunique() for i in self.cat_cols]
        self.features = []
        self._build()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        feat_i =  self.features[i]
        feat_j = self.features[np.random.randint(len(self.features))]
        return {
            'cat_feature_i': feat_i['cat_feature'],
            'num_feature_i': feat_i['num_feature'],
            'cat_feature_j': feat_j['cat_feature'],
            'num_feature_j': feat_j['num_feature'],
            'label': torch.tensor([int(feat_i['label'] > feat_j['label'])],dtype=torch.float).to(device)
        }
    
    def _build(self):
        cat_features = torch.tensor(self.df[self.cat_cols].values,dtype=torch.long).to(device)
        num_features = torch.tensor(self.df[self.num_cols].values,dtype=torch.float).to(device)
        label = self.df[self.label_col].values
        for i in tqdm(range(self.df.shape[0])):
            feat = {
                'cat_feature': cat_features[i],
                'num_feature': num_features[i],
                'label': label[i]
            }
            self.features.append(feat)