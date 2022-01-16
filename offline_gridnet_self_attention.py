from argparse import ArgumentParser
import numpy as np
from pathlib import Path
from typing import Union

import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim import Adam
from pytorch_lightning import LightningModule, Trainer

class OfflineSelfAttention(LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        return parent_parser

    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.attention_block = None
    
    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch
        patch_x = None # xxx(okachaiev): todo
        y_hat = self.forward(patch_x)
        log_probs = torch.stack([dist.log_prob(action) for dist, action in zip(y_hat, y)])
        entropy = torch.stack([dist.entropy() for dist in y_hat])
        loss = -log_probs + entropy
        # xxx(okachaiev): try out focal loss (as we doing objects detection, basically)
        return loss.sum(dim=-1)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)


class OfflineTrajDataset(Dataset):

    def __init__(self, obs, mask, action):
        self.obs = torch.from_numpy(obs)
        print(self.obs.shape)
        self.mask = torch.from_numpy(mask)
        self.action = torch.from_numpy(action)

    def __getitem__(self, ind):
        return (self.obs[ind], self.mask[ind], self.action[ind])

    def __len__(self):
        return self.obs.size(0)


def load_dataset(folder: Union[str, Path], max_items:int=0) -> Dataset:
    if isinstance(folder, str):
        folder = Path(folder)

    obs, mask, action = [], [], []
    num_loaded = 0
    for filepath in folder.glob("*.npz"):
        if not max_items or num_loaded < max_items:
            dataset = np.load(filepath)
            obs.extend(dataset['obs'])
            mask.extend(dataset['mask'])
            action.extend(dataset['action'])
            num_loaded += len(dataset['obs'])

    return OfflineTrajDataset(np.stack(obs), np.stack(mask), np.stack(action))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = OfflineSelfAttention.add_model_specific_args(parser)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dataset", type=Path, default="offline_rl/1642325330/")
    args = parser.parse_args()

    train_dataset = load_dataset(args.dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    print(f"Loaded dataset, n_rows={len(train_dataset)}")

    model = OfflineSelfAttention()
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader)