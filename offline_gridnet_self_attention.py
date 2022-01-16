from argparse import ArgumentParser
import numpy as np
import torch
from torch import nn
from torch.distribution.categorical import Categorical
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


def load_dataset(folder: Path) -> Dataset:
    obs, actions = [], []
    with (
        folder.joinpath("obs.npy").open("rb") as obs_fd,
        folder.joinpath("actions.npy").open("rb") as action_fd
    ):
        while True:
            try:
                obs.append(np.load(obs_fd))
                actions.append(np.load(actions_fd))
            except Exception as e:
                print(e)
                break

    obs = np.concatenate(obs, axis=0)
    actions = np.concatenate(actions, axis=0)
    return TensorDataset(obs, actions)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = OfflineSelfAttention.add_model_specific_args(parser)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dataset", type=Path, default="offline_rl/1642325330/")
    args = parser.parse_args()

    train_dataset = load_dataset(parser.dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    print(f"Loaded dataset, n_rows={len(train_dataset)}")

    model = OfflineSelfAttention()
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader)