from argparse import ArgumentParser
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from pytorch_lightning import LightningModule, Trainer


class OfflineSelfAttention(LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        return parent_parser

    def __init__(
        self,
        input_channels=27,
        output_channels=78,
        embed_dim=64,
        bias=True,
        num_heads=1,
        context_dropout=0.1,
        context_norm_eps=1e-5,
        ortho_init=np.sqrt(2),
        lr=1e-4
    ):
        super().__init__()
        self.save_hyperparameters()
        self.action_dims = (6, 4, 4, 4, 4, 7, 49)
        self.proj_q = nn.Linear(input_channels, embed_dim, bias)
        self.proj_k = nn.Linear(input_channels, embed_dim, bias)
        self.proj_v = nn.Linear(input_channels, embed_dim, bias)
        self.patch_attention = nn.MultiheadAttention(
            num_heads=num_heads,
            embed_dim=embed_dim,
            batch_first=True
        )
        self.context_dropout = nn.Dropout(context_dropout)
        if context_norm_eps > 0.:
            self.context_norm = nn.LayerNorm(embed_dim, context_norm_eps)
        else:
            self.context_norm = nn.Identity()
        self.actor = nn.Linear(input_channels + embed_dim, output_channels)
        if ortho_init > 0.:
            self._ortho_init(ortho_init)

    def _ortho_init(self, gain=np.sqrt(2)):
        nn.init.orthogonal_(self.proj_q.weight.data, gain)
        nn.init.orthogonal_(self.proj_k.weight.data, gain)
        nn.init.orthogonal_(self.proj_v.weight.data, gain)

    def forward(self, cell, patch):
        """Inpt dims:
         * cell: [B, C]
         * patch: [B, P_s, C]

        The algorithm works as the following:

        - each cell in the patch generates K and V
        - central cell generates Q
        - attention applied to a single Q (repeated to match the shape)

        Future considerations:

        It might be interesting to see if we can (or need) to have different
        attention weights for choosing between different actions. Intuitivly,
        attention necessary for a range unit is different from others.
        """
        q = self.proj_q(cell).unsqueeze(1)
        k, v = self.proj_k(patch), self.proj_v(patch)
        # xxx(okachaiev): I guess I need to put mask on the central cell
        # of the patch, so it cannot be used as a part of attention
        context, attn_W = self.patch_attention(q, k, v)
        context = self.context_norm(self.context_dropout(context))
        context = context.squeeze(1)
        # xxx(okachaiev): concat here requires additional LayerNorm
        logits = self.actor(torch.cat([cell, context], dim=-1))
        return logits, attn_W

    # xxx(okachaiev): if I want to add RNN for global context,
    # this should be very different
    def training_step(self, batch, batch_idx):
        x_cell, x_patch, y_action = batch
        y_logits, _ = self.forward(x_cell, x_patch)
        # easier with NLLLoss but I want to regularize entropy
        dists = [Categorical(logits=ls) for ls in torch.split(y_logits, self.action_dims, dim=-1)]
        log_probs = torch.stack([dist.log_prob(y_action[:, ind]) for ind, dist in enumerate(dists)])
        entropy = torch.stack([dist.entropy() for dist in dists])

        self.log("train/log_prob", log_probs.detach().mean())
        self.log("train/entropy", entropy.detach().mean())

        # xxx(okachaiev): try out focal loss (as we doing objects detection, basically)
        return (-log_probs+entropy).mean()

    # xxx(okachaiev): add validation
    # xxx(okachaiev): add F1 and other accuracy metrics

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)


# xxx(okachaiev): filter out empty cells, they don't matter
class OfflineTrajectoryPatchDataset(Dataset):

    EMPTY_CELL_IND = 13

    def __init__(self, obs, mask, action, patch=(7,7)):
        # obs [B, H, W, C] -> [B, C, H, W]
        obs = torch.from_numpy(obs).float().permute((0, 3, 1, 2))
        # mask = torch.from_numpy(mask)
        action = torch.from_numpy(action).long()
        B, C, H, W = obs.shape
        h, w = patch
        with torch.no_grad():
            kernel = patch_kernel(7, 7)
            # patches: [B, P_num, C, h, w] -> [B*P_num, h*w, C]
            patches = to_patches(obs, kernel, padding=(h//2, w//2))
            patches = patches.reshape((B*H*W , C, h*w)).permute((0, 2, 1))

        cells = obs.permute((0, 2, 3, 1)).reshape((B*H*W, C))
        non_empty = cells[:,self.EMPTY_CELL_IND] == 0.
        self.obs = patches[non_empty]
        self.cells = cells[non_empty]
        # action: [B, H*W*C_out] -> [B*H*W, C_out]
        self.action = action.reshape((B*H*W, -1))[non_empty]

    def __getitem__(self, ind):
        return (self.cells[ind], self.obs[ind], self.action[ind])

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

    return OfflineTrajectoryPatchDataset(np.stack(obs), np.stack(mask), np.stack(action))


def patch_kernel(H: int, W: int, eps: float = 1e-6) -> torch.Tensor:
    """Creates a binary kernel to extract the patches"""
    window = H*W
    kernel = torch.zeros(window, window) + torch.eye(window) + eps
    return kernel.view(window, 1, H, W)

def to_patches(
        x: torch.Tensor,
        patch: Union[torch.Tensor, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        padding: Optional[Union[int, Tuple[int, int]]] = 0
    ) -> torch.Tensor:
    """Input shape:
      * x is (B, C, H, W)
      * kernel is (h*w, 1, h, w)

    Output shape: (B, num_patches, C, h, w)

    Example usage:

        >>> x = torch.arange(25.).view(1, 1, 5, 5)
        >>> x.shape
        torch.Size([1, 1, 5, 5])
        >>> patches = to_patches(x, (3,3), padding=(1,1))
        >>> patches.shape
        torch.Size([1, 25, 1, 3, 3])
    """
    B, C, H, W = x.shape

    if isinstance(patch, tuple):
        h, w = patch
        patch = patch_kernel(h, w)
    else:
        h, w = patch.size(-2), patch.size(-1)
    kernel = patch.repeat(C, 1, 1, 1).to(x.device).to(x.dtype)
    y = F.conv2d(x, kernel, stride=stride, padding=padding, groups=C)
    return y.view(B, C, h, w, -1).permute(0, 4, 1, 2, 3)


# good dataset to practice on:
# offline_rl/1642390171 with 7* 1_000 steps
# offline_rl/1642390276 with 7*50_000 steps

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = OfflineSelfAttention.add_model_specific_args(parser)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dataset", type=Path, default="offline_rl/1642325330/")
    args = parser.parse_args()

    train_dataset = load_dataset(args.dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    print(f"Loaded dataset, n_rows={len(train_dataset)}")

    model = OfflineSelfAttention()
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader)