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

def _ortho_init(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

class OfflinePatchAwareAttention(LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        return parent_parser

    def __init__(
        self,
        input_channels=27,
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
        self.action_dims = (4, 4, 4, 4, 7)
        self.output_channels = sum(self.action_dims)
        self.proj_q = nn.Linear(input_channels, embed_dim, bias)
        self.proj_k = nn.Linear(input_channels, embed_dim, bias)
        self.proj_v = nn.Linear(input_channels, embed_dim, bias)
        self.patch_attention = nn.MultiheadAttention(
            num_heads=num_heads,
            embed_dim=embed_dim,
            batch_first=True
        )
        self.context_dropout = nn.Dropout(context_dropout)
        self.context_norm = nn.LayerNorm(embed_dim, context_norm_eps)
        self.actor = nn.Sequential(
            nn.Linear(input_channels + embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.5), # unstable actor here to regularize encoder
            nn.Linear(embed_dim, self.output_channels),
        )
        self.action_gate = nn.Sequential(
            nn.Linear(input_channels+embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, 6),
            nn.LogSoftmax(dim=-1),
        )
        if ortho_init > 0.:
            self.proj_q.apply(_ortho_init)
            self.proj_k.apply(_ortho_init)
            self.proj_v.apply(_ortho_init)
            self.action_gate.apply(_ortho_init)

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
        context = self.context_dropout(context)
        if self.hparams.context_norm_eps > 0.:
            context = self.context_norm(context)
        context = context.squeeze(1)
        # xxx(okachaiev): concat here requires additional LayerNorm
        x = torch.cat([cell, context], dim=-1)
        g_weights = self.action_gate(x)
        logits = self.actor(x)
        # xxx(okachaiev): log visuals for attention weights (tensorboard supports images)
        return g_weights, logits, attn_W

    # xxx(okachaiev): if I want to add RNN or CNN for global context,
    # this should be very different. in such a case I will need to have
    # batch of (obs, patches, cells, actions). or just (obs, actions)
    # and maintain patches here (otherwise num of items might be different)
    def training_step(self, batch, batch_idx):
        x_cell, x_patch, y_action = batch
        y_g_weights, y_logits, _ = self.forward(x_cell, x_patch)

        # gating loss
        loss_G = F.nll_loss(y_g_weights, y_action[:, 0], reduction='mean')

        self.log("train/gate_nll_loss", loss_G)

        # action prediction loss
        # easier with NLLLoss but I want to regularize entropy
        dists = [Categorical(logits=ls) for ls in torch.split(y_logits, self.action_dims, dim=-1)]
        # as of now, just removing the last component
        # i should either introduce weights, or train separate networks
        log_probs = torch.stack([dist.log_prob(y_action[:, ind+1]) for ind, dist in enumerate(dists)])
        entropy = torch.stack([dist.entropy() for dist in dists[:-1]])

        self.log("train/log_prob", log_probs.mean())
        self.log("train/entropy", entropy.mean())

        # xxx(okachaiev): try out focal loss (as we doing objects detection, basically)
        # return gate_loss.mean(dim=-1) + (-1)*log_probs.mean(dim=-1) + entropy.mean(dim=-1)
        return loss_G

    # xxx(okachaiev): add validation
    # xxx(okachaiev): add F1 and other accuracy metrics

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)


# xxx(okachaiev): option to save/load tensors to avoid waiting for compute each time
class OfflineTrajectoryPatchDataset(Dataset):

    def __init__(self, patch, cell, mask, action):
        self.patch = patch
        self.cell = cell
        self.action = action
        self.mask = mask

    @classmethod
    def load_from_numpy(cls, files, patch=(7,7)):
        obs, mask, action = [], [], []
        for filepath in files:
            data = np.load(filepath)
            obs.extend(data['obs'])
            mask.extend(data['mask'])
            action.extend(data['action'])

        return cls.from_numpy(np.stack(obs), np.stack(mask), np.stack(action), patch=patch)

    @classmethod
    def from_numpy(cls, obs, mask, action, patch=(7,7)):
        # obs [B, H, W, C] -> [B, C, H, W]
        obs = torch.from_numpy(obs).float().permute((0, 3, 1, 2))
        mask = torch.from_numpy(mask)
        action = torch.from_numpy(action).long()
        B, C, H, W = obs.shape
        h, w = patch
        with torch.no_grad():
            kernel = patch_kernel(7, 7)
            # patches: [B, P_num, C, h, w] -> [B*P_num, h*w, C]
            patches = to_patches(obs, kernel, padding=(h//2, w//2))
            patches = patches.reshape((B*H*W , C, h*w)).permute((0, 2, 1))

        cell = obs.permute((0, 2, 3, 1)).reshape((B*H*W, C))
        # action, mask: [B, H*W*C_out] -> [B*H*W, C_out]
        action = action.reshape((B*H*W, -1))
        mask = mask.reshape((B*H*W, -1))

        # this is not exactly the most accurate way to identify
        # which of the performed actions were masked or not.
        # effectively, this only filters out those cases where
        # no actions could be performed at all. not sure why it
        # disregards so many observasions though
        # to do so, we need to compute cat(*one_hot()) @ masks
        # and after that collapse tensor back to sampled values
        non_masked = mask.sum(dim=-1) > 0
        return cls(patches[non_masked], cell[non_masked], action[non_masked], mask[non_masked])

    @classmethod
    def load(cls, filepath):
        data = torch.load(filepath)
        return cls(data['patch'], data['cell'], data['action'], data['mask'])

    def save(self, filepath):
        torch.save({
            "patch": self.patch,
            "cell": self.cell,
            "action": self.action,
            "mask": self.mask,
        }, filepath)

    def __getitem__(self, ind):
        return (self.cell[ind], self.patch[ind], self.action[ind])

    def __len__(self):
        return self.patch.size(0)


# xxx(okachaiev): i need to maintain different compress files for different patches
def load_dataset(folder: Union[str, Path], compressed_filepath:str="compressed.pt") -> Dataset:
    if isinstance(folder, str):
        folder = Path(folder)

    compressed_file = folder.joinpath(compressed_filepath)
    if compressed_file.exists():
        return OfflineTrajectoryPatchDataset.load(str(compressed_file))

    numpy_files = list(folder.glob("*.npz"))
    if numpy_files:
        dataset = OfflineTrajectoryPatchDataset.load_from_numpy(numpy_files)
        dataset.save(str(compressed_file))
        return dataset

    raise ValueError("No dataset files are found (*.npz or *.pt)")


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
# offline_rl/1642390171 with 7* 1_000 steps (114_509 state/action pairs)
# offline_rl/1642390276 with 7*50_000 steps

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = OfflinePatchAwareAttention.add_model_specific_args(parser)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dataset", type=Path, default="offline_rl/1642325330/")
    args = parser.parse_args()

    train_dataset = load_dataset(args.dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    print(f"Loaded dataset, n_rows={len(train_dataset)}")

    model = OfflinePatchAwareAttention(lr=1e-3, context_norm_eps=0., embed_dim=128)
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader)