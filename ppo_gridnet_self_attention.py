from distutils.util import strtobool

import torch
from torch import nn

from stable_baselines3.common.utils import get_device
from stable_baselines3.common.policies import register_policy

from ppo_gridnet_diverse_encode_decode_sb3 import (
    layer_init,
    main,
    make_parser,
    MicroRTSExtractor,
    MicroRTSGridActorCritic,
    parse_arguments,
    Reshape,
    Transpose,
)


class GlobalMultiHeadAttentionEncoder(nn.Module):

    def __init__(
        self,
        input_channels: int,
        embed_dim: int,
        num_heads: int = 1,
        seq_len: int = 256,
        bias: bool = True,
        context_dropout: float = 0.1,
        context_norm_eps: float = 1e-5,
        embed_norm_eps: float = 1e-5,
        embed_dropout: float = 0.1,
        combine_inputs: bool = True
    ):
        super(GlobalMultiHeadAttentionEncoder, self).__init__()
        self.input_channels = input_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.combine_inputs = combine_inputs
        self.proj_q = nn.Linear(input_channels, embed_dim, bias)
        self.proj_k = nn.Linear(input_channels, embed_dim, bias)
        self.proj_v = nn.Linear(input_channels, embed_dim, bias)
        self.attention_block = nn.MultiheadAttention(
            num_heads=num_heads,
            embed_dim=embed_dim,
            batch_first=True
        )
        self.context_dropout = nn.Dropout(context_dropout)
        if context_norm_eps > 0.:
            self.context_norm = nn.LayerNorm(embed_dim, context_norm_eps)
        else:
            self.context_norm = nn.Identity()
        self.proj_embed = nn.Linear(input_channels, embed_dim)
        self.embed_dropout = nn.Dropout(embed_dropout)
        if embed_norm_eps > 0.:
            self.embed_norm = nn.LayerNorm(embed_dim, embed_norm_eps)
        else:
            self.embed_norm = nn.Identity()

    # input: [B, S, I]
    # output: [B, S, E], [B, S, S]
    def forward(self, x):
        batch_size = x.size(0)
        x = x.reshape((batch_size, self.seq_len, self.input_channels))
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        context, attention_weights = self.attention_block(q, k, v)
        context = self.context_norm(self.context_dropout(context))
        if self.combine_inputs:
            x = self.embed_dropout(self.proj_embed(x))
            embed = self.embed_norm(x + context)
        else:
            embed = context
        return embed, attention_weights


class Actor(nn.Module):

    def __init__(self, output_channels, hidden_dim=32, num_cells=256):
        super(Actor, self).__init__()

        self.output_channels = output_channels
        self.hidden_dim = hidden_dim
        self.num_cells = num_cells
        self.actor = nn.Linear(self.hidden_dim, self.output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.actor(x)
        return x.reshape((batch_size, self.output_channels*self.num_cells))


class MicroRTSExtractorSelfAttention(MicroRTSExtractor):

    def __init__(
        self,
        input_channels: int = 27,
        output_channels: int = 78,
        action_space_size: int = 78*256,
        actor_hidden_dim: int = 32,
        attn_num_heads: int = 1,
        attn_bias: bool = True,
        attn_context_dropout: float = 0.1,
        attn_context_norm_eps: float = 1e-5,
        attn_embed_dropout: float = 0.1,
        attn_embed_norm_eps: float = 1e-5,
        attn_combine_inputs: bool = True,
        device: str = "auto"
    ):
        super(MicroRTSExtractorSelfAttention, self).__init__()

        self.latent_dim_pi = action_space_size
        self.latent_dim_vf = actor_hidden_dim*256

        self.device = get_device(device)

        self.latent_net = GlobalMultiHeadAttentionEncoder(
            input_channels,
            embed_dim=actor_hidden_dim,
            num_heads=attn_num_heads,
            seq_len=256,
            bias=attn_bias,
            # switch off all complications to get baseline
            context_dropout=attn_context_dropout,
            context_norm_eps=attn_context_norm_eps,
            embed_dropout=attn_embed_dropout,
            embed_norm_eps=attn_embed_norm_eps,
            combine_inputs=attn_combine_inputs,
        ).to(self.device)
        self.policy_net = Actor(output_channels, hidden_dim=actor_hidden_dim).to(self.device)
        self.value_net = nn.Flatten(start_dim=1)

    def forward(self, features):
        obs, masks = features
        latent_context, _ = self.latent_net(obs)
        return self._mask_action_logits(self.policy_net(latent_context), masks), self.value_net(latent_context)

    def forward_actor(self, features):
        obs, masks = features
        latent_context, _ = self.latent_net(obs)
        return self._mask_action_logits(self.policy_net(latent_context), masks)

    def forward_critic(self, features):
        obs, _ = features
        latent_context, _ = self.latent_net(obs)
        return self.value_net(latent_context)


class MicroRTSGridSelfAttention(MicroRTSGridActorCritic):

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MicroRTSExtractorSelfAttention(
            input_channels=self.input_channels,
            output_channels=self.action_plane.sum(),
            action_space_size=self.action_space.nvec.sum(),
            actor_hidden_dim=self.hparams['actor_hidden_dim'],
            attn_num_heads=self.hparams['attn_num_heads'],
            attn_bias=self.hparams['attn_bias'],
            attn_context_dropout=self.hparams['attn_context_dropout'],
            attn_context_norm_eps=self.hparams['attn_context_norm_eps'],
            attn_embed_dropout=self.hparams['attn_embed_dropout'],
            attn_embed_norm_eps=self.hparams['attn_embed_norm_eps'],
            attn_combine_inputs=self.hparams['attn_combine_inputs'],
        )


if __name__ == "__main__":
    register_policy('MicroRTSGridActorCritic', MicroRTSGridSelfAttention)

    parser = make_parser()
    parser.add_argument('--actor-hidden-dim', type=int, default=32,
                        help="hidden layer for linear actor network")
    parser.add_argument('--attn-num-heads', type=int, default=1,
                        help="number of attention heads")
    parser.add_argument('--attn-bias', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help="weather attention projections should be biased")
    parser.add_argument('--attn-context-dropout', type=float, default=0.1,
                        help="dropout for attention context (0.0 to disable)")
    parser.add_argument('--attn-context-norm-eps', type=float, default=1e-5,
                        help="layer norm eps for attention context (0.0 to disable)")
    parser.add_argument('--attn-embed-dropout', type=float, default=0.1,
                        help="dropout for embedding layer in attention encoder (0.0 to disable)")
    parser.add_argument('--attn-embed-norm-eps', type=float, default=1e-5,
                        help="layer norm eps for embedding layer in attention encoder (0.0 to disable)")
    parser.add_argument('--attn-combine-inputs', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help="weather attention inputs should be added to the context")

    args = parse_arguments(parser)

    hparams = dict(
        actor_hidden_dim=args.actor_hidden_dim,
        attn_num_heads=args.attn_num_heads,
        attn_bias=args.attn_bias,
        attn_context_dropout=args.attn_context_dropout,
        attn_context_norm_eps=args.attn_context_norm_eps,
        attn_embed_dropout=args.attn_embed_dropout,
        attn_embed_norm_eps=args.attn_embed_norm_eps,
        attn_combine_inputs=args.attn_combine_inputs,
    )
    main(args, policy_hparams=hparams)