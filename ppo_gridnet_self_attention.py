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


class Encoder(nn.Module):

    def __init__(self, input_channels: int, hidden_dim: int, n_heads: int = 1):
        super(Encoder, self).__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.fc_q = nn.Linear(input_channels, hidden_dim)
        self.fc_k = nn.Linear(input_channels, hidden_dim)
        self.fc_v = nn.Linear(input_channels, hidden_dim)
        self.attention_block = nn.MultiheadAttention(num_heads=n_heads, embed_dim=hidden_dim, batch_first=True)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.reshape((batch_size, 256, self.input_channels))
        q = self.fc_q(x)
        k = self.fc_k(x)
        v = self.fc_v(x)
        att, _ = self.attention_block(q, k, v)
        return att


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
        n_heads: int = 1,
        device: str = "auto"
    ):
        super(MicroRTSExtractorSelfAttention, self).__init__()

        self.latent_dim_pi = action_space_size
        self.latent_dim_vf = actor_hidden_dim*256

        self.device = get_device(device)

        self.latent_net = Encoder(input_channels, hidden_dim=actor_hidden_dim, n_heads=n_heads).to(self.device)
        self.policy_net = Actor(output_channels, hidden_dim=actor_hidden_dim).to(self.device)
        self.value_net = nn.Flatten(start_dim=1)


class MicroRTSGridSelfAttention(MicroRTSGridActorCritic):

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MicroRTSExtractorSelfAttention(
            input_channels=self.input_channels,
            output_channels=self.action_plane.sum(),
            action_space_size=self.action_space.nvec.sum(),
            actor_hidden_dim=self.hparams['actor_hidden_dim'],
            n_heads=self.hparams['n_heads'],
        )


if __name__ == "__main__":
    register_policy('MicroRTSGridActorCritic', MicroRTSGridSelfAttention)

    parser = make_parser()
    parser.add_argument('--actor-hidden-dim', type=int, default=32,
                        help="Hidden layer for linear actor network")
    parser.add_argument('--n-heads', type=int, default=1,
                        help="Number of attention heads")
    args = parse_arguments(parser)

    hparams = dict(actor_hidden_dim=args.actor_hidden_dim, n_heads=args.n_heads)
    main(args, policy_hparams=hparams)