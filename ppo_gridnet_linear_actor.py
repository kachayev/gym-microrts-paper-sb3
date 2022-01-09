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

    def __init__(self, input_channels: int):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 128, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
            layer_init(nn.Conv2d(128, 256, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

    def forward(self, x):
        x = x.permute((0,3,1,2))
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        return x


class Actor(nn.Module):
    
    def __init__(self, output_channels, hidden_dim=32, num_cells=256):
        super(Actor, self).__init__()

        self.output_channels = output_channels
        self.hidden_dim = hidden_dim
        self.num_cells = num_cells
        self.network = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_channels)
        )

    def forward(self, x):
        x = self.network(x.unsqueeze(-1))
        x = x.reshape((-1, self.output_channels*self.num_cells))
        return x


class MicroRTSExtractorLinearActor(MicroRTSExtractor):

    def __init__(
        self,
        input_channels: int = 27,
        output_channels: int = 78,
        encoder_norm_type: float = 2.,
        action_space_size: int = 78*256,
        actor_hidden_dim: int = 32,
        device: str = "auto"
    ):
        super(MicroRTSExtractorLinearActor, self).__init__()

        self.latent_dim_pi = action_space_size
        self.latent_dim_vf = 256

        self.device = get_device(device)

        self.latent_net = Encoder(input_channels).to(self.device)
        self.policy_net = Actor(output_channels, hidden_dim=actor_hidden_dim).to(self.device)
        self.value_net = nn.Identity()


class MicroRTSGridActorLinearActor(MicroRTSGridActorCritic):

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MicroRTSExtractorLinearActor(
            input_channels=self.input_channels,
            output_channels=self.action_plane.sum(),
            action_space_size=self.action_space.nvec.sum(),
            encoder_norm_type=self.hparams['encoder_norm'],
            actor_hidden_dim=self.hparams['actor_hidden_dim'],
        )


if __name__ == "__main__":
    register_policy('MicroRTSGridActorCritic', MicroRTSGridActorLinearActor)

    parser = make_parser()
    parser.add_argument('--encoder-norm', type=float, default=0.,
                        help="Lp norm for CNN embeddings, setting to 0 skips normalization")
    parser.add_argument('--actor-hidden-dim', type=int, default=32,
                        help="Hidden layer for linear actor network")
    args = parse_arguments(parser)

    hparams = dict(
        encoder_norm=args.encoder_norm,
        actor_hidden_dim=args.actor_hidden_dim,
    )
    main(args, policy_hparams=hparams)