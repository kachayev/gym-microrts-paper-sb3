import torch
from torch import nn

from stable_baselines3.common.utils import get_device
from stable_baselines3.common.policies import register_policy

from ppo_gridnet_diverse_encode_decode_sb3 import (
    layer_init,
    main,
    MicroRTSExtractor,
    MicroRTSGridActorCritic,
    parse_arguments,
    Reshape,
    Transpose,
)


class Encoder(nn.Module):

    def __init__(self, input_channels):
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
        x = x / torch.linalg.norm(x, 2, -1, True)
        return x


class MicroRTSExtractorLinearCritic(MicroRTSExtractor):

    def __init__(self, input_channels=27, output_channels=78, action_space_size=None, device = "auto"):
        super(MicroRTSExtractorLinearCritic, self).__init__()

        self.latent_dim_pi = action_space_size
        self.latent_dim_vf = 256

        self.device = get_device(device)

        self.latent_net = Encoder(input_channels).to(self.device)

        self.policy_net = nn.Sequential(
            Reshape((-1, 256, 1, 1)),
            layer_init(nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(32, output_channels, 3, stride=2, padding=1, output_padding=1)),
            Transpose((0, 2, 3, 1)),
            Reshape((-1, action_space_size))
        ).to(self.device)

        self.value_net = nn.Identity()


class MicroRTSGridActorLinearCritic(MicroRTSGridActorCritic):

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MicroRTSExtractorLinearCritic(
            input_channels=self.input_channels,
            output_channels=self.action_plane.sum(),
            action_space_size=self.action_space.nvec.sum(),
        )


if __name__ == "__main__":
    register_policy('MicroRTSGridActorCritic', MicroRTSGridActorLinearCritic)

    args = parse_arguments()

    main(args)