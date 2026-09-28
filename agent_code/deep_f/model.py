
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CHANNELS, COLS, N_ACTIONS, ROWS


class ResBlock(nn.Module):

    def __init__(self, width, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, width)

    def forward(self, x):
        h = F.relu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.relu(x + h)


class ActorCritic(nn.Module):

    def __init__(self, width=64, blocks=6, in_channels=CHANNELS):
        super().__init__()
        self.width = width
        self.blocks = blocks
        self.in_channels = in_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResBlock(width) for _ in range(blocks)])

        self.pi_conv = nn.Sequential(
            nn.Conv2d(width, 16, 1, bias=False), nn.GroupNorm(4, 16), nn.ReLU(inplace=True))
        self.pi_fc = nn.Linear(16 * COLS * ROWS, N_ACTIONS)

        self.v_conv = nn.Sequential(
            nn.Conv2d(width, 8, 1, bias=False), nn.GroupNorm(4, 8), nn.ReLU(inplace=True))
        self.v_fc = nn.Sequential(
            nn.Linear(8 * COLS * ROWS, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))

        nn.init.orthogonal_(self.pi_fc.weight, gain=0.01)
        nn.init.zeros_(self.pi_fc.bias)

    def forward(self, obs_uint8, action_mask=None):
        x = obs_uint8.float().mul_(1.0 / 255.0) if obs_uint8.dtype == torch.uint8 else obs_uint8
        h = self.body(self.stem(x))
        logits = self.pi_fc(self.pi_conv(h).flatten(1)).float()
        value = self.v_fc(self.v_conv(h).flatten(1)).squeeze(-1).float()
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e9)
        return logits, value

    def config(self):
        return {'width': self.width, 'blocks': self.blocks, 'in_channels': self.in_channels}

    def save(self, path, **extra):
        torch.save({'config': self.config(), 'state_dict': self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path, device='cpu'):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(**ckpt['config'])
        model.load_state_dict(ckpt['state_dict'])
        model.to(device)
        model.eval()
        return model, ckpt
