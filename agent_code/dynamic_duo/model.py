import torch
import torch.nn as nn
import torch.nn.functional as F


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


class PolicyNet(nn.Module):
    def __init__(self, channels=12, width=64, blocks=6, size=17, n_actions=6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(channels, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResBlock(width) for _ in range(blocks)])
        self.pi_conv = nn.Sequential(
            nn.Conv2d(width, 16, 1, bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
        )
        self.pi_fc = nn.Linear(16 * size * size, n_actions)

    def forward(self, obs, mask):
        x = obs.float().mul_(1.0 / 255.0)
        h = self.body(self.stem(x))
        logits = self.pi_fc(self.pi_conv(h).flatten(1)).float()
        return logits.masked_fill(mask == 0, -1e9)


def load_policy(path):
    net = PolicyNet()
    net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    net.eval()
    return net
