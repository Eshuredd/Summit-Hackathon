"""One tiny high-level multimodal classifier; no robot control outputs."""

import torch
from torch import nn

from bimind.policy_features import LABELS, ROBOT_FEATURES, VOCABULARY


class TinyPolicy(nn.Module):
    """Fuse a tiny RGB CNN, command bag of words, and robot feedback flags."""

    def __init__(self):
        """Construct the fixed architecture used for every training run."""
        super().__init__()
        self.image = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 12, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(4),
            nn.Flatten(),
        )
        self.text = nn.Sequential(nn.Linear(len(VOCABULARY), 8), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(16 * 3 * 3 + 8 + len(ROBOT_FEATURES), 24),
            nn.ReLU(),
            nn.Linear(24, len(LABELS)),
        )

    def forward(self, rgb, instruction, robot_state):
        """Return seven skill logits from the three input modalities.

        Args:
            rgb: Float RGB tensor of shape N,3,96,96.
            instruction: Normalized vocabulary counts of shape N,V.
            robot_state: Six boolean feedback features represented as floats.

        Returns:
            Unnormalized skill logits of shape N,7.
        """
        return self.head(torch.cat((self.image(rgb), self.text(instruction), robot_state), dim=1))
