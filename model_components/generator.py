import torch 
import torch.nn as nn

class AttackNetwork(nn.Module):
    """Generates differentiable distortions to maximize the message loss"""
    def __init__(
        self,
        epsilon: float=0.06,
        capped: bool=False
    ):
        super().__init__()
        self.epsilon = epsilon
        self.capped = capped

        self.G_adv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=5, padding=2)
        )

    def forward(self, I_en: torch.Tensor,) -> torch.Tensor:
        """Accepts image encoded with message (I_en) and outputs distorted image I_adv"""
        if self.capped:
            # G_adv(I) = I + eplison * tanh(CNN(I))
            return I_en + self.epsilon * torch.tanh(self.G_adv(I_en))

        return self.G_adv(I_en)