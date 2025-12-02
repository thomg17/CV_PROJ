import torch 
import torch.nn as nn
import torch.nn.functional as F

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


class Adversarial_Loss(nn.Module):
    """Computes the Adversarial Loss (L_adv) for the Attack Network according to specification in DADW:
    a_1 * L2(I_adv,I_en) - a_2 * L2(F_dec, X)
    
    Parameters
    ----------
    alpha1 : float
        scalar that weights the strength of the distortion from the attack network

    alpha2 : float
        scalar that weights the strength of the message loss' effect on the attack network
    """
    def __init__(
        self, 
        alpha1: float=15.0,
        alpha2: float=1.0
    ):
        super().__init__()
        self.a1 = alpha1
        self.a2 = alpha2
        
        # This i think actually would agg over a batch
        #self.l2_adv = nn.MSELoss()
        #self.l2_msg = nn.MSELoss()

    @staticmethod
    def _l2_adv(
        I_adv: torch.Tensor,
        I_en: torch.Tensor,
    ) -> torch.Tensor:
        """asdf"""
        # [B, C, H, W]
        return ((I_adv - I_en)**2).flatten(1).mean(dim=1)
    
    @staticmethod
    def _l2_msg(
        F_prob: torch.Tensor,
        X: torch.Tensor
    ) -> torch.Tensor:
        """asdf"""
        return ((F_prob - X)**2).mean(dim=1)
    
    def forward(
        self,
        I_adv: torch.Tensor,
        I_en: torch.Tensor,
        F_dec: torch.Tensor,
        X: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        I_adv : torch.Tensor
            Output of the Attack Network (image distorted by the generator)

        I_en : torch.Tensor
            Output of the Watermark Encoder (image encoded w/ message)

        F_dec : torch.Tensor
            Output of the Watermark Decoder (decoded message)

        X : torch.Tensor
            Original input message

        Returns
        -------
        torch.Tensor
            The computed loss
        """
        F_prob = F.sigmoid(F_dec)
        l2_adv = self._l2_adv(I_adv, I_en)
        l2_msg = self._l2_msg(F_prob, X)
        return (self.a1 * l2_adv - self.a2 * l2_msg).mean()