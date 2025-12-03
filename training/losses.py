import torch
import torch.nn as nn
import torch.nn.functional as F

class Image_Loss(nn.Module):
    """Compute the Image Loss as specified in DADW:
    L_I = a_1 * L2(I_co, I_en) + a_2 * L_G(I_en)

    Where L_G is the GAN loss (same as the adversarial loss in HiDDeN)
    """
    def __init__(self, alpha1: float, alpha2: float):
        super().__init__()
        self.a1 = alpha1
        self.a2 = alpha2

    @staticmethod
    def _l2(
        I_co: torch.Tensor,
        I_en: torch.Tensor
    ) -> torch.Tensor:
        """asdf"""
        return ((I_co - I_en)**2).flatten(1).mean(dim=1)

    def forward(
        self, 
        I_co: torch.Tensor, 
        I_en: torch.Tensor,
        Discriminator_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        I_co : torch.Tensor
            The cover image (original) image
        
        I_en : torch.Tensor
            The image encoded with the message

        Discriminator_out: torch.Tensor
            The output of the discriminator 
        """
        L_G = torch.log(1 - Discriminator_out + 1e-12)
        L_I = self.a1 * self._l2(I_co, I_en) + self.a2 * L_G
        return L_I.mean()


class  Message_Loss(nn.Module):
    """Computes the message loss from the output of the decoded message as specified in DADW:
    a_m * L2(X'_dec, X')
    """
    def __init__(
        self,
        alpha_m: float
    ):
        super().__init__()
        self.am = alpha_m
    
    def forward(
        self, 
        X_prime_dec: torch.Tensor, 
        X_prime: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        X_prime_dec : torch.Tensor
            The decoded message output from the channel decoder
        
        X_prime : torch.Tensor
            not sure what this actually is.
        """
        L_M = self.am * ((X_prime_dec - X_prime)**2).flatten(1).mean(dim=1)
        return L_M.mean()


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
            Output of the Watermark Decoder (decoded message also called X'_adv)

        X : torch.Tensor
            Channel encoded message (noted as X')

        Returns
        -------
        torch.Tensor
            The computed loss
        """
        F_prob = F.sigmoid(F_dec)
        l2_adv = self._l2_adv(I_adv, I_en)
        l2_msg = self._l2_msg(F_prob, X)
        return (self.a1 * l2_adv - self.a2 * l2_msg).mean()
