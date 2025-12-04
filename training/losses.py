import torch
import torch.nn as nn
import torch.nn.functional as F

class FFTConsistencyLoss(nn.Module):
    """
    Custom Fourier-domain consistency loss between two images (BxCxHxW).
    L_fft = || w(f) * (log|F(pred)| - log|F(target)|) ||_p
    
    Parameters
    ----------
    p : int
        1 or 2. Norm on spectral diff (L1 or L2)
    
    log_mag : bool
        If true, uses log-magnitude

    alpha : float
        Frequency weighting exponent; 0.0 = uniform, >0 boosts high frequencies,\
        <0 boosts low frequencies
    
    eps : float
        numerical stablitity term (avoiding divide-by-zero type errors)
    """
    def __init__(self, p: int = 1, log_mag: bool = True, alpha: float = 1.0, eps: float = 1e-6):
        super().__init__()
        assert p >= 1
        self.p = p
        self.log_mag = log_mag
        self.alpha = float(alpha)
        self.eps = float(eps)
        self._radial_cache = {}

    def _radial_weights(self, H, W, device, dtype):
        key = (H, W, self.alpha, device, dtype)
        w = self._radial_cache.get(key)
        if w is None:
            fy = torch.fft.fftfreq(H, d=1.0, device=device).view(H, 1)
            fx = torch.fft.fftfreq(W, d=1.0, device=device).view(1, W)
            r = torch.sqrt(fx * fx + fy * fy)
            r = r / r.max().clamp_min(self.eps)
            w = torch.ones((H, W), dtype=dtype, device=device) if self.alpha == 0.0 else r.pow(self.alpha)
            self._radial_cache[key] = w
        return w

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pred.shape
        # device, dtype = pred.device, pred.dtype

        Fx = torch.fft.fft2(pred, dim=(-2, -1), norm="ortho")
        Fy = torch.fft.fft2(target, dim=(-2, -1), norm="ortho")
        Sx = torch.sqrt(Fx.real**2 + Fx.imag**2 + self.eps)
        Sy = torch.sqrt(Fy.real**2 + Fy.imag**2 + self.eps)
        if self.log_mag:
            Sx = torch.log(Sx)
            Sy = torch.log(Sy)

        # For now lets avoid weighting and add it later.
        # w = self._radial_weights(H, W, device, dtype).view(1, 1, H, W)
        D = (Sx - Sy)# * w  # BxCxHxW

        if self.p == 1:
            per = D.abs()
        elif self.p == 2:
            per = D.pow(2)
        else:
            per = D.abs().pow(self.p)

        return (per.view(B, -1).mean(dim=1)).mean()
 
class Image_Loss(nn.Module):
    """Compute the Image Loss as specified in DADW:
    L_I = a_1 * L2(I_co, I_en) + a_2 * L_G(I_en)
    """
    def __init__(
            self,
            alpha1: float,
            alpha2: float
    ):
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
        return self.a1 * self._l2(I_co, I_en) + self.a2 * L_G


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
        return L_M


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


class Watermarking_Loss(nn.Module):
    """Watermarking Loss as specified in DADW:
    L_W = L_I + L_M + a_w * L2(X'_adv, X')
    """
    def __init__(
        self,
        alpha_w: float,
        alpha_m: float,
        alpha1: float,
        alpha2: float,
    ):
        super().__init__()
        self.aw = alpha_w
        self.L_I = Image_Loss(alpha1, alpha2)
        self.L_M = Message_Loss(alpha_m)
    
    @staticmethod
    def _l2(
        x_prime_adv: torch.Tensor,
        x_prime: torch.Tensor
    ) -> torch.Tensor:
        """asdf"""
        return ((x_prime_adv - x_prime)**2).flatten(1).mean(dim=1)

    def forward(
        self,
        I_en: torch.Tensor,
        I_co: torch.Tensor,
        Discriminator_out: torch.Tensor,
        X_prime_dec: torch.Tensor, 
        X_prime: torch.Tensor,
        X_prime_adv: torch.Tensor
    ) -> torch.Tensor:
        """asdf"""
        L_I = self.L_I(I_co, I_en, Discriminator_out)
        L_M = self.L_M(X_prime_dec, X_prime)
        L_W = L_I + L_M + self.aw * self._l2(X_prime_adv, X_prime)
        return L_W.mean()