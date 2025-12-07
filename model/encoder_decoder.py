import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from model.generator import AttackNetwork
from options import HiDDenConfiguration


class EncoderDecoder(nn.Module):
    """
    Combines Encoder->AttackNetwork->Decoder into single pipeline.
    The input is the cover image and the watermark message. The module inserts the watermark into the image
    (obtaining encoded_image), then applies the attack network (obtaining noised_image), then passes the noised_image
    to the Decoder which tries to recover the watermark (called decoded_message). The module outputs
    a three-tuple: (encoded_image, noised_image, decoded_message)
    """
    def __init__(self, config: HiDDenConfiguration, hybrid_distorter=None):

        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(config)
        self.attacker = AttackNetwork()
        self.decoder = Decoder(config)
        self.hybrid_distorter = hybrid_distorter

    def forward(self, image, message):
        encoded_image = self.encoder(image, message)
        # Use hybrid_distorter if provided, otherwise use standard attacker
        if self.hybrid_distorter is not None:
            noised_image = self.hybrid_distorter(encoded_image)
        else:
            noised_image = self.attacker(encoded_image)
        decoded_message = self.decoder(noised_image)
        return encoded_image, noised_image, decoded_message
