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
    def __init__(self, config: HiDDenConfiguration):

        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(config)
        self.attacker = AttackNetwork()
        self.decoder = Decoder(config)

    def forward(self, image, message):
        encoded_image = self.encoder(image, message)
        noised_image = self.attacker(image)
        decoded_message = self.decoder(noised_image)
        return encoded_image, noised_image, decoded_message
